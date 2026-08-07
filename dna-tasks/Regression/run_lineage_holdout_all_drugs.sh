#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_SH="/work/pi_annagreen_umass_edu/saishradha/miniconda3/etc/profile.d/conda.sh"
CNN_ENV="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/cnn"
PYTHON="$CNN_ENV/bin/python"

BASE_PARAM_FILE="model_training/parameter_files/logreg_iters1000.txt"
TRAINER="lineage_aware_data_split/run_logreg_l2_lineage_holdout.py"
OVERRIDE_OUTPUT_DIR="training_output/lineage_aware_results_logreg_1000"
MIN_CLASS_COUNT="${MIN_CLASS_COUNT:-50}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda initialization script: $CONDA_SH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing CNN environment Python: $PYTHON" >&2
  exit 1
fi

source "$CONDA_SH"
if ! conda activate "$CNN_ENV"; then
  echo "Failed to activate CNN conda environment: $CNN_ENV" >&2
  exit 1
fi

if [[ ! -f "$BASE_PARAM_FILE" ]]; then
  echo "Missing base parameter file: $BASE_PARAM_FILE" >&2
  exit 1
fi

if [[ ! -f "$TRAINER" ]]; then
  echo "Missing lineage-aware trainer: $TRAINER" >&2
  exit 1
fi

readarray -t DRUGS < <(
"$PYTHON" - <<'PY'
from model_training.parameters.locus_order import DRUG_TO_LOCI
for drug in sorted(DRUG_TO_LOCI.keys()):
    print(drug)
PY
)

if [[ ${#DRUGS[@]} -eq 0 ]]; then
  echo "No drugs found in DRUG_TO_LOCI" >&2
  exit 1
fi

echo "Running lineage-aware regression for ${#DRUGS[@]} drugs"
echo "Python: $PYTHON"
echo "Base params: $BASE_PARAM_FILE"
echo "Output dir: $OVERRIDE_OUTPUT_DIR"
echo "Min class count: $MIN_CLASS_COUNT"

FAILURES=()
SUCCESS=()

for DRUG in "${DRUGS[@]}"; do
  echo
  echo "=== [$DRUG] starting ==="

  TMP_PARAM_FILE="$(mktemp "${TMPDIR:-/tmp}/lineage_holdout_${DRUG}_XXXX.yaml")"

  if ! "$PYTHON" - "$BASE_PARAM_FILE" "$TMP_PARAM_FILE" "$DRUG" "$OVERRIDE_OUTPUT_DIR" <<'PY'
import sys
import yaml

base_param_file, out_param_file, drug, output_dir = sys.argv[1:5]

with open(base_param_file, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["drug"] = drug
cfg["output_dir"] = output_dir

with open(out_param_file, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  then
    echo "[$DRUG] failed to prepare parameter file"
    rm -f "$TMP_PARAM_FILE"
    FAILURES+=("$DRUG")
    continue
  fi

  if "$PYTHON" "$TRAINER" "$TMP_PARAM_FILE" "--min-class-count=$MIN_CLASS_COUNT"; then
    echo "=== [$DRUG] done ==="
    SUCCESS+=("$DRUG")
  else
    echo "=== [$DRUG] failed ==="
    FAILURES+=("$DRUG")
  fi

  rm -f "$TMP_PARAM_FILE"
done

echo

echo "Completed lineage-aware runs"
echo "Successful (${#SUCCESS[@]}): ${SUCCESS[*]:-none}"
echo "Failed (${#FAILURES[@]}): ${FAILURES[*]:-none}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  exit 1
fi

exit 0
