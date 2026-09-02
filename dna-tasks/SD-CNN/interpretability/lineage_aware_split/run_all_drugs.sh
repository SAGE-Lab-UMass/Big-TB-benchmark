#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/cnn/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
PARAM_FILE="${1:-$SCRIPT_DIR/parameter_files/lineage_aware_interpretability.yaml}"
ELIGIBILITY_FILE="/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/SD-CNN/model_training/training_output/lineage_aware_split/all_drugs_split_counts.csv"
RUNNER="$SCRIPT_DIR/run_interpret_lineage_aware.py"

if [[ ! -f "$PARAM_FILE" ]]; then
  echo "Missing parameter file: $PARAM_FILE" >&2
  exit 1
fi
if [[ ! -f "$ELIGIBILITY_FILE" ]]; then
  echo "Missing eligibility file: $ELIGIBILITY_FILE" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import h5py, matplotlib, numpy, pandas, shap, sparse, tensorflow, yaml' 2>/dev/null; then
  echo "Python environment is missing an SD-CNN interpretability dependency: $PYTHON" >&2
  echo "Set PYTHON to the SD-CNN environment's Python executable." >&2
  exit 1
fi

mapfile -t DRUGS < <(
  awk -F, 'NR > 1 && tolower($6) == "true" {print toupper($1)}' "$ELIGIBILITY_FILE" \
    | sort -u
)
if [[ ${#DRUGS[@]} -eq 0 ]]; then
  echo "No eligible drug/lineage rows found in $ELIGIBILITY_FILE" >&2
  exit 1
fi

failures=()
for drug in "${DRUGS[@]}"; do
  echo "=== $drug ==="
  if ! "$PYTHON" "$RUNNER" "$PARAM_FILE" --drug "$drug"; then
    failures+=("$drug")
  fi
done

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "Failed drugs: ${failures[*]}" >&2
  exit 1
fi
echo "Completed ${#DRUGS[@]} drugs: ${DRUGS[*]}"
