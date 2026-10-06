#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO2_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_PYTHON="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
PARAM_FILE="${1:-$SCRIPT_DIR/parameter_files/shap_interpret_random_split.yaml}"
MODEL_DIR="$EVO2_DIR/training_output/zero_shot/random_split"
MODEL_SEED="${MODEL_SEED:-42}"
EMBED_TYPE="${EMBED_TYPE:-token}"
RUNNER="$SCRIPT_DIR/run_interpret_evo2_random_split.py"

if [[ ! -f "$PARAM_FILE" ]]; then
  echo "Missing parameter file: $PARAM_FILE" >&2
  exit 1
fi

if ! "$PYTHON" -c 'import shap, torch, yaml, pandas, numpy' 2>/dev/null; then
  echo "Python environment is missing an interpretability dependency: $PYTHON" >&2
  echo "Set PYTHON to the Python executable for the Evo2 training environment." >&2
  exit 1
fi

mapfile -t DRUGS < <(
  find "$MODEL_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort -u
)

if [[ ${#DRUGS[@]} -eq 0 ]]; then
  echo "No drugs found below $MODEL_DIR" >&2
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
