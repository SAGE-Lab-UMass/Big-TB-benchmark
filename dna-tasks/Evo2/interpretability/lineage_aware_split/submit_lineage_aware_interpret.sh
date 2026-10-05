#!/usr/bin/env bash
# Discover every eligible held-out lineage (seed 42) for a drug and launch:
#   1. sbatch_prep.sh        (one job; caches dedup indices + relevant features)
#   2. sbatch_interpret_array.sh  (one array task per lineage/explainer shard,
#                                  dependent on #1)
#   3. sbatch_combine.sh      (merges shards and lineage outputs, dependent on #2)
#
# Usage:
#   ./submit_lineage_aware_interpret.sh AMIKACIN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DRUG="${1:-${DRUG:-AMIKACIN}}"
DRUG="$(echo "${DRUG}" | tr '[:lower:]' '[:upper:]')"
SHARD_COUNT="${SHARD_COUNT:-4}"
PARAM_FILE="${PARAM_FILE:-${SCRIPT_DIR}/parameter_files/shap_interpret_lineage_aware.yaml}"

DEFAULT_PYTHON="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

yaml_value() {
  local key="$1"
  "${PYTHON}" - "${PARAM_FILE}" "${key}" <<'PY'
import sys
from pathlib import Path
import yaml

path, key = sys.argv[1:]
with Path(path).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
value = config.get(key, "")
if value is None:
    value = ""
print(value)
PY
}

EMBED_TYPE="${EMBED_TYPE:-$(yaml_value embed_type)}"
EMBED_TYPE="${EMBED_TYPE:-token}"
MODEL_FILENAME="${MODEL_FILENAME:-$(yaml_value model_filename)}"
MODEL_FILENAME="${MODEL_FILENAME:-auto}"
MODEL_SEED="${MODEL_SEED:-$(yaml_value model_seed)}"
MODEL_SEED="${MODEL_SEED:-42}"
MODEL_DIR="${MODEL_DIR:-$(yaml_value model_dir)}"
MODEL_DIR="${MODEL_DIR:-/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/training_output/lineage_aware_holdout}"

SAVED_MODELS_DIR="${MODEL_DIR}/${DRUG}/saved_models/evo2/${EMBED_TYPE}"
CLASSIFICATION_DIR="${MODEL_DIR}/${DRUG}/classification_results/evo2/${EMBED_TYPE}"

if [[ ! -d "${SAVED_MODELS_DIR}" ]]; then
  echo "No trained models found for ${DRUG}: ${SAVED_MODELS_DIR}" >&2
  exit 1
fi

ELIGIBLE=()
ELIGIBLE_MODELS=()
for lineage_dir in "${SAVED_MODELS_DIR}"/heldout_lineage_*; do
  [[ -d "${lineage_dir}" ]] || continue
  lineage="$(basename "${lineage_dir}" | sed 's/^heldout_lineage_//')"
  auc_path="${CLASSIFICATION_DIR}/heldout_lineage_${lineage}/${DRUG}/seed_${MODEL_SEED}/test_set_auc_${DRUG}.csv"
  if [[ "${MODEL_FILENAME}" == "auto" ]]; then
    [[ -f "${auc_path}" ]] || continue
    evaluated_algorithm="$(tail -n 1 "${auc_path}" | cut -d, -f1)"
    evaluated_model="${evaluated_algorithm#Evo2-}.pt"
  else
    evaluated_model="${MODEL_FILENAME}"
  fi
  model_path="${lineage_dir}/${DRUG}/seed_${MODEL_SEED}/${evaluated_model}"
  if [[ -f "${model_path}" && -f "${auc_path}" ]]; then
    ELIGIBLE+=("${lineage}")
    ELIGIBLE_MODELS+=("${evaluated_model}")
  fi
done

if [[ ${#ELIGIBLE[@]} -eq 0 ]]; then
  echo "No eligible held-out lineages for ${DRUG} at seed ${MODEL_SEED} below ${SAVED_MODELS_DIR}" >&2
  exit 1
fi
if (( SHARD_COUNT < 1 )); then
  echo "SHARD_COUNT must be positive" >&2
  exit 1
fi

echo "Eligible lineages for ${DRUG} (seed ${MODEL_SEED}): ${ELIGIBLE[*]}"
for index in "${!ELIGIBLE[@]}"; do
  echo "  lineage ${ELIGIBLE[${index}]} -> ${ELIGIBLE_MODELS[${index}]}"
done

mkdir -p "${SCRIPT_DIR}/logs/out" "${SCRIPT_DIR}/logs/error"
LINEAGE_LIST_FILE="${SCRIPT_DIR}/logs/${DRUG}_eligible_lineages_seed${MODEL_SEED}.txt"
printf '%s\n' "${ELIGIBLE[@]}" > "${LINEAGE_LIST_FILE}"
LINEAGE_MODEL_LIST_FILE="${SCRIPT_DIR}/logs/${DRUG}_lineage_models_seed${MODEL_SEED}.csv"
for index in "${!ELIGIBLE[@]}"; do
  printf '%s,%s\n' "${ELIGIBLE[${index}]}" "${ELIGIBLE_MODELS[${index}]}"
done > "${LINEAGE_MODEL_LIST_FILE}"

PREP_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" MODEL_FILENAME="${ELIGIBLE_MODELS[0]}" \
  sbatch --parsable --job-name="evo2prep_${DRUG}" \
  "${SCRIPT_DIR}/sbatch_prep.sh")
echo "Submitted prep job: ${PREP_JOB_ID}"

ARRAY_TASK_COUNT=$(( ${#ELIGIBLE[@]} * SHARD_COUNT ))
ARRAY_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" SHARD_COUNT="${SHARD_COUNT}" LINEAGE_MODEL_LIST_FILE="${LINEAGE_MODEL_LIST_FILE}" \
  sbatch --parsable --dependency=afterok:"${PREP_JOB_ID}" \
  --job-name="evo2shap_${DRUG}" \
  --array=0-$(( ARRAY_TASK_COUNT - 1 )) \
  "${SCRIPT_DIR}/sbatch_interpret_array.sh")
echo "Submitted interpretability array job: ${ARRAY_JOB_ID} (${#ELIGIBLE[@]} lineages x ${SHARD_COUNT} shards = ${ARRAY_TASK_COUNT} tasks)"

COMBINE_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" SHARD_COUNT="${SHARD_COUNT}" \
  sbatch --parsable --dependency=afterok:"${ARRAY_JOB_ID}" \
  --job-name="evo2merge_${DRUG}" \
  "${SCRIPT_DIR}/sbatch_combine.sh")
echo "Submitted combine job: ${COMBINE_JOB_ID} (merges into map_mar_${DRUG}_by_lineage.csv / _mean.csv)"
