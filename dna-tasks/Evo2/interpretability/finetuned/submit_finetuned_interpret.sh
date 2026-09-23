#!/usr/bin/env bash
# Discover eligible finetuned held-out lineages for a drug and launch:
#   1. sbatch_prep.sh             (selects fixed explainer + caches features)
#   2. sbatch_interpret_array.sh  (one task per lineage/shard pair)
#   3. sbatch_combine.sh          (merges shards and MAP/MAR summaries)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRUG="${1:-${DRUG:-ISONIAZID}}"
DRUG="$(echo "${DRUG}" | tr '[:lower:]' '[:upper:]')"
PARAM_FILE="${PARAM_FILE:-${SCRIPT_DIR}/parameter_files/shap_interpret_finetuned.yaml}"
MODEL_DIR="${MODEL_DIR:-/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/training_output/lora_finetuned}"
SHARD_COUNT="${SHARD_COUNT:-4}"
REQUIRE_TEST_METRICS="${REQUIRE_TEST_METRICS:-1}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Missing parameter file: ${PARAM_FILE}" >&2
  exit 1
fi
if (( SHARD_COUNT < 1 )); then
  echo "SHARD_COUNT must be positive" >&2
  exit 1
fi

ELIGIBLE=()
for lineage_dir in "${MODEL_DIR}/${DRUG}/final"/heldout_lineage_*; do
  [[ -d "${lineage_dir}" ]] || continue
  lineage="$(basename "${lineage_dir}" | sed 's/^heldout_lineage_//')"
  checkpoint_dir="${lineage_dir}/best"
  if [[ ! -f "${checkpoint_dir}/training_config.json" || ! -f "${checkpoint_dir}/lora_adapter.pt" || ! -f "${checkpoint_dir}/classifier_head.pt" ]]; then
    continue
  fi
  if [[ "${REQUIRE_TEST_METRICS}" == "1" && ! -f "${lineage_dir}/test_metrics.json" ]]; then
    continue
  fi
  ELIGIBLE+=("${lineage}")
done

if [[ ${#ELIGIBLE[@]} -eq 0 ]]; then
  echo "No eligible finetuned held-out lineages for ${DRUG} below ${MODEL_DIR}/${DRUG}/final" >&2
  exit 1
fi

printf 'Eligible lineages for %s: %s\n' "${DRUG}" "${ELIGIBLE[*]}"
mkdir -p "${SCRIPT_DIR}/logs/out" "${SCRIPT_DIR}/logs/error"
LINEAGE_LIST_FILE="${SCRIPT_DIR}/logs/${DRUG}_eligible_lineages_finetuned.txt"
printf '%s\n' "${ELIGIBLE[@]}" > "${LINEAGE_LIST_FILE}"

PREP_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" \
  sbatch --parsable --job-name="evo2ftprep_${DRUG}" "${SCRIPT_DIR}/sbatch_prep.sh")
echo "Submitted prep job: ${PREP_JOB_ID}"

ARRAY_TASK_COUNT=$(( ${#ELIGIBLE[@]} * SHARD_COUNT ))
ARRAY_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" SHARD_COUNT="${SHARD_COUNT}" LINEAGE_LIST_FILE="${LINEAGE_LIST_FILE}" \
  sbatch --parsable --dependency=afterok:"${PREP_JOB_ID}" \
  --job-name="evo2ftshap_${DRUG}" \
  --array=0-$(( ARRAY_TASK_COUNT - 1 )) \
  "${SCRIPT_DIR}/sbatch_interpret_array.sh")
echo "Submitted interpretability array job: ${ARRAY_JOB_ID} (${#ELIGIBLE[@]} lineages x ${SHARD_COUNT} shards = ${ARRAY_TASK_COUNT} tasks)"

COMBINE_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" SHARD_COUNT="${SHARD_COUNT}" \
  sbatch --parsable --dependency=afterok:"${ARRAY_JOB_ID}" \
  --job-name="evo2ftmerge_${DRUG}" \
  "${SCRIPT_DIR}/sbatch_combine.sh")
echo "Submitted combine job: ${COMBINE_JOB_ID} (merges into map_mar_${DRUG}_by_lineage.csv / _mean.csv)"
