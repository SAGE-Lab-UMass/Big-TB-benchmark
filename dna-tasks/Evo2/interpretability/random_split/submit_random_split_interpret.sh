#!/usr/bin/env bash
# Select the best-val-AUC fold (or a requested override) for a drug and launch:
#   1. sbatch_prep.sh             (one job; picks the fold, caches dedup indices
#                                   + background/explainer split + relevant
#                                   features)
#   2. sbatch_interpret_array.sh  (one array task per explainer shard, in
#                                  parallel, dependent on #1)
#   3. sbatch_combine.sh          (merges shard rankings + computes MAP/MAR,
#                                  dependent on #2)
#
# Usage:
#   ./submit_random_split_interpret.sh AMIKACIN
#   FOLD=3 ./submit_random_split_interpret.sh AMIKACIN   # override fold selection
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRUG="${1:-${DRUG:-AMIKACIN}}"
DRUG="$(echo "${DRUG}" | tr '[:lower:]' '[:upper:]')"
PARAM_FILE="${PARAM_FILE:-${SCRIPT_DIR}/parameter_files/shap_interpret_random_split.yaml}"
MODEL_FILENAME="${MODEL_FILENAME:-auto}"
SHARD_COUNT="${SHARD_COUNT:-4}"
FOLD="${FOLD:-}"

mkdir -p "${SCRIPT_DIR}/logs/out" "${SCRIPT_DIR}/logs/error"

PREP_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" MODEL_FILENAME="${MODEL_FILENAME}" FOLD="${FOLD}" \
  sbatch --parsable --job-name="evo2rsprep_${DRUG}" \
  "${SCRIPT_DIR}/sbatch_prep.sh")
echo "Submitted prep job: ${PREP_JOB_ID}"

ARRAY_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" MODEL_FILENAME="${MODEL_FILENAME}" SHARD_COUNT="${SHARD_COUNT}" FOLD="${FOLD}" \
  sbatch --parsable --dependency=afterok:"${PREP_JOB_ID}" \
  --job-name="evo2rsshap_${DRUG}" \
  --array=0-$(( SHARD_COUNT - 1 )) \
  "${SCRIPT_DIR}/sbatch_interpret_array.sh")
echo "Submitted interpretability array job: ${ARRAY_JOB_ID} (${SHARD_COUNT} explainer shard task(s))"

COMBINE_JOB_ID=$(DRUG="${DRUG}" PARAM_FILE="${PARAM_FILE}" SHARD_COUNT="${SHARD_COUNT}" \
  sbatch --parsable --dependency=afterok:"${ARRAY_JOB_ID}" \
  --job-name="evo2rsmerge_${DRUG}" \
  "${SCRIPT_DIR}/sbatch_combine.sh")
echo "Submitted combine job: ${COMBINE_JOB_ID} (merges into map_mar_${DRUG}.csv)"
