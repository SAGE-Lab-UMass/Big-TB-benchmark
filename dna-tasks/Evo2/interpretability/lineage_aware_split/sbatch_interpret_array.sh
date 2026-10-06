#!/bin/bash
# Array job body: each task runs the lineage-aware Evo2 interpretability
# workflow for exactly one held-out-lineage/explainer-shard pair in parallel.
# Submitted via submit_lineage_aware_interpret.sh, which computes the eligible
# lineage list and the --array range.
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=superpod-a100
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=evo2_interp_lineage
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/interpretability/lineage_aware_split/logs/out/%x_%A_%a.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/interpretability/lineage_aware_split/logs/error/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/logs/out" "${SCRIPT_DIR}/logs/error"
cd "${SCRIPT_DIR}"

DEFAULT_PYTHON="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

DRUG="${DRUG:-AMIKACIN}"
PARAM_FILE="${PARAM_FILE:-${SCRIPT_DIR}/parameter_files/shap_interpret_lineage_aware.yaml}"
MODEL_FILENAME="${MODEL_FILENAME:-DNABERTCNN.pt}"
SHARD_COUNT="${SHARD_COUNT:-4}"

if [[ -z "${LINEAGE_MODEL_LIST_FILE:-}" || ! -f "${LINEAGE_MODEL_LIST_FILE}" ]]; then
  echo "LINEAGE_MODEL_LIST_FILE must contain lineage,checkpoint rows" >&2
  exit 1
fi

LINEAGE_INDEX=$(( SLURM_ARRAY_TASK_ID / SHARD_COUNT ))
SHARD_INDEX=$(( SLURM_ARRAY_TASK_ID % SHARD_COUNT ))
TASK_SPEC="$(sed -n "$((LINEAGE_INDEX + 1))p" "${LINEAGE_MODEL_LIST_FILE}")"
IFS=',' read -r HELDOUT_LINEAGE MODEL_FILENAME <<< "${TASK_SPEC}"
if [[ -z "${HELDOUT_LINEAGE}" || -z "${MODEL_FILENAME}" ]]; then
  echo "Invalid task specification at line $((LINEAGE_INDEX + 1)): ${TASK_SPEC}" >&2
  exit 1
fi
echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${DRUG} heldout_lineage_${HELDOUT_LINEAGE} shard $((SHARD_INDEX + 1))/${SHARD_COUNT}"

"${PYTHON}" run_interpret_evo2_lineage_aware.py "${PARAM_FILE}" \
  --drug "${DRUG}" \
  --heldout-lineage "${HELDOUT_LINEAGE}" \
  --model-filename "${MODEL_FILENAME}" \
  --explainer-shard-index "${SHARD_INDEX}" \
  --explainer-shard-count "${SHARD_COUNT}"
