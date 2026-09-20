#!/bin/bash
# Array job body: each task explains one disjoint shard of the fixed
# explainer set for the drug's selected fold, in parallel.
# Submitted via submit_random_split_interpret.sh, which computes the
# --array range from SHARD_COUNT.
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=superpod-a100
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=evo2_interp_rsplit
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/interpretability/random_split/logs/out/%x_%A_%a.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/interpretability/random_split/logs/error/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/logs/out" "${SCRIPT_DIR}/logs/error"
cd "${SCRIPT_DIR}"

DEFAULT_PYTHON="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

DRUG="${DRUG:-AMIKACIN}"
PARAM_FILE="${PARAM_FILE:-${SCRIPT_DIR}/parameter_files/shap_interpret_random_split.yaml}"
MODEL_FILENAME="${MODEL_FILENAME:-auto}"
SHARD_COUNT="${SHARD_COUNT:-4}"

EXTRA_ARGS=()
if [[ -n "${FOLD:-}" ]]; then
  EXTRA_ARGS+=(--fold "${FOLD}")
fi

echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${DRUG} explainer shard $((SLURM_ARRAY_TASK_ID + 1))/${SHARD_COUNT}"

"${PYTHON}" run_interpret_evo2_random_split.py "${PARAM_FILE}" \
  --drug "${DRUG}" \
  --model-filename "${MODEL_FILENAME}" \
  "${EXTRA_ARGS[@]}" \
  --explainer-shard-index "${SLURM_ARRAY_TASK_ID}" \
  --explainer-shard-count "${SHARD_COUNT}"
