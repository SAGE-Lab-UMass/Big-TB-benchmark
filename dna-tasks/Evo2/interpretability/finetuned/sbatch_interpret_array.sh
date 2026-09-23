#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=superpod-a100
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=evo2_ftinterp
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB-evo2-finetuned/dna-tasks/Evo2/interpretability/finetuned/logs/out/%x_%A_%a.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB-evo2-finetuned/dna-tasks/Evo2/interpretability/finetuned/logs/error/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/logs/out" "${SCRIPT_DIR}/logs/error"
cd "${SCRIPT_DIR}"

DEFAULT_PYTHON="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/evo2/bin/python"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

EVO2_TASK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${EVO2_TASK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DRUG="${DRUG:-ISONIAZID}"
PARAM_FILE="${PARAM_FILE:-${SCRIPT_DIR}/parameter_files/shap_interpret_finetuned.yaml}"
SHARD_COUNT="${SHARD_COUNT:-4}"

if [[ -z "${LINEAGE_LIST_FILE:-}" || ! -f "${LINEAGE_LIST_FILE}" ]]; then
  echo "LINEAGE_LIST_FILE must contain one eligible lineage per line" >&2
  exit 1
fi
mapfile -t LINEAGES < "${LINEAGE_LIST_FILE}"
LINEAGE_INDEX=$(( SLURM_ARRAY_TASK_ID / SHARD_COUNT ))
SHARD_INDEX=$(( SLURM_ARRAY_TASK_ID % SHARD_COUNT ))
HELDOUT_LINEAGE="${LINEAGES[${LINEAGE_INDEX}]}"

echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${DRUG} heldout_lineage_${HELDOUT_LINEAGE} shard $((SHARD_INDEX + 1))/${SHARD_COUNT}"

"${PYTHON}" run_interpret_evo2_finetuned.py "${PARAM_FILE}" \
  --drug "${DRUG}" \
  --heldout-lineage "${HELDOUT_LINEAGE}" \
  --explainer-shard-index "${SHARD_INDEX}" \
  --explainer-shard-count "${SHARD_COUNT}"
