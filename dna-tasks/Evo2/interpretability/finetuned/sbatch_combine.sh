#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=evo2_ftinterp_combine
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB-evo2-finetuned/dna-tasks/Evo2/interpretability/finetuned/logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB-evo2-finetuned/dna-tasks/Evo2/interpretability/finetuned/logs/error/%x_%J.err

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

"${PYTHON}" combine_finetuned_results.py "${PARAM_FILE}" \
  --drug "${DRUG}" \
  --shard-count "${SHARD_COUNT}"
