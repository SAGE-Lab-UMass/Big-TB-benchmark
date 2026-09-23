#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=superpod-a100
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --job-name=evo2_ftinterp_prep
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

"${PYTHON}" run_interpret_evo2_finetuned.py "${PARAM_FILE}" \
  --drug "${DRUG}" \
  --prep-only
