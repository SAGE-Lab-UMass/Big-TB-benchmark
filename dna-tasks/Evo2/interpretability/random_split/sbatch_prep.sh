#!/bin/bash
# One-time prep step: builds the full drug dataset, selects the best-val-AUC
# fold (or the requested override), caches dedup indices, and freezes the
# background/explainer split. Run this once (as a SLURM dependency) before
# launching the parallel explainer-shard array job, so concurrent array tasks
# never race on writing the same shared cache files.
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=superpod-a100
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --job-name=evo2_interp_rsplit_prep
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/interpretability/random_split/logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/interpretability/random_split/logs/error/%x_%J.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/logs/out" "${SCRIPT_DIR}/logs/error"
cd "${SCRIPT_DIR}"

DEFAULT_PYTHON="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

DRUG="${DRUG:-AMIKACIN}"
PARAM_FILE="${PARAM_FILE:-${SCRIPT_DIR}/parameter_files/shap_interpret_random_split.yaml}"
MODEL_FILENAME="${MODEL_FILENAME:-auto}"

EXTRA_ARGS=()
if [[ -n "${FOLD:-}" ]]; then
  EXTRA_ARGS+=(--fold "${FOLD}")
fi

"${PYTHON}" run_interpret_evo2_random_split.py "${PARAM_FILE}" \
  --drug "${DRUG}" \
  --model-filename "${MODEL_FILENAME}" \
  "${EXTRA_ARGS[@]}" \
  --prep-only
