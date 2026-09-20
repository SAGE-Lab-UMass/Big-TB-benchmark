#!/bin/bash
# Final step: merge explainer-shard rankings, calculate each lineage's MAP/MAR,
# and create the standard map_mar_<DRUG>_by_lineage.csv / _mean.csv files.
# Lightweight CSV merge only -- no GPU/model work.
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=evo2_interp_combine
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/interpretability/lineage_aware_split/logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/interpretability/lineage_aware_split/logs/error/%x_%J.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/logs/out" "${SCRIPT_DIR}/logs/error"
cd "${SCRIPT_DIR}"

DEFAULT_PYTHON="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

DRUG="${DRUG:-AMIKACIN}"
PARAM_FILE="${PARAM_FILE:-${SCRIPT_DIR}/parameter_files/shap_interpret_lineage_aware.yaml}"
SHARD_COUNT="${SHARD_COUNT:-4}"

"${PYTHON}" combine_lineage_results.py "${PARAM_FILE}" \
  --drug "${DRUG}" \
  --shard-count "${SHARD_COUNT}"
