#!/usr/bin/env bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --partition=cpu
#SBATCH --job-name=sdcnn_lineage_interp_cpu
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/SD-CNN/interpretability/lineage_aware_split/logs/slurm-%j.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/SD-CNN/interpretability/lineage_aware_split/logs/slurm-%j.err

set -euo pipefail

SCRIPT_DIR="/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/SD-CNN/interpretability/lineage_aware_split"
PARAM_FILE="${1:-/project/pi_annagreen_umass_edu/saishradha/.worktrees/sdcnn-lineage-aware-split-regression/lineage_aware_interpretability_branch.yaml}"
PYTHON="${PYTHON:-/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/cnn/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${PARAM_FILE}" ]]; then
    echo "Parameter file not found: ${PARAM_FILE}" >&2
    exit 1
fi

# Keep TensorFlow and BLAS on the allocated CPU cores; do not initialize CUDA.
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL="1"
export TF_ENABLE_ONEDNN_OPTS="0"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-20}"
export TF_NUM_INTEROP_THREADS="2"
export PYTHONUNBUFFERED="1"

cd "${SCRIPT_DIR}"
echo "[info] host=$(hostname)"
echo "[info] job_id=${SLURM_JOB_ID:-local}"
echo "[info] python=${PYTHON}"
echo "[info] parameter_file=${PARAM_FILE}"
echo "[info] cpu_only=1"

"${SCRIPT_DIR}/run_all_drugs.sh" "${PARAM_FILE}"
