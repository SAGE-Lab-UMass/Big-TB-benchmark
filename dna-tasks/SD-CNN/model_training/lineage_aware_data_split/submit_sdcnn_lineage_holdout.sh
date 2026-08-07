#!/usr/bin/env bash
#SBATCH --job-name=sdcnn_lineage_inh
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	THIS_DIR="${SLURM_SUBMIT_DIR}"
else
	THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
MODEL_TRAINING_DIR="$(cd "${THIS_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${THIS_DIR}/inh_lineage_holdout.yaml}"
HELDOUT_LINEAGE="${HELDOUT_LINEAGE:-1}"
PY_SCRIPT="${THIS_DIR}/run_sdcnn_lineage_holdout.py"

CONDA_SH="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/path/to/miniconda3/envs/cnn}"

mkdir -p "${THIS_DIR}/logs"

if [[ ! -f "${CONDA_SH}" ]]; then
	echo "Set CONDA_SH to your conda.sh path or edit this script: ${CONDA_SH}" >&2
	exit 1
fi

source "${CONDA_SH}"
if ! conda activate "${CONDA_ENV}"; then
	echo "Set CONDA_ENV to your conda environment path or edit this script: ${CONDA_ENV}" >&2
	exit 1
fi
export TF_ENABLE_ONEDNN_OPTS=0

cd "${MODEL_TRAINING_DIR}"

echo "[info] config=${CONFIG_PATH}"
echo "[info] heldout_lineage=${HELDOUT_LINEAGE}"
echo "[info] dry-run split check"
python "${PY_SCRIPT}" "${CONFIG_PATH}" --heldout-lineage "${HELDOUT_LINEAGE}" --dry-run

echo "[info] training run"
python "${PY_SCRIPT}" "${CONFIG_PATH}" --heldout-lineage "${HELDOUT_LINEAGE}"
