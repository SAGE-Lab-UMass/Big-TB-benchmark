#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TRAINING_DIR="$(cd "${THIS_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${THIS_DIR}/inh_lineage_holdout.yaml}"
PY_SCRIPT="${THIS_DIR}/run_sdcnn_lineage_holdout.py"

CONDA_SH="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/path/to/miniconda3/envs/cnn}"

LOG_DIR="${THIS_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sdcnn_lineage_holdout_$(date +%Y%m%d_%H%M%S).log"

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

echo "[info] config=${CONFIG_PATH}" | tee -a "${LOG_FILE}"
echo "[info] dry-run split checks for INH lineages 1 and 2" | tee -a "${LOG_FILE}"
python "${PY_SCRIPT}" "${CONFIG_PATH}" --heldout-lineage 1 --dry-run | tee -a "${LOG_FILE}"
python "${PY_SCRIPT}" "${CONFIG_PATH}" --heldout-lineage 2 --dry-run | tee -a "${LOG_FILE}"

echo "[info] training for INH heldout lineage 1 using config N_epochs" | tee -a "${LOG_FILE}"
python "${PY_SCRIPT}" "${CONFIG_PATH}" --heldout-lineage 1 | tee -a "${LOG_FILE}"

echo "[done] completed. log=${LOG_FILE}" | tee -a "${LOG_FILE}"
