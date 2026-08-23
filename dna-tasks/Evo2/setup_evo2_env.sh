#!/bin/bash
set -euo pipefail

CONDA_ROOT="/work/pi_annagreen_umass_edu/saishradha/miniconda3"
ENV_PREFIX="${CONDA_ROOT}/envs/evo2"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if [ ! -x "${ENV_PREFIX}/bin/python" ] || ! "${ENV_PREFIX}/bin/python" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
    rm -rf "${ENV_PREFIX}"
    conda create -y -p "${ENV_PREFIX}" python=3.12 pip
fi

conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

python - <<'PY'
import torch

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torch_path", torch.__file__)
PY
