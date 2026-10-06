#!/bin/bash
# Convert pre-generated mean_dim/mean_seq embeddings to memmaps for downstream training.
# Submit with sbatch.

#SBATCH -A pi_annagreen_umass_edu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --job-name=convert_mean_embeds_to_memmaps
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB-evo2-optimized/dna-tasks/Evo2/logs/convert_mean/%x_%j.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB-evo2-optimized/dna-tasks/Evo2/logs/convert_mean/%x_%j.err

set -euo pipefail

PROJECT_DIR="/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB-evo2-optimized/dna-tasks/Evo2"
CONDA_ROOT="/work/pi_annagreen_umass_edu/saishradha/miniconda3"
ENV_PREFIX="${CONDA_ROOT}/envs/dnabert_s"

mkdir -p "${PROJECT_DIR}/logs/convert_mean"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Defaults
GENES="${GENES:-all}"
EMBED_TYPES="${EMBED_TYPES:-mean_dim mean_seq}"
MEAN_DIM_RAW_ROOT="${MEAN_DIM_RAW_ROOT:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings/zero-shot/mean_dim/layer20/full}"
MEAN_SEQ_RAW_ROOT="${MEAN_SEQ_RAW_ROOT:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings/zero-shot/mean_seq/layer20/full}"
MEAN_DIM_MEMMAP_ROOT="${MEAN_DIM_MEMMAP_ROOT:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/downstream_inputs/layer20/mean_dim/memmaps}"
MEAN_SEQ_MEMMAP_ROOT="${MEAN_SEQ_MEMMAP_ROOT:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/downstream_inputs/layer20/mean_seq/memmaps}"

echo "[INFO] Converting pre-generated mean embeddings to memmaps"
echo "[INFO] Genes: ${GENES}"
echo "[INFO] Embed types: ${EMBED_TYPES}"
echo "[INFO] Mean dim raw root: ${MEAN_DIM_RAW_ROOT}"
echo "[INFO] Mean seq raw root: ${MEAN_SEQ_RAW_ROOT}"
echo "[INFO] Mean dim output: ${MEAN_DIM_MEMMAP_ROOT}"
echo "[INFO] Mean seq output: ${MEAN_SEQ_MEMMAP_ROOT}"

cd "${PROJECT_DIR}"

python -m evo2_embed_gen.utils.convert_pregenerated_mean_embeds_to_memmaps \
    --genes "${GENES}" \
    --embed_types ${EMBED_TYPES} \
    --mean_dim_raw_embed_root "${MEAN_DIM_RAW_ROOT}" \
    --mean_seq_raw_embed_root "${MEAN_SEQ_RAW_ROOT}" \
    --mean_dim_memmap_root "${MEAN_DIM_MEMMAP_ROOT}" \
    --mean_seq_memmap_root "${MEAN_SEQ_MEMMAP_ROOT}" \
    "$@"

echo "[INFO] Conversion complete"
