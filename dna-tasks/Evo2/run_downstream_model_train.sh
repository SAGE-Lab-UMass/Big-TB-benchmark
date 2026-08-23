#!/bin/bash
#SBATCH -G 1
#SBATCH --gres=gpu:1
# Four DataLoader workers serve large full-token batches. This leaves enough
# host memory for prefetching while allowing multiple jobs per A100 node.
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --job-name=evo2_downstream_train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evo2_env.sh
source "${SCRIPT_DIR}/evo2_env.sh"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DRUG="${DRUG:-AMIKACIN}"
EMBED_TYPE="${EMBED_TYPE:-token}"
MODEL_NAME="${MODEL_NAME:-DNABERTCNN}"
MEMMAP_ROOT="${MEMMAP_ROOT:-${EVO2_DOWNSTREAM_DATA_ROOT}/${EMBED_TYPE}/memmaps}"
PHENOTYPE_LABEL_PATH="${PHENOTYPE_LABEL_PATH:-${EVO2_EMBED_ROOT}/zs_full_stacked_phenotypes.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-${EVO2_DIR}/training_output/zero_shot/classification_results/evo2/${EMBED_TYPE}}"
SAVED_MODEL_PATH="${SAVED_MODEL_PATH:-${EVO2_DIR}/training_output/zero_shot/saved_models/evo2/${EMBED_TYPE}}"
RANDOM_SEED="${RANDOM_SEED:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
if [[ -z "${TRAIN_BATCH_SIZE+x}" ]]; then
    if [[ "${DRUG}" == "CAPREOMYCIN" || "${DRUG}" == "STREPTOMYCIN" ]]; then
        TRAIN_BATCH_SIZE="8"
    else
        TRAIN_BATCH_SIZE="32"
    fi
fi

if [[ -z "${VAL_BATCH_SIZE+x}" ]]; then
    if [[ "${DRUG}" == "CAPREOMYCIN" || "${DRUG}" == "STREPTOMYCIN" ]]; then
        VAL_BATCH_SIZE="8"
    else
        VAL_BATCH_SIZE="32"
    fi
fi
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
TEST_SPLIT="${TEST_SPLIT:-0.2}"
PCA_COMPONENTS="${PCA_COMPONENTS:-10}"
DATA_LOADER_WORKERS="${DATA_LOADER_WORKERS:-0}"
TRAIN_FOLD="${TRAIN_FOLD:-${SLURM_ARRAY_TASK_ID:-}}"

EXTRA_ARGS=()
if [[ -n "${TRAIN_FOLD}" ]]; then
    EXTRA_ARGS+=(--fold "${TRAIN_FOLD}")
fi
if [[ "${SKIP_COMPLETED:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_completed)
fi

evo2_require_executable "${EVO2_TRAIN_PYTHON}"
if [[ "${EVO2_LAUNCH_DRY_RUN:-0}" != "1" ]]; then
    evo2_require_paths "${MEMMAP_ROOT}" "${PHENOTYPE_LABEL_PATH}"
fi

cd "${EVO2_DIR}"

evo2_run "${EVO2_TRAIN_PYTHON}" -m evo2_downstream.train \
    --drug "${DRUG}" \
    --embed_type "${EMBED_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --saved_embed_memmap_dir "${MEMMAP_ROOT}" \
    --phenotype_label_path "${PHENOTYPE_LABEL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --saved_model_path "${SAVED_MODEL_PATH}" \
    --random_seed "${RANDOM_SEED}" \
    --num_epochs "${NUM_EPOCHS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --test_split "${TEST_SPLIT}" \
    --pca_components "${PCA_COMPONENTS}" \
    --data_loader_workers "${DATA_LOADER_WORKERS}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
