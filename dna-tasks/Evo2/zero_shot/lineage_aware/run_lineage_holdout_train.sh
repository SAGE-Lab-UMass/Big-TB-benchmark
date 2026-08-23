#!/bin/bash
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=16:00:00
#SBATCH --job-name=evo2_lineage_train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../evo2_env.sh
source "${EVO2_DIR}/evo2_env.sh"
LINEAGE_SPLIT_DIR="${SCRIPT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── configurable via env vars ─────────────────────────────────────────────────
DRUG="${DRUG:-ISONIAZID}"
HELDOUT_LINEAGE="${HELDOUT_LINEAGE:-1}"
EMBED_TYPE="${EMBED_TYPE:-token}"
MODEL_NAME="${MODEL_NAME:-DNABERTCNN}"

MEMMAP_ROOT="${MEMMAP_ROOT:-${EVO2_DOWNSTREAM_DATA_ROOT}/${EMBED_TYPE}/memmaps}"
PHENOTYPE_LABEL_PATH="${PHENOTYPE_LABEL_PATH:-${EVO2_EMBED_ROOT}/zs_full_stacked_phenotypes.npz}"

# Output paths default to training_output/zero_shot/lineage_aware_holdout/<drug>/
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVO2_DIR}/training_output/zero_shot/lineage_aware_holdout}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${DRUG}/classification_results/evo2/${EMBED_TYPE}}"
SAVED_MODEL_PATH="${SAVED_MODEL_PATH:-${OUTPUT_ROOT}/${DRUG}/saved_models/evo2/${EMBED_TYPE}}"

RANDOM_SEED="${RANDOM_SEED:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"

# Optimized batch sizes for better A100 GPU utilization (increased from 32/8)
# Note: If OOM errors occur for multi-gene drugs, reduce these values
if [[ -z "${TRAIN_BATCH_SIZE+x}" ]]; then
    if [[ "${DRUG}" == "CAPREOMYCIN" || "${DRUG}" == "STREPTOMYCIN" ]]; then
        TRAIN_BATCH_SIZE="32"
    elif [[ "${DRUG}" == "ETHIONAMIDE" || "${DRUG}" == "AMIKACIN" || "${DRUG}" == "MOXIFLOXACIN" ]]; then
        # High-memory multi-gene drugs: ultra-conservative to avoid OOM
        TRAIN_BATCH_SIZE="32"
    elif [[ "${DRUG}" == "ISONIAZID" || "${DRUG}" == "RIFAMPICIN" || "${DRUG}" == "ETHAMBUTOL" ]]; then
        # Multi-gene drugs: conservative batch size to avoid OOM
        TRAIN_BATCH_SIZE="64"
    else
        TRAIN_BATCH_SIZE="128"
    fi
fi
if [[ -z "${VAL_BATCH_SIZE+x}" ]]; then
    if [[ "${DRUG}" == "CAPREOMYCIN" || "${DRUG}" == "STREPTOMYCIN" ]]; then
        VAL_BATCH_SIZE="32"
    elif [[ "${DRUG}" == "ETHIONAMIDE" || "${DRUG}" == "AMIKACIN" || "${DRUG}" == "MOXIFLOXACIN" ]]; then
        # High-memory multi-gene drugs: ultra-conservative to avoid OOM
        VAL_BATCH_SIZE="32"
    elif [[ "${DRUG}" == "ISONIAZID" || "${DRUG}" == "RIFAMPICIN" || "${DRUG}" == "ETHAMBUTOL" ]]; then
        # Multi-gene drugs: conservative batch size to avoid OOM
        VAL_BATCH_SIZE="64"
    else
        VAL_BATCH_SIZE="128"
    fi
fi

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# Early stopping parameters (training-loss based, matching SD-CNN defaults)
EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT="${EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT:-1e-3}"
EARLY_STOPPING_SMOOTHING_WINDOW="${EARLY_STOPPING_SMOOTHING_WINDOW:-3}"

# Optimized DataLoader workers: reduced from 6 to 4 to leave more CPU for main process
# Further reduced for high-memory drugs to minimize prefetch overhead
if [[ -z "${DATA_LOADER_WORKERS+x}" ]]; then
    if [[ "${DRUG}" == "ETHIONAMIDE" || "${DRUG}" == "AMIKACIN" || "${DRUG}" == "MOXIFLOXACIN" ]]; then
        # High-memory drugs: minimal workers to reduce memory pressure
        DATA_LOADER_WORKERS="1"
    else
        SLURM_CPUS="${SLURM_CPUS_PER_TASK:-8}"
        if [[ "${SLURM_CPUS}" -ge 8 ]]; then
            DATA_LOADER_WORKERS="4"
        elif [[ "${SLURM_CPUS}" -ge 6 ]]; then
            DATA_LOADER_WORKERS="3"
        elif [[ "${SLURM_CPUS}" -ge 4 ]]; then
            DATA_LOADER_WORKERS="2"
        else
            DATA_LOADER_WORKERS="1"
        fi
    fi
fi
TRAIN_FOLD="${TRAIN_FOLD:-${SLURM_ARRAY_TASK_ID:-}}"

EXTRA_ARGS=()
if [[ -n "${TRAIN_FOLD}" ]]; then
    EXTRA_ARGS+=(--fold "${TRAIN_FOLD}")
fi
if [[ "${SKIP_COMPLETED:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_completed)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
if [[ "${USE_AUC_EARLY_STOPPING:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--use_validation_early_stopping)
fi

echo "[runner] DRUG=${DRUG} HELDOUT_LINEAGE=${HELDOUT_LINEAGE} DATA_LOADER_WORKERS=${DATA_LOADER_WORKERS}"
echo "[runner] Early stopping: min_epochs=${EARLY_STOPPING_MIN_EPOCHS} patience=${EARLY_STOPPING_PATIENCE}"
echo "[runner] AUC early stopping: ${USE_AUC_EARLY_STOPPING:-0}"

evo2_require_executable "${EVO2_TRAIN_PYTHON}"
if [[ "${EVO2_LAUNCH_DRY_RUN:-0}" != "1" ]]; then
    evo2_require_paths \
        "${MEMMAP_ROOT}" \
        "${PHENOTYPE_LABEL_PATH}" \
        "${EVO2_GENO_PHENO_CSV}" \
        "${EVO2_LINEAGE_CSV}"
fi

cd "${EVO2_DIR}"

evo2_run "${EVO2_TRAIN_PYTHON}" "${LINEAGE_SPLIT_DIR}/train_lineage_holdout.py" \
    --drug "${DRUG}" \
    --heldout-lineage "${HELDOUT_LINEAGE}" \
    --embed_type "${EMBED_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --saved_embed_memmap_dir "${MEMMAP_ROOT}" \
    --phenotype_label_path "${PHENOTYPE_LABEL_PATH}" \
    --geno-pheno-csv "${EVO2_GENO_PHENO_CSV}" \
    --lineage-csv "${EVO2_LINEAGE_CSV}" \
    --output_path "${OUTPUT_PATH}" \
    --saved_model_path "${SAVED_MODEL_PATH}" \
    --random_seed "${RANDOM_SEED}" \
    --num_epochs "${NUM_EPOCHS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --data_loader_workers "${DATA_LOADER_WORKERS}" \
    --early_stopping_min_epochs "${EARLY_STOPPING_MIN_EPOCHS}" \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
    --early_stopping_min_relative_improvement "${EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT}" \
    --early_stopping_smoothing_window "${EARLY_STOPPING_SMOOTHING_WINDOW}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
