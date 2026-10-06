#!/bin/bash
#SBATCH --job-name=evo2_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/evo2_lora_%A_%a.out
#SBATCH --error=logs/evo2_lora_%A_%a.err
#SBATCH --array=1-4

# ── Environment setup ─────────────────────────────────────────────────────────
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# Validate the prebuilt environment; training jobs must not mutate it.
"${PYTHON_BIN}" -c 'import peft; from peft import inject_adapter_in_model; print("PEFT", peft.__version__)'

# ── Configuration ─────────────────────────────────────────────────────────────
DRUG="${1:-ISONIAZID}"
HELDOUT_LINEAGE="${SLURM_ARRAY_TASK_ID}"  # 1, 2, 3, or 4

# LoRA hyperparameters
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_LR=1e-4
CLASSIFIER_LR=1e-3

# Training hyperparameters
EPOCHS=30
BATCH_SIZE=4
GRAD_ACCUM=4  # Effective batch size = 16
PATIENCE=5
MIN_EPOCHS=5

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Run training ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "Evo2 LoRA Fine-tuning"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Drug: ${DRUG}"
echo "Heldout lineage: ${HELDOUT_LINEAGE}"
echo "=========================================="
echo ""

cd "${SCRIPT_DIR}"

"${PYTHON_BIN}" train_evo2_lora.py \
    --drug "${DRUG}" \
    --heldout-lineage "${HELDOUT_LINEAGE}" \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-dropout ${LORA_DROPOUT} \
    --lora-lr ${LORA_LR} \
    --classifier-lr ${CLASSIFIER_LR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRAD_ACCUM} \
    --patience ${PATIENCE} \
    --min-epochs ${MIN_EPOCHS} \
    --use-amp \
    --gradient-checkpointing \
    --num-workers 4 \
    --seed 42

echo ""
echo "=========================================="
echo "Training complete"
echo "=========================================="
