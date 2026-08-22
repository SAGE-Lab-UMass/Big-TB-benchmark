#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=superpod-a100
#SBATCH -G 1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=16:00:00

#SBATCH --job-name=evo2_finetune

#SBATCH --mail-user=saishradhamo@umass.edu

#SBATCH --output=../../sbatch_finetune_logs/out/%x_%J.out
#SBATCH --error=../../sbatch_finetune_logs/error/%x_%J.err


set -euo pipefail


# ============================================================================
# Arguments
# ============================================================================

if [[ $# -ne 2 ]]; then
    echo "Usage:"
    echo "  sbatch run_evo2_finetuning_per_lineage.sh <DRUG> <HELDOUT_LINEAGE>"
    echo
    echo "Example:"
    echo "  sbatch run_evo2_finetuning_per_lineage.sh RIFAMPICIN 1"
    echo "  sbatch run_evo2_finetuning_per_lineage.sh AMIKACIN 2"
    exit 1
fi

# Normalize drug name to uppercase
DRUG="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
LINEAGE="$2"

if [[ ! "${LINEAGE}" =~ ^[1-4]$ ]]; then
    echo "ERROR: held-out lineage must be one of: 1 2 3 4"
    exit 1
fi


# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORKDIR="${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"

FINETUNE_OUTPUT_ROOT="${EVO2_DIR}/training_output/lora_finetuned"


# ============================================================================
# Experiment
# ============================================================================

OUTPUT_DIR="${FINETUNE_OUTPUT_ROOT}/${DRUG}/final/heldout_lineage_${LINEAGE}"


# ============================================================================
# Setup
# ============================================================================

cd "${WORKDIR}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${EVO2_DIR}/sbatch_finetune_logs/out" "${EVO2_DIR}/sbatch_finetune_logs/error"

echo "======================================================================"
echo "Evo2-LoRA FINAL FINE-TUNING"
echo "======================================================================"
echo "Drug:                 ${DRUG}"
echo "Held-out lineage:     ${LINEAGE}"
echo "Slurm Job ID:         ${SLURM_JOB_ID}"
echo "Job name:             ${SLURM_JOB_NAME}"
echo "Node:                 ${SLURM_NODELIST}"
echo "CPUs:                 ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "Output directory:     ${OUTPUT_DIR}"
echo "Start time:           $(date)"
echo "======================================================================"

"${PYTHON}" --version

echo
echo "GPU information:"
nvidia-smi

echo
echo "Starting Evo2-LoRA fine-tuning..."


# ============================================================================
# Evo2-LoRA fine-tuning
# ============================================================================

"${PYTHON}" train_evo2_lora.py \
    --drug "${DRUG}" \
    --heldout-lineage "${LINEAGE}" \
    --epochs 10 \
    --min-epochs 3 \
    --patience 3 \
    --min-delta 1e-4 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-lr 1e-4 \
    --classifier-lr 1e-3 \
    --num-workers 4 \
    --resume-from auto \
    --output-dir "${OUTPUT_DIR}"


echo
echo "======================================================================"
echo "Fine-tuning completed successfully"
echo "Drug:             ${DRUG}"
echo "Held-out lineage: ${LINEAGE}"
echo "End time:         $(date)"
echo "Results:          ${OUTPUT_DIR}"
echo "======================================================================"