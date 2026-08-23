#!/bin/bash
#SBATCH -G 1                  # Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=500G
#SBATCH --time=7:00:00
#SBATCH --job-name=evo2_gene_array

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evo2_env.sh
source "${SCRIPT_DIR}/evo2_env.sh"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EMBED_ROOT="${EMBED_ROOT:-${EVO2_EMBED_ROOT}}"
GENE_FILE="${GENE_FILE:-${EVO2_DIR}/ordered_genes.txt}"

evo2_require_vars EVO2_PHENOTYPE_FILE EVO2_GENOTYPE_INPUT_DIRECTORY
evo2_require_executable "${EVO2_EMBED_PYTHON}"
evo2_require_paths "${GENE_FILE}"
if [[ "${EVO2_LAUNCH_DRY_RUN:-0}" != "1" ]]; then
    evo2_require_paths "${EVO2_PHENOTYPE_FILE}" "${EVO2_GENOTYPE_INPUT_DIRECTORY}"
    mkdir -p "${EMBED_ROOT}"
    evo2_load_cuda_module
fi

if [ -f "${HOME}/.hf_token.env" ]; then
    source "${HOME}/.hf_token.env"
fi
if [ -n "${HF_AUTH_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="${HF_AUTH_TOKEN}"
fi

if [[ -z "${GENE:-}" ]]; then
    mapfile -t GENES < "${GENE_FILE}"
    ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    if (( ARRAY_TASK_ID < 0 || ARRAY_TASK_ID >= ${#GENES[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID} is outside 0..$((${#GENES[@]} - 1))" >&2
        exit 1
    fi
    GENE="${GENES[ARRAY_TASK_ID]}"
fi

OUT_DIR="${EMBED_ROOT}/${GENE}"
echo "Generating/resuming ${GENE}; output directory: ${OUT_DIR}"

cd "${EVO2_DIR}"

evo2_run "${EVO2_EMBED_PYTHON}" -m evo2_embed_gen.embeddings.generate_embeddings \
    --embed_dir "${EMBED_ROOT}" \
    --model_name "evo2_7b" \
    --layer_name "blocks.20.mlp.l3" \
    --max_length 5000 \
    --full_batch_size 2 \
    --datapath "${EVO2_DATA_DIR}" \
    --phenotype_file "${EVO2_PHENOTYPE_FILE}" \
    --genotype_input_directory "${EVO2_GENOTYPE_INPUT_DIRECTORY}" \
    --genes "${GENE}" \
    --drug "ALL" \
    --is_single_gene_algo \
    --embed_type "token" \
    --save_dtype "float16" \
    --resume \
    --resume_validation_batches 3 \
    --stack_phenotypes \
    "$@"
