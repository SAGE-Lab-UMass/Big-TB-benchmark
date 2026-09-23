#!/bin/bash
# Generate Evo2 embeddings (token/mean_seq/mean_dim) for one gene on Slurm.
# Use SMOKE_TEST=1 for the five-isolate validation profile.
# --partition here is only a default; override with `sbatch --partition=<name>`
# if your account requires a different GPU partition (e.g. a lab-specific one).
#SBATCH --partition=gpu
#SBATCH -G 1                  # Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --job-name=evo2_embeddings

set -euo pipefail

# Slurm copies this script into a spool directory before running it, so
# BASH_SOURCE no longer points at the real file; fall back to the submit dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=evo2_env.sh
source "${SCRIPT_DIR}/evo2_env.sh"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GENE_FILE="${GENE_FILE:-${EVO2_DIR}/ordered_genes.txt}"

# EMBED_TYPE selects the pooling saved by the python entry point; the default
# root mirrors the existing scratch token layout so runs never collide.
EMBED_TYPE="${EMBED_TYPE:-token}"
SCRATCH_EMBED_BASE="${SCRATCH_EMBED_BASE:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings/zero-shot}"
DEFAULT_EMBED_ROOT="${SCRATCH_EMBED_BASE}/${EMBED_TYPE}/layer20"

# The same worker handles both a small smoke test and production array tasks.
# SMOKE_TEST=1 selects conservative defaults without duplicating the Evo2
# command in a second launcher. Every value remains individually overridable.
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    EMBED_ROOT="${EMBED_ROOT:-${DEFAULT_EMBED_ROOT}/smoke}"
    GENE="${GENE:-rpoB}"
    DRUG="${DRUG:-RIFAMPICIN}"
    FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-5}"
    MAX_ISOLATES="${MAX_ISOLATES:-5}"
    RESUME="${RESUME:-0}"
else
    EMBED_ROOT="${EMBED_ROOT:-${DEFAULT_EMBED_ROOT}/full}"
    DRUG="${DRUG:-ALL}"
    FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-2}"
    MAX_ISOLATES="${MAX_ISOLATES:-}"
    RESUME="${RESUME:-1}"
fi

evo2_require_vars EVO2_PHENOTYPE_FILE EVO2_GENOTYPE_INPUT_DIRECTORY
evo2_require_executable "${EVO2_EMBED_PYTHON}"
if [[ "${EVO2_LAUNCH_DRY_RUN:-0}" != "1" ]]; then
    evo2_require_paths "${EVO2_PHENOTYPE_FILE}" "${EVO2_GENOTYPE_INPUT_DIRECTORY}"
    mkdir -p "${EMBED_ROOT}"
    evo2_load_cuda_module
fi

EVO2_HF_TOKEN_FILE="${EVO2_HF_TOKEN_FILE:-${HOME}/.hf_token.env}"
if [[ -r "${EVO2_HF_TOKEN_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${EVO2_HF_TOKEN_FILE}"
fi
if [[ -n "${HF_AUTH_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="${HF_AUTH_TOKEN}"
fi

if [[ -z "${GENE:-}" ]]; then
    evo2_require_paths "${GENE_FILE}"
    mapfile -t GENES < <(awk 'NF' "${GENE_FILE}")
    ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    if (( ARRAY_TASK_ID < 0 || ARRAY_TASK_ID >= ${#GENES[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID} is outside 0..$((${#GENES[@]} - 1))" >&2
        exit 1
    fi
    GENE="${GENES[ARRAY_TASK_ID]}"
fi

EXTRA_ARGS=()
if [[ -n "${MAX_ISOLATES}" ]]; then
    EXTRA_ARGS+=(--max_isolates "${MAX_ISOLATES}")
fi
if [[ "${RESUME}" == "1" ]]; then
    EXTRA_ARGS+=(--resume --resume_validation_batches "${RESUME_VALIDATION_BATCHES:-3}")
fi

OUT_DIR="${EMBED_ROOT}/${GENE}"
echo "Generating ${GENE} embeddings; output directory: ${OUT_DIR}"

cd "${EVO2_DIR}"

evo2_run "${EVO2_EMBED_PYTHON}" -m evo2_embed_gen.embeddings.generate_embeddings \
    --embed_dir "${EMBED_ROOT}" \
    --model_name "evo2_7b" \
    --layer_name "blocks.20.mlp.l3" \
    --max_length 5000 \
    --full_batch_size "${FULL_BATCH_SIZE}" \
    --datapath "${EVO2_DATA_DIR}" \
    --phenotype_file "${EVO2_PHENOTYPE_FILE}" \
    --genotype_input_directory "${EVO2_GENOTYPE_INPUT_DIRECTORY}" \
    --genes "${GENE}" \
    --drug "${DRUG}" \
    --is_single_gene_algo \
    --embed_type "${EMBED_TYPE}" \
    --save_dtype "float16" \
    --stack_phenotypes \
    "${EXTRA_ARGS[@]}" \
    "$@"
