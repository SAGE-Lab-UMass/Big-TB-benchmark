#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=superpod-a100
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=500G
#SBATCH --time=7:00:00
#SBATCH --mail-user=saishradhamo@umass.edu
#SBATCH --job-name=evo2_embed_gen
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/sbatch_embed_gen_logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/sbatch_embed_gen_logs/error/%x_%J.err

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

PROJECT_DIR="/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2"
CONDA_ROOT="/work/pi_annagreen_umass_edu/saishradha/miniconda3"
ENV_PREFIX="${CONDA_ROOT}/envs/evo2"

mkdir -p "${PROJECT_DIR}/sbatch_embed_gen_logs/out" "${PROJECT_DIR}/sbatch_embed_gen_logs/error"
mkdir -p "/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings"

if [ -f "${HOME}/.hf_token.env" ]; then
    source "${HOME}/.hf_token.env"
fi
if [ -n "${HF_AUTH_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="${HF_AUTH_TOKEN}"
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
module load cuda/12.8
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    bash "${PROJECT_DIR}/setup_evo2_env.sh"
fi
conda activate "${ENV_PREFIX}"
if ! python -m pip show torch evo2 flash-attn huggingface_hub >/dev/null 2>&1; then
    bash "${PROJECT_DIR}/setup_evo2_env.sh"
fi

cd "${PROJECT_DIR}"

python -m evo2_embed_gen.embeddings.generate_embeddings \
    --embed_dir "/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings/zero-shot/token/train/new" \
    --model_name "evo2_7b" \
    --layer_name "blocks.20.mlp.l3" \
    --max_length 5000 \
    --full_batch_size 5 \
    --genotype_input_directory "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/genomic_data/aligned" \
    --genes "rpoB" \
    --drug "RIFAMPICIN" \
    --is_single_gene_algo \
    --embed_type "token" \
    --save_dtype "float16" \
    --max_isolates 5 \
    --stack_phenotypes
