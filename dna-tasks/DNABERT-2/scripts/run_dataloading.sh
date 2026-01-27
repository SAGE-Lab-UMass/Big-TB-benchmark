#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu   # Account
#SBATCH --partition=superpod-a100
#SBATCH -G 1                  # Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=600G
#SBATCH --time=9:00:00
#SBATCH --mail-user=saishradhamo@umass.edu
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_dataloading_logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_dataloading_logs/error/%x_%J.err



# Enable better CUDA memory handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ====================================
# Hugging Face setup
# ====================================

# Load Hugging Face token securely from a hidden env file
source ~/.hf_token.env

# ====================================
# Python environment setup
# ====================================

# Activate your conda environment (uncomment if needed)
source /work/pi_annagreen_umass_edu/saishradha/miniconda3/etc/profile.d/conda.sh
conda activate /work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s

# Load CUDA modules if your cluster requires it (uncomment if needed)
# module load cuda/11.8.2 cudnn/8.7.0.84-11.8 

# ====================================
# Script execution
# ====================================
# Run zero shot inference
dnabert_file="../train/zero_shot/dataloading.py"

# Run zero shot inference with separated genes

# for full token embeddings
# python "$dnabert_file" \
#   --embed_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new' \
#   --memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps' \
#   --gene "rrl" \
#   --embed_method "zs" \
#   --embed_name_prefix "full"

# for mean seq reduction
# python "$dnabert_file" \
#   --memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps' \
#   --mean_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_seq/full/memmaps' \
#   --gene "eis" \
#   --embed_type "mean_seq" \
#   --embed_method "zs" \
#   --embed_name_prefix "full"


# for mean dim reduction
python "$dnabert_file" \
  --memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps' \
  --mean_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_dim/full/memmaps' \
  --gene "eis" \
  --embed_type "mean_dim" \
  --embed_method "zs" \
  --embed_name_prefix "full"