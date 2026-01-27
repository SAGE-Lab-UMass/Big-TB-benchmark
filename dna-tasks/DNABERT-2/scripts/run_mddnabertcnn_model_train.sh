#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu   # Account
#SBATCH --partition=superpod-a100
#SBATCH -G 1                  # Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=600G
#SBATCH --time=10:00:00
#SBATCH --mail-user=saishradhamo@umass.edu
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_train_logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_train_logs/error/%x_%J.err



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

dnabert_s_file="../train/zero_shot/resistance_classification_train.py"

# Run with DNABert-S zs embeddings for multi drug model (mean seq method)
# python $dnabert_s_file --saved_embed_dir /training_output/transfer_learn/embeddings_5000 --output_path 'training_output/zero_shot/classification_results/dnabertS' --saved_model_path 'training_output/zero_shot/saved_models/dnabertS'

# Run with DNABert-2 zs embeddings for multi drug model (mean seq method)
# python $dnabert_s_file --saved_embed_dir /training_output/zero_shot/embeddings_5000/dnabert2 --output_path 'training_output/zero_shot/classification_results/dnabert2' --saved_model_path 'training_output/zero_shot/saved_models/dnabert2'

# Run with DNABert-2 zs embeddings for single drug model (full token method)
# python $dnabert_s_file \
#     --saved_embed_memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps' \
#     --phenotype_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
#     --output_path 'training_output/zero_shot/classification_results/dnabert2/token_embeds' \
#     --saved_model_path 'training_output/zero_shot/saved_models/dnabert2/token_embeds' \
#     --embed_type 'token' \
#     --num_epochs 30 \
#     --drug "ETHAMBUTOL" \
#     --random_seed 42 \
#     --model_name "DNABERTCNN" \


# Run with DNABert-2 zs embeddings for single drug model (PCA method)
# python $dnabert_s_file \
#     --saved_embed_memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps/PCA' \
#     --phenotype_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
#     --output_path 'training_output/zero_shot/classification_results/dnabert2/pca_embeds' \
#     --saved_model_path 'training_output/zero_shot/saved_models/dnabert2/pca_embeds' \
#     --embed_type 'pca' \
#     --num_epochs 30 \
#     --drug "ETHIONAMIDE" \
#     --random_seed 42 \
#     --model_name "DNABERTCNN" \


# Run with DNABert-2 zs embeddings for single drug model (mean dim method)
python $dnabert_s_file \
    --saved_embed_memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_dim/full/memmaps' \
    --phenotype_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
    --output_path 'training_output/zero_shot/classification_results/dnabert2/mean_dim_embeds' \
    --saved_model_path 'training_output/zero_shot/saved_models/dnabert2/mean_dim_embeds' \
    --embed_type 'mean_dim' \
    --num_epochs 30 \
    --drug "AMIKACIN" \
    --random_seed 42 \
    --model_name "DNABERTCNN"


# Run with DNABert-2 zs embeddings for single drug model (mean seq method)
# python $dnabert_s_file \
#     --saved_embed_memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_seq/full/memmaps' \
#     --phenotype_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
#     --output_path 'training_output/zero_shot/classification_results/dnabert2/mean_seq_embeds' \
#     --saved_model_path 'training_output/zero_shot/saved_models/dnabert2/mean_seq_embeds' \
#     --embed_type 'mean_seq' \
#     --num_epochs 30 \
#     --drug "AMIKACIN" \
#     --random_seed 42 \
#     --model_name "DNABERTMLP" \
#     --learning_rate "1e-5"
