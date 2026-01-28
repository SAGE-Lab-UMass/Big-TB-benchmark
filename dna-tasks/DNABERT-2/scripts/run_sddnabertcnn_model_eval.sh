#!/bin/bash
#SBATCH --job-name=dnabert_s_run   # or use another scheduler directive if not using Slurm
#SBATCH --output=log_%j.txt        # save output to log file

# Enable better CUDA memory handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ====================================
# Hugging Face setup
# ====================================

# Load Hugging Face token securely from a hidden env file
source ~/.hf_token.env


# Optional: Set shared cache directory (recommended on clusters with shared storage)
# mkdir -p /project/pi_annagreen_umass_edu/saishradha/project_data_curation/huggingface_cache
# mkdir -p $HF_HOME  # Ensure it exists

# ====================================
# Python environment setup
# ====================================

# Activate your conda environment (uncomment if needed)
# source /work/pi_annagreen_umass_edu/saishradha/miniconda3/etc/profile.d/conda.sh
# conda activate /work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/cnn

# Load CUDA modules if your cluster requires it (uncomment if needed)
# module load cuda/11.8.2 cudnn/8.7.0.84-11.8 

# ====================================
# Script execution
# ====================================

dnabert_file="../train/zero_shot/sd_resistance_classification_eval.py"

# Run eval for the zs full token embeddings
python $dnabert_file \
    --saved_embed_memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps' \
    --phenotype_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
    --output_path 'training_output/zero_shot/classification_results/dnabert2/token_embeds' \
    --saved_model_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_models/dnabert2/token_embeds' \
    --saved_model_name 'DNABERTCNN' \
    --threshold_dir 'training_output/zero_shot/saved_parameters/dnabert2/token_embeds' \
    --embed_type 'token' \
    --drug "ETHIONAMIDE" 


# Run eval for the zs pca embeddings
# python $dnabert_file \
#     --saved_embed_memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps/PCA' \
#     --phenotype_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
#     --output_path 'training_output/zero_shot/classification_results/dnabert2/pca_embeds' \
#     --saved_model_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_models/dnabert2/pca_embeds' \
#     --saved_model_name 'DNABERTCNN' \
#     --threshold_dir 'training_output/zero_shot/saved_parameters/dnabert2/pca_embeds' \
#     --embed_type 'pca' \
#     --drug "ETHIONAMIDE" 


# Run eval for the zs mean dim embeddings
# python $dnabert_file \
#     --saved_embed_memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_dim/full/memmaps' \
#     --phenotype_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
#     --output_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/classification_results/dnabert2/mean_dim_embeds' \
#     --saved_model_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_models/dnabert2/mean_dim_embeds' \
#     --saved_model_name 'DNABERTCNN' \
#     --threshold_dir '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_parameters/dnabert2/mean_dim_embeds' \
#     --embed_type 'mean_dim' \
#     --drug "PYRAZINAMIDE" 


# Run eval for the zs mean seq embeddings
# python $dnabert_file \
#     --saved_embed_memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_seq/full/memmaps' \
#     --phenotype_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
#     --output_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/classification_results/dnabert2/mean_seq_embeds' \
#     --saved_model_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_models/dnabert2/mean_seq_embeds' \
#     --saved_model_name 'DNABERTMLP' \
#     --threshold_dir '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_parameters/dnabert2/mean_seq_embeds' \
#     --embed_type 'mean_seq' \
#     --drug "STREPTOMYCIN" 







