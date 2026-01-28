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

dnabert_file="train/transfer_learn/resistance_classification_eval.py"


# Run wih DNABert-2 zs embeddings for multi drug model (mean seq method)
python $dnabert_file --saved_embed_dir "training_output/finetune/embeddings_5000/dnabert2" --output_path 'training_output/finetune/classification_results/dnabert2' --saved_model_path 'training_output/finetune/saved_models/dnabert2/cv_seed_1' --saved_model_name 'dnabert-mdcnn_cv_split_3.pt'