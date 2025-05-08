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

dnabert_s_file="finetune/finetune_main.py"

# Run with DNABert-S + MLP head
python $dnabert_s_file 
