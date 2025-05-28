#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu   # Account
#SBATCH --partition=superpod-a100
#SBATCH -G 1                  # Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=600G
#SBATCH --time=5:00:00
#SBATCH --mail-user=saishradhamo@umass.edu
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_ft_logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_ft_logs/error/%x_%J.err


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

dnabert_s_file="train/finetune_learn/multigene_main.py"

# Run with DNABert-S 
# python $dnabert_s_file
# echo "Finished running DNABERT-S finetuning."


# Run with DNABert-2
python $dnabert_s_file --model_name_or_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/pretrained_models/DNABERT2' --output_dir "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/finetune/dnabert2" --resume_checkpoint_path "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/finetune/dnabert2/saved_checkpoints/checkpoint_epoch_2.pth"

echo "Finished running DNABERT-2 finetuning."
