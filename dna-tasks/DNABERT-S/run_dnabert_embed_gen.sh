#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu   # Account
#SBATCH --partition=superpod-a100
#SBATCH -G 1                  # Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=600G
#SBATCH --time=10:00:00
#SBATCH --mail-user=saishradhamo@umass.edu
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_embed_gen_logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_embed_gen_logs/error/%x_%J.err



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

# dnabert_s_file="train/transfer_learn/generate_embeddings.py"

# # Run with DNABert-S + MLP head
# python $dnabert_s_file # --max_length 5000 --batch_size 10
# echo "Finished running DNABERT-S embedding generation script."
# dnabert_s_file="train/transfer_learn/generate_embeddings.py"
# python $dnabert_s_file --embed_dir 'training_output/zero_shot/embeddings_5000' --embed_type "zs" --max_length 5000 --train_batch_size 10 --val_batch_size 10

# for dnabert2 - zero shot
# dnabert_s_file="train/transfer_learn/generate_embeddings.py"
# python $dnabert_s_file --embed_dir 'training_output/zero_shot/embeddings_5000/dnabert2' --embed_type "zs" --max_length 5000 --train_batch_size 10 --val_batch_size 10 --model_name_or_path 'pretrained_models/DNABERT2'


# for dnabert2 - fine-tune
dnabert_s_file="train/transfer_learn/generate_embeddings.py"
python $dnabert_s_file --embed_dir 'training_output/finetune/embeddings_5000/dnabert2' --embed_type "ft" --ft_model_path "training_output/finetune/dnabert2/saved_models/dnabert_only_finetuned_epoch_4.pth" --max_length 5000 --train_batch_size 10 --val_batch_size 10 --model_name_or_path 'pretrained_models/DNABERT2'


# Run inference with fine-tuned model
# dnabert_s_file="train/transfer_learn/generate_embeddings.py"
# python $dnabert_s_file --embed_type "ft" --ft_model_path "training_output/finetune/saved_models/dnabert_only_finetuned_epoch_4.pth" --embed_dir 'training_output/finetune/embeddings'

