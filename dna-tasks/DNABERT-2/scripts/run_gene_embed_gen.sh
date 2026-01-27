#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu   # Account
#SBATCH --partition=superpod-a100
#SBATCH -G 1                  # Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=600G
#SBATCH --time=8:00:00
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

dnabert_s_file="../train/zero_shot/generate_gene_embeddings.py"

# Run zero shot inference with separated genes (For SD-DNABERT-CNN cases)
# is_single gene algo parameter will always use the "full" dataset

# For generating full token embeddings for a specified gene
python $dnabert_s_file --embed_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new' --embed_method "zs" --max_length 5000 --full_batch_size 12 --model_name_or_path 'pretrained_models/DNABERT2' --genotype_input_directory "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/genomic_data/aligned" --genes "rpoB" --is_single_gene_algo --embed_type "token"

# For generating mean sequence embeddings for a specified gene
# python $dnabert_s_file --embed_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_seq/full' --embed_method "zs" --max_length 5000 --full_batch_size 12 --model_name_or_path 'pretrained_models/DNABERT2' --genotype_input_directory "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/genomic_data/aligned" --genes "rrs" --is_single_gene_algo --embed_type "mean_seq"

# For generating mean dim embeddings for a specified gene
# python $dnabert_s_file --embed_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_seq/full' --embed_method "zs" --max_length 5000 --full_batch_size 12 --model_name_or_path 'pretrained_models/DNABERT2' --genotype_input_directory "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/genomic_data/aligned" --genes "rrs" --is_single_gene_algo --embed_type "mean_seq"


#----------#

# For generating mean dim embeddings for multiple genes (all genes for MD-DNABERT-CNN) 
# WARNING: WE RECOMMEND SEPARATE RUNS PER GENE OR AS SEPARATE SETS OF GENES DUE TO MEMORY CONSTRAINTS AND LATER STACKING
# python $dnabert_s_file --embed_dir 'training_output/zero_shot/embeddings/mean_dim' --embed_type "zs" --max_length 5000 --train_batch_size 10 --val_batch_size 10 --model_name_or_path 'pretrained_models/DNABERT2' --genotype_input_directory "finetune_data/multidrug_classification/training/genotype/combined_aligned" --drug "ALL" --genes "gyrB","gyrA","rpoB","rpoC","rpsL","fabG1","inhA","rrs","rrl","tlyA","katG","pncA","embC","embA","embB","ethA","ethR","gid"

