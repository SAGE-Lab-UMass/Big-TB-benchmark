#!/bin/bash
#SBATCH -A pi_annagreen_umass_edu   # Account
#SBATCH --partition=superpod-a100
#SBATCH -G 1                  # Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=600G
#SBATCH --time=10:00:00
#SBATCH --mail-user=saishradhamo@umass.edu
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_interpret_logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/sbatch_interpret_logs/error/%x_%J.err



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

shap_interpret_file="../interpretability/run_shap_interpret.py"

# Run SHAP interpretability analysis
python $shap_interpret_file \
    --dnabert_model 'pretrained_models/DNABERT2' \
    --dnabert_model_max_len 5000 \
    --memmap_dir '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps' \
    --model_dir '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_models/dnabert2/token_embeds' \
    --pheno_label_path '/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz' \
    --ref_seq_json_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/interpretability/dataloader/ref_gene_seq.json' \
    --vcf_who_map_dir '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/vcf_who_mapped_data' \
    --drug 'RIFAMPICIN' \
    --embed_type 'token' \
    --in_dim 768 \
    --background_frac 0.1 \
    --explain_frac 1.0 \
    --top_n_positions 100 \
    --output_path '/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT-2/interpretability/output/shap_results' \
    --has_neg_strand True \
    --random_seed 42
