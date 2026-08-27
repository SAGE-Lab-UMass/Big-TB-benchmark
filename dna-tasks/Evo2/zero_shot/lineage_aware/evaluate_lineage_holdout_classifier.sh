#!/bin/bash
# Evaluate a saved downstream classifier on its held-out lineage.
#SBATCH -A pi_annagreen_umass_edu
#SBATCH --partition=superpod-a100
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --job-name=evo2_lineage_holdout_eval
#SBATCH --mail-user=saishradhamo@umass.edu
#SBATCH --output=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/sbatch_zero_shot_lineage_holdout_logs/out/%x_%J.out
#SBATCH --error=/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/sbatch_zero_shot_lineage_holdout_logs/error/%x_%J.err

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

# Prevent the submitting shell's Python/virtualenv configuration from leaking
# into this job.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

EVO2_DIR="${EVO2_DIR:-/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2}"
SCRIPT_DIR="${EVO2_DIR}/zero_shot/lineage_aware"

EVO2_CONDA_ROOT="${EVO2_CONDA_ROOT:-/work/pi_annagreen_umass_edu/saishradha/miniconda3}"
EVO2_ENV_PREFIX="${EVO2_ENV_PREFIX:-${EVO2_CONDA_ROOT}/envs/dnabert_s}"

mkdir -p "${EVO2_DIR}/sbatch_zero_shot_lineage_holdout_logs/out" "${EVO2_DIR}/sbatch_zero_shot_lineage_holdout_logs/error"

source "${EVO2_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${EVO2_ENV_PREFIX}"

DRUG="${DRUG:-AMIKACIN}"
HELDOUT_LINEAGE="${HELDOUT_LINEAGE:-2}"
EMBED_TYPE="${EMBED_TYPE:-token}"
MODEL_NAME="${MODEL_NAME:-DNABERTCNN}"
SAVED_MODEL_NAME="${SAVED_MODEL_NAME:-DNABERTCNN}"

MEMMAP_ROOT="${MEMMAP_ROOT:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/downstream_inputs/layer20/${EMBED_TYPE}/memmaps}"
PHENOTYPE_LABEL_PATH="${PHENOTYPE_LABEL_PATH:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings/zero-shot/token/layer20/full/zs_full_stacked_phenotypes.npz}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${EVO2_DIR}/training_output/zero_shot/lineage_aware_holdout}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${DRUG}/classification_results/evo2/${EMBED_TYPE}}"
SAVED_MODEL_PATH="${SAVED_MODEL_PATH:-${OUTPUT_ROOT}/${DRUG}/saved_models/evo2/${EMBED_TYPE}}"
THRESHOLD_DIR="${THRESHOLD_DIR:-${OUTPUT_ROOT}/${DRUG}/saved_parameters/evo2/${EMBED_TYPE}}"

RANDOM_SEED="${RANDOM_SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
TEST_SPLIT="${TEST_SPLIT:-0.2}"
PCA_COMPONENTS="${PCA_COMPONENTS:-10}"
MIN_CLASS_COUNT="${MIN_CLASS_COUNT:-50}"

cd "${EVO2_DIR}"

"${EVO2_ENV_PREFIX}/bin/python" -I "${SCRIPT_DIR}/eval_lineage_holdout.py" \
    --drug "${DRUG}" \
    --heldout-lineage "${HELDOUT_LINEAGE}" \
    --embed_type "${EMBED_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --saved_model_name "${SAVED_MODEL_NAME}" \
    --saved_embed_memmap_dir "${MEMMAP_ROOT}" \
    --phenotype_label_path "${PHENOTYPE_LABEL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --saved_model_path "${SAVED_MODEL_PATH}" \
    --threshold_dir "${THRESHOLD_DIR}" \
    --random_seed "${RANDOM_SEED}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --test_split "${TEST_SPLIT}" \
    --pca_components "${PCA_COMPONENTS}" \
    --min-class-count "${MIN_CLASS_COUNT}" \
    "$@"
