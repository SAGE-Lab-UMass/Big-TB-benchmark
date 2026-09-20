#!/usr/bin/env bash
# Submit per-lineage held-out test evaluation jobs for one or more drugs.
#
# Usage: ./submit_lineage_holdout_eval.sh DRUG [DRUG ...]
#
# Results land under
#   <OUTPUT_ROOT>/<DRUG>/classification_results/evo2/token/heldout_lineage_<N>/<DRUG>/seed_42/test_set_auc_<DRUG>.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Checkout that holds training_output/ and the lineage/geno-pheno CSVs.
DATA_EVO2_DIR="${DATA_EVO2_DIR:-/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_EVO2_DIR}/training_output/zero_shot/lineage_aware_holdout}"
GENO_PHENO_CSV="${GENO_PHENO_CSV:-${DATA_EVO2_DIR}/data/multidrug_classification/training/geno_pheno_full_combined.csv}"
LINEAGE_CSV="${LINEAGE_CSV:-$(cd "${DATA_EVO2_DIR}/../.." && pwd)/BIG_TB_isolates_with_lineages.csv}"

EMBED_TYPE="${EMBED_TYPE:-token}"
RANDOM_SEED="${RANDOM_SEED:-42}"
SAVED_MODEL_NAME="${SAVED_MODEL_NAME:-DNABERTCNN_best_model}"
LINEAGES="${LINEAGES:-1 2 3 4}"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 DRUG [DRUG ...]" >&2
    exit 1
fi

mkdir -p "${DATA_EVO2_DIR}/sbatch_zero_shot_lineage_holdout_logs/out" \
         "${DATA_EVO2_DIR}/sbatch_zero_shot_lineage_holdout_logs/error"

SUBMITTED=()
for DRUG in "$@"; do
    SAVED_MODEL_PATH="${OUTPUT_ROOT}/${DRUG}/saved_models/evo2/${EMBED_TYPE}"
    OUTPUT_PATH="${OUTPUT_ROOT}/${DRUG}/classification_results/evo2/${EMBED_TYPE}"
    THRESHOLD_DIR="${OUTPUT_ROOT}/${DRUG}/saved_parameters/evo2/${EMBED_TYPE}"

    for lineage in ${LINEAGES}; do
        CHECKPOINT="${SAVED_MODEL_PATH}/heldout_lineage_${lineage}/${DRUG}/seed_${RANDOM_SEED}/${SAVED_MODEL_NAME}.pt"
        if [[ ! -f "${CHECKPOINT}" ]]; then
            echo "Skipping ${DRUG} lineage ${lineage}: missing ${CHECKPOINT}"
            continue
        fi

        JOB_NAME="evo2eval_lineage_${DRUG}_L${lineage}_s${RANDOM_SEED}"
        JOB_ID=$(sbatch \
            --job-name="${JOB_NAME}" \
            --export="ALL,EVO2_DIR=${EVO2_DIR},DRUG=${DRUG},HELDOUT_LINEAGE=${lineage},EMBED_TYPE=${EMBED_TYPE},SAVED_MODEL_NAME=${SAVED_MODEL_NAME},RANDOM_SEED=${RANDOM_SEED},OUTPUT_ROOT=${OUTPUT_ROOT},OUTPUT_PATH=${OUTPUT_PATH},SAVED_MODEL_PATH=${SAVED_MODEL_PATH},THRESHOLD_DIR=${THRESHOLD_DIR},GENO_PHENO_CSV=${GENO_PHENO_CSV},LINEAGE_CSV=${LINEAGE_CSV}" \
            "${SCRIPT_DIR}/evaluate_lineage_holdout_classifier.sh" | awk '{print $NF}')

        SUBMITTED+=("${JOB_ID}")
        echo "Submitted ${DRUG} heldout lineage ${lineage}: Job ID ${JOB_ID}"
    done
done

echo ""
echo "Submitted ${#SUBMITTED[@]} job(s): ${SUBMITTED[*]}"
echo "Logs: ${DATA_EVO2_DIR}/sbatch_zero_shot_lineage_holdout_logs/out"
