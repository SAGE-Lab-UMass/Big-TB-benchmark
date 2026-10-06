#!/usr/bin/env bash
# Submit per-fold random-split test-set evaluation jobs for one or more drugs.
#
# Usage: ./submit_random_split_eval.sh DRUG [DRUG ...]
#
# Results land under
#   <RESULTS_ROOT>/<DRUG>/classification_results/evo2/token/<DRUG>/seed_42/fold_<N>/test_set_auc_<DRUG>.csv
# using the best-validation-AUC checkpoint (DNABERTCNN_best_model.pt).

set -euo pipefail

EVO2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_ROOT="${RESULTS_ROOT:-/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/training_output/zero_shot/random_split}"
LOG_ROOT="${LOG_ROOT:-/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/sbatch_downstream_logs}"
RANDOM_SEED="${RANDOM_SEED:-42}"
EMBED_TYPE="${EMBED_TYPE:-token}"
SAVED_MODEL_NAME="${SAVED_MODEL_NAME:-DNABERTCNN_best_model}"
FOLDS="${FOLDS:-1 2 3 4 5}"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 DRUG [DRUG ...]" >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}/out" "${LOG_ROOT}/error"

SUBMITTED=()
for DRUG in "$@"; do
    DRUG_ROOT="${RESULTS_ROOT}/${DRUG}"
    SAVED_MODEL_PATH="${DRUG_ROOT}/saved_models/evo2/${EMBED_TYPE}"
    OUTPUT_PATH="${DRUG_ROOT}/classification_results/evo2/${EMBED_TYPE}"
    THRESHOLD_DIR="${DRUG_ROOT}/saved_parameters/evo2/${EMBED_TYPE}"

    if [[ ! -d "${SAVED_MODEL_PATH}" ]]; then
        echo "ERROR: saved model dir not found: ${SAVED_MODEL_PATH}" >&2
        exit 1
    fi

    for fold in ${FOLDS}; do
        CHECKPOINT="${SAVED_MODEL_PATH}/${DRUG}/seed_${RANDOM_SEED}/fold_${fold}/${SAVED_MODEL_NAME}.pt"
        if [[ ! -f "${CHECKPOINT}" ]]; then
            echo "Skipping ${DRUG} fold ${fold}: missing ${CHECKPOINT}"
            continue
        fi

        JOB_NAME="evo2eval_random_${DRUG}_f${fold}_s${RANDOM_SEED}"
        JOB_ID=$(sbatch \
            --account=pi_annagreen_umass_edu \
            --partition=superpod-a100 \
            --mail-user=saishradhamo@umass.edu \
            --mail-type=END,FAIL \
            --job-name="${JOB_NAME}" \
            --output="${LOG_ROOT}/out/${JOB_NAME}_%j.out" \
            --error="${LOG_ROOT}/error/${JOB_NAME}_%j.err" \
            --export="ALL,EVO2_DIR=${EVO2_DIR},DRUG=${DRUG},EMBED_TYPE=${EMBED_TYPE},SAVED_MODEL_NAME=${SAVED_MODEL_NAME},RANDOM_SEED=${RANDOM_SEED},EVAL_FOLD=${fold},OUTPUT_PATH=${OUTPUT_PATH},SAVED_MODEL_PATH=${SAVED_MODEL_PATH},THRESHOLD_DIR=${THRESHOLD_DIR}" \
            "${EVO2_DIR}/evaluate_random_split_classifier.sh" | awk '{print $NF}')

        SUBMITTED+=("${JOB_ID}")
        echo "Submitted ${DRUG} fold ${fold}: Job ID ${JOB_ID}"
    done
done

echo ""
echo "Submitted ${#SUBMITTED[@]} job(s): ${SUBMITTED[*]}"
echo "Logs: ${LOG_ROOT}/out"
