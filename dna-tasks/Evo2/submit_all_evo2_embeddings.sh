#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evo2_env.sh
source "${SCRIPT_DIR}/evo2_env.sh"

GENE_FILE="${GENE_FILE:-${EVO2_DIR}/ordered_genes.txt}"
evo2_require_paths "${GENE_FILE}"
GENE_COUNT=$(awk 'NF {count++} END {print count + 0}' "${GENE_FILE}")
if (( GENE_COUNT == 0 )); then
    echo "No genes found in ${GENE_FILE}" >&2
    exit 1
fi
ARRAY_END=$((GENE_COUNT - 1))

LOG_DIR="${EVO2_DIR}/sbatch_embed_gen_logs"
mkdir -p "${LOG_DIR}/out" "${LOG_DIR}/error"

evo2_build_sbatch_site_args
evo2_submit "${EVO2_SBATCH_ARGS[@]}" \
    --array="0-${ARRAY_END}%${EVO2_EMBED_ARRAY_CONCURRENCY:-3}" \
    --output="${LOG_DIR}/out/%x_%A_%a.out" \
    --error="${LOG_DIR}/error/%x_%A_%a.err" \
    "${EVO2_DIR}/run_all_evo2_embeddings_sbatch.sh"
