#!/usr/bin/env bash
# Submit one Evo2 Slurm entry point with site-specific settings from
# .evo2-site.env. Additional arguments are passed to sbatch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evo2_env.sh
source "${SCRIPT_DIR}/evo2_env.sh"

if (( $# == 0 )); then
    echo "Usage: $0 JOB_SCRIPT [SBATCH_OPTIONS ...]" >&2
    exit 2
fi

JOB_SCRIPT="$1"
shift
if [[ "${JOB_SCRIPT}" != /* ]]; then
    JOB_SCRIPT="${EVO2_DIR}/${JOB_SCRIPT}"
fi
evo2_require_paths "${JOB_SCRIPT}"

JOB_KIND="$(basename "${JOB_SCRIPT}" .sh)"
LOG_DIR="${EVO2_DIR}/sbatch_logs/${JOB_KIND}"
mkdir -p "${LOG_DIR}"

evo2_build_sbatch_site_args
evo2_submit "${EVO2_SBATCH_ARGS[@]}" \
    --output="${LOG_DIR}/%x_%J.out" \
    --error="${LOG_DIR}/%x_%J.err" \
    "$@" \
    "${JOB_SCRIPT}"
