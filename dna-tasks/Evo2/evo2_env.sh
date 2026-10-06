#!/usr/bin/env bash
# Shared, portable runtime configuration for Evo2 shell entry points.

if [[ -z "${EVO2_DIR:-}" ]]; then
    EVO2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
export EVO2_DIR

EVO2_SITE_CONFIG="${EVO2_SITE_CONFIG:-${EVO2_DIR}/.evo2-site.env}"
if [[ -r "${EVO2_SITE_CONFIG}" ]]; then
    # shellcheck disable=SC1090
    source "${EVO2_SITE_CONFIG}"
elif [[ -n "${EVO2_SITE_CONFIG_REQUIRED:-}" ]]; then
    echo "Required Evo2 site configuration not found: ${EVO2_SITE_CONFIG}" >&2
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return 1
    fi
    exit 1
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${EVO2_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Portable defaults. A site profile can override these with cluster storage and
# environment locations without changing any tracked launcher.
export EVO2_EMBED_PYTHON="${EVO2_EMBED_PYTHON:-python}"
export EVO2_TRAIN_PYTHON="${EVO2_TRAIN_PYTHON:-python}"
export EVO2_DATA_DIR="${EVO2_DATA_DIR:-${EVO2_DIR}/data/multidrug_classification/training}"
export EVO2_PHENOTYPE_FILE="${EVO2_PHENOTYPE_FILE:-${EVO2_DATA_DIR}/phenotype/master_resistance_table.csv}"
export EVO2_GENOTYPE_INPUT_DIRECTORY="${EVO2_GENOTYPE_INPUT_DIRECTORY:-${EVO2_DIR}/data/aligned}"
export EVO2_EMBED_ROOT="${EVO2_EMBED_ROOT:-${EVO2_DIR}/embeddings/zero-shot/token/layer20/full}"
# evo2_downstream.config historically used the longer variable name.
export EVO2_RAW_TOKEN_EMBED_ROOT="${EVO2_RAW_TOKEN_EMBED_ROOT:-${EVO2_EMBED_ROOT}}"
export EVO2_DOWNSTREAM_DATA_ROOT="${EVO2_DOWNSTREAM_DATA_ROOT:-${EVO2_DIR}/downstream_inputs/layer20}"
export EVO2_GENO_PHENO_CSV="${EVO2_GENO_PHENO_CSV:-${EVO2_DATA_DIR}/geno_pheno_full_combined.csv}"
export EVO2_LINEAGE_CSV="${EVO2_LINEAGE_CSV:-$(cd "${EVO2_DIR}/../.." && pwd)/BIG_TB_isolates_with_lineages.csv}"

evo2_require_vars() {
    local name
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            echo "Required configuration variable is unset: ${name}" >&2
            return 1
        fi
    done
}

evo2_require_executable() {
    local executable="$1"
    if [[ "${executable}" == */* ]]; then
        if [[ ! -x "${executable}" ]]; then
            echo "Python executable is missing or not executable: ${executable}" >&2
            return 1
        fi
    elif ! command -v "${executable}" >/dev/null 2>&1; then
        echo "Executable is not available on PATH: ${executable}" >&2
        return 1
    fi
}

evo2_require_paths() {
    local path
    for path in "$@"; do
        if [[ ! -e "${path}" ]]; then
            echo "Required input path does not exist: ${path}" >&2
            return 1
        fi
    done
}

evo2_load_cuda_module() {
    if [[ -z "${EVO2_CUDA_MODULE:-}" ]]; then
        return
    fi
    if ! type module >/dev/null 2>&1; then
        echo "EVO2_CUDA_MODULE=${EVO2_CUDA_MODULE}, but the module command is unavailable" >&2
        return 1
    fi
    module load "${EVO2_CUDA_MODULE}"
}

evo2_run() {
    if [[ "${EVO2_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
        printf '[evo2 dry-run]'
        printf ' %q' "$@"
        printf '\n'
        return
    fi
    exec "$@"
}

evo2_add_sbatch_option() {
    local option="$1"
    local value="${2:-}"
    if [[ -n "${value}" ]]; then
        EVO2_SBATCH_ARGS+=("${option}=${value}")
    fi
}

evo2_build_sbatch_site_args() {
    EVO2_SBATCH_ARGS=()
    evo2_add_sbatch_option --account "${EVO2_SLURM_ACCOUNT:-}"
    evo2_add_sbatch_option --partition "${EVO2_SLURM_PARTITION:-}"
    evo2_add_sbatch_option --mail-user "${EVO2_SLURM_MAIL_USER:-}"
    if [[ -n "${EVO2_SLURM_MAIL_USER:-}" ]]; then
        evo2_add_sbatch_option --mail-type "${EVO2_SLURM_MAIL_TYPE:-END,FAIL}"
    fi
}

evo2_submit() {
    if [[ "${EVO2_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
        printf '[sbatch dry-run]'
        printf ' %q' sbatch "$@"
        printf '\n'
        return
    fi
    sbatch "$@"
}
