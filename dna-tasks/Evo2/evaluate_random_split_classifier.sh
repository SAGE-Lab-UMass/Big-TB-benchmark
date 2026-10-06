#!/bin/bash
# Evaluate an Evo2 random-split downstream classifier on the held-out test set.
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --job-name=evo2_random_split_eval

set -euo pipefail

if [[ -z "${EVO2_DIR:-}" ]]; then
    EVO2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ ! -d "${EVO2_DIR}/evo2_downstream" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        EVO2_DIR="${SLURM_SUBMIT_DIR}"
    fi
fi
if [[ ! -d "${EVO2_DIR}/evo2_downstream" ]]; then
    echo "Cannot find evo2_downstream in ${EVO2_DIR}. Set EVO2_DIR before submitting." >&2
    exit 1
fi
export EVO2_DIR
export PYTHONNOUSERSITE=1
export PYTHONPATH="${EVO2_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_ROOT="${CONDA_ROOT:-/work/pi_annagreen_umass_edu/saishradha/miniconda3}"
EVAL_PYTHON="${EVAL_PYTHON:-${CONDA_ROOT}/envs/dnabert_s/bin/python}"

DRUG="${DRUG:?DRUG must be set}"
EMBED_TYPE="${EMBED_TYPE:-token}"
MODEL_NAME="${MODEL_NAME:-DNABERTCNN}"
# DNABERTCNN_best_model.pt is the checkpoint saved at the best validation AUC.
SAVED_MODEL_NAME="${SAVED_MODEL_NAME:-DNABERTCNN_best_model}"
MEMMAP_ROOT="${MEMMAP_ROOT:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/downstream_inputs/layer20/${EMBED_TYPE}/memmaps}"
PHENOTYPE_LABEL_PATH="${PHENOTYPE_LABEL_PATH:-/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings/zero-shot/token/layer20/full/zs_full_stacked_phenotypes.npz}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH must be set}"
SAVED_MODEL_PATH="${SAVED_MODEL_PATH:?SAVED_MODEL_PATH must be set}"
THRESHOLD_DIR="${THRESHOLD_DIR:?THRESHOLD_DIR must be set}"
RANDOM_SEED="${RANDOM_SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
TEST_SPLIT="${TEST_SPLIT:-0.2}"
PCA_COMPONENTS="${PCA_COMPONENTS:-10}"
EVAL_FOLD="${EVAL_FOLD:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ "${USE_BEST_FOLD:-0}" == "1" ]]; then
    if [[ -n "${EVAL_FOLD}" ]]; then
        echo "USE_BEST_FOLD=1 requires a single job without EVAL_FOLD or a Slurm array" >&2
        exit 1
    fi
    MANIFEST="${SAVED_MODEL_PATH}/${DRUG}/seed_${RANDOM_SEED}/best_fold.json"
    EVAL_FOLD="$("${EVAL_PYTHON}" - "${MANIFEST}" "${DRUG}" "${RANDOM_SEED}" "${MODEL_NAME}" "${EMBED_TYPE}" "${PCA_COMPONENTS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path, drug, seed, model, embed_type, components = sys.argv[1:]
record = json.loads(Path(path).read_text())
expected = {"drug": drug, "seed": int(seed), "model_name": model,
            "embed_type": embed_type}
for key, value in expected.items():
    if record.get(key) != value:
        raise SystemExit(f"Best-fold record mismatch for {key}: {record.get(key)!r} != {value!r}")
if embed_type == "pca" and record.get("pca_components") != int(components):
    raise SystemExit("Best-fold record has a different PCA component count")
fold = int(record["best_fold"])
if fold not in range(1, 6):
    raise SystemExit(f"Invalid best fold: {fold}")
checkpoint = Path(path).parent / f"fold_{fold}" / f"{model}_best_model.pt"
if checkpoint.resolve() != Path(record["checkpoint"]).resolve():
    raise SystemExit("Best-fold record points to a different checkpoint")
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != record["checkpoint_sha256"]:
    raise SystemExit("Best-fold checkpoint changed after selection; regenerate best_fold.json")
print(fold)
PY
)"
    echo "Selected fold ${EVAL_FOLD} from ${MANIFEST}"
fi

EXTRA_ARGS=()
if [[ -n "${EVAL_FOLD}" ]]; then
    EXTRA_ARGS+=(--fold "${EVAL_FOLD}")
fi

for path in "${MEMMAP_ROOT}" "${PHENOTYPE_LABEL_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required input path does not exist: ${path}" >&2
        exit 1
    fi
done

CHECKPOINT="${SAVED_MODEL_PATH}/${DRUG}/seed_${RANDOM_SEED}${EVAL_FOLD:+/fold_${EVAL_FOLD}}/${SAVED_MODEL_NAME}.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

cd "${EVO2_DIR}"

exec "${EVAL_PYTHON}" -m evo2_downstream.eval \
    --drug "${DRUG}" \
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
    "${EXTRA_ARGS[@]}" \
    "$@"
