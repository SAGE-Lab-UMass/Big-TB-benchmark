"""Shared configuration for Evo2 downstream training."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

# Local vendored finetuning utilities (now under finetuning/modules).
LOCAL_FINETUNE_UTILS_DIR = PROJECT_DIR / "finetuning" / "modules"
LEGACY_FINETUNE_UTILS_DIR_V2 = PROJECT_DIR / "evo2_finetune_modules"
LEGACY_FINETUNE_UTILS_DIR = PROJECT_DIR / "dnabert_modules"

# Backward-compatible resolver: prefer new folder name, then legacy folder name.
if LOCAL_FINETUNE_UTILS_DIR.exists():
    FINETUNE_UTILS_DIR = LOCAL_FINETUNE_UTILS_DIR
elif LEGACY_FINETUNE_UTILS_DIR_V2.exists():
    FINETUNE_UTILS_DIR = LEGACY_FINETUNE_UTILS_DIR_V2
else:
    FINETUNE_UTILS_DIR = LEGACY_FINETUNE_UTILS_DIR

RAW_TOKEN_EMBED_ROOT = Path(
    os.environ.get(
        "EVO2_RAW_TOKEN_EMBED_ROOT",
        str(PROJECT_DIR / "embeddings" / "zero_shot" / "token" / "layer20" / "full"),
    )
)
DOWNSTREAM_DATA_ROOT = Path(
    os.environ.get(
        "EVO2_DOWNSTREAM_DATA_ROOT",
        str(PROJECT_DIR / "downstream_inputs" / "layer20"),
    )
)

DEFAULT_PHENOTYPE_LABEL_PATH = RAW_TOKEN_EMBED_ROOT / "zs_full_stacked_phenotypes.npz"


def ensure_finetune_utils_on_path() -> None:
    """Expose vendored finetuning utility modules to Evo2 wrappers."""
    if not FINETUNE_UTILS_DIR.exists():
        raise FileNotFoundError(
            "Missing finetuning utility folder. Expected one of: "
            f"{LOCAL_FINETUNE_UTILS_DIR}, {LEGACY_FINETUNE_UTILS_DIR_V2}, "
            f"or {LEGACY_FINETUNE_UTILS_DIR}"
        )
    utils_path = str(FINETUNE_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)


def ensure_dnabert_transfer_learn_on_path() -> None:
    """Backward-compatible alias for older imports."""
    ensure_finetune_utils_on_path()


def memmap_root(embed_type: str) -> Path:
    """Return the default memmap root for an Evo2 embedding type."""
    return DOWNSTREAM_DATA_ROOT / embed_type / "memmaps"


def classification_output_root(embed_type: str) -> Path:
    return PROJECT_DIR / "training_output" / "zero_shot" / "classification_results" / "evo2" / embed_type


def saved_model_root(embed_type: str) -> Path:
    return PROJECT_DIR / "training_output" / "zero_shot" / "saved_models" / "evo2" / embed_type


def threshold_root(embed_type: str) -> Path:
    return PROJECT_DIR / "training_output" / "zero_shot" / "saved_parameters" / "evo2" / embed_type


# Column ordering of drugs in the Evo2 phenotype NPZ
# (matches the order written by evo2_embed_gen, verified from embed gen logs)
EVO2_DRUG_INDEX: dict[str, int] = {
    'ISONIAZID':    0,
    'RIFAMPICIN':   1,
    'ETHAMBUTOL':   2,
    'PYRAZINAMIDE': 3,
    'STREPTOMYCIN': 4,
    'KANAMYCIN':    5,
    'AMIKACIN':     6,
    'CAPREOMYCIN':  7,
    'LEVOFLOXACIN': 8,
    'MOXIFLOXACIN': 9,
    'ETHIONAMIDE':  10,
}
