"""Recovers the exact isolate identities of the original random-split
 SHAP explain set for the Ref-Alt/SHAP logistic-regression pipeline
fully deterministically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROTEIN_TASKS_DIR))
sys.path.insert(0, str(THIS_DIR))

from regression_utils import encode_labels, load_feature_matrix_and_labels
from lineage_holdout_regression import _load_manifest
from shap_pr_utils import dedup_Xy, stratified_bg_explain_indices

SEED = 42
BG_FRAC = 0.10
MAX_BG = 160


def recover_original_explain_filenames(drug: str) -> list[str]:
    X, y_raw = load_feature_matrix_and_labels(drug)
    y = encode_labels(y_raw)
    n = X.shape[0]

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=SEED, stratify=y)
    full_idx = np.concatenate([train_idx, test_idx])
    X_full = X[full_idx]
    y_full = y[full_idx]

    X_u, y_u, dedup_idx = dedup_Xy(X_full, y_full)
    bg_idx, ex_idx = stratified_bg_explain_indices(y_u, bg_frac=BG_FRAC, seed=SEED, max_bg=MAX_BG)

    # ex_idx (into X_u) -> dedup_idx (into X_full) -> full_idx (into original X)
    orig_row_idx = full_idx[dedup_idx[ex_idx]]

    manifest = _load_manifest(drug)
    row_to_filename = dict(zip(manifest['matrix_row_idx'], manifest['Filename'].astype(str)))
    filenames = [row_to_filename[i] for i in orig_row_idx if i in row_to_filename]
    return filenames
