"""Lineage-aware data split utilities for Evo2 downstream training.

The ``full_N`` identifiers in the Evo2 memmap (e.g. ``full_000042``) correspond
to row N (0-based) in ``geno_pheno_full_combined.csv``.  That CSV contains the
biological isolate ID (``Unnamed: 0`` column, e.g. ``SAMEA104394571``), which is
joined to ``BIG_TB_isolates_with_lineages.csv`` to obtain the MTB lineage.

Main public API
---------------
- ``load_isolate_id_map(csv_path)``  →  ``{row_index: isolate_id}``
- ``load_lineage_map(lineage_csv_path)``  →  ``{isolate_id: major_lineage_str}``
- ``make_lineage_aware_split_fn(heldout_lineage, isolate_id_map, lineage_map)``
      returns a drop-in replacement for
      ``resistance_classification_train.stratified_split_dataset``
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd


MAJOR_LINEAGES: tuple[str, ...] = ("1", "2", "3", "4")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_isolate_id_map(csv_path: str) -> dict[int, str]:
    """Return ``{row_index: isolate_id}`` from ``geno_pheno_full_combined.csv``.

    Row index N corresponds to the Evo2 memmap identifier ``full_N``.
    """
    df = pd.read_csv(csv_path, usecols=["Unnamed: 0"])
    return {i: str(v) for i, v in enumerate(df["Unnamed: 0"])}


def _normalize_lineage(val) -> str | None:
    """Return the major MTB lineage as a plain string ('1'–'4'), or None.
    
    IMPORTANT: Mixed lineages (e.g., "1,4", "2,3") are kept as-is and will NOT
    match pure lineages "1", "2", "3", "4", so samples with mixed lineages are
    effectively excluded from lineage-specific splits. This matches SD-CNN behavior.
    """
    if pd.isna(val):
        return None
    txt = str(val).strip()
    if not txt or txt.lower() in ("nan", "none", ""):
        return None
    # Do NOT split mixed-lineage entries - keep them as-is (e.g., "1,4" stays "1,4")
    # They will not match "1", "2", "3", "4" and will be excluded from splits
    if "," not in txt:
        try:
            f = float(txt)
            if float(f).is_integer():
                return str(int(f))
        except ValueError:
            pass
    return txt


def load_lineage_map(lineage_csv_path: str) -> dict[str, str | None]:
    """Return ``{isolate_id: major_lineage_str}`` from
    ``BIG_TB_isolates_with_lineages.csv``."""
    df = pd.read_csv(lineage_csv_path, usecols=["ROLLINGDB_ID", "Lineage"])
    df["ROLLINGDB_ID"] = df["ROLLINGDB_ID"].astype(str)
    lineage_norm = df["Lineage"].apply(_normalize_lineage)
    return dict(zip(df["ROLLINGDB_ID"], lineage_norm))


# ──────────────────────────────────────────────────────────────────────────────
# Core split logic
# ──────────────────────────────────────────────────────────────────────────────

def _parse_full_index(seq_id: str) -> int:
    """Extract row index from identifiers like ``'full_000042'`` → ``42``."""
    m = re.match(r"full_(\d+)$", seq_id)
    if not m:
        raise ValueError(f"Cannot parse numeric index from seq_id: {seq_id!r}")
    return int(m.group(1))


def make_lineage_aware_split_fn(
    heldout_lineage: str,
    isolate_id_map: dict[int, str],
    lineage_map: dict[str, str | None],
    min_class_count: int = 50,
) -> Callable:
    """Return a drop-in replacement for ``stratified_split_dataset``.

    The returned callable has the same signature::

        fn(full_dataset, label_dict, test_size=0.2, seed=42)
            -> (train_indices, test_indices, y_train, y_test)

    ``test_size`` and ``seed`` are accepted for API compatibility but ignored.

    Samples whose isolate lacks lineage annotation are placed in the training
    set so that no labelled data is wasted.

    Args:
        heldout_lineage: Major MTB lineage (1-4) to hold out as test set.
        isolate_id_map: Mapping from row index to isolate ID.
        lineage_map: Mapping from isolate ID to lineage annotation.
        min_class_count: Minimum samples required in each class (train_S, train_R, test_S, test_R).
            If any class has fewer samples, raises ValueError.
    """
    held = str(heldout_lineage)

    def lineage_split(
        full_dataset,
        label_dict: dict[str, float],
        test_size: float = 0.2,
        seed: int = 42,
    ):
        print(f"\n[LineageSplit] Leave-one-lineage-out: held-out lineage = {held}")

        # ── collect ordered seq_ids from the dataset ──────────────────────────
        if hasattr(full_dataset, "lookup"):
            # TokenMemmapMap / PcaMemmapMap
            seq_ids = [
                full_dataset.blocks[bidx][0][ridx]
                for bidx, ridx in full_dataset.lookup
            ]
        elif hasattr(full_dataset, "ids"):
            # MultiGeneConcatDataset and variants
            seq_ids = full_dataset.ids
        else:
            raise ValueError(
                "Dataset type not recognised: missing 'lookup' or 'ids' attribute"
            )

        id_to_dataset_idx = {sid: i for i, sid in enumerate(seq_ids)}

        # ── assign each labelled sample to train or test ──────────────────────
        train_ids: list[str] = []
        test_ids: list[str] = []
        no_lineage_count = 0

        for sid in seq_ids:
            if sid not in label_dict:
                continue
            row_idx = _parse_full_index(sid)
            isolate_id = isolate_id_map.get(row_idx)
            lineage = lineage_map.get(isolate_id) if isolate_id is not None else None

            if lineage is None:
                no_lineage_count += 1
                train_ids.append(sid)  # no annotation → train
                continue

            if lineage == held:
                test_ids.append(sid)
            else:
                train_ids.append(sid)

        if no_lineage_count:
            print(
                f"[LineageSplit] {no_lineage_count} samples had no lineage "
                "annotation and were placed in the training set."
            )

        train_indices = np.array([id_to_dataset_idx[sid] for sid in train_ids])
        test_indices = np.array([id_to_dataset_idx[sid] for sid in test_ids])
        y_train = np.array([label_dict[sid] for sid in train_ids], dtype=float)
        y_test = np.array([label_dict[sid] for sid in test_ids], dtype=float)

        # Count class samples
        train_s = int((y_train == 1).sum())
        train_r = int((y_train == 0).sum())
        test_s = int((y_test == 1).sum())
        test_r = int((y_test == 0).sum())

        print(
            f"[LineageSplit] train={len(train_indices)} "
            f"(S={train_s}  R={train_r}),  "
            f"test(lineage_{held})={len(test_indices)} "
            f"(S={test_s}  R={test_r})"
        )

        # Feasibility check: require min_class_count in ALL four classes
        if min(train_s, train_r, test_s, test_r) < min_class_count:
            raise ValueError(
                f"Insufficient samples for feasible training (min_class_count={min_class_count}). "
                f"train_S={train_s}, train_R={train_r}, test_S={test_s}, test_R={test_r}. "
                f"At least {min_class_count} samples required in each class."
            )

        return train_indices, test_indices, y_train, y_test

    return lineage_split
