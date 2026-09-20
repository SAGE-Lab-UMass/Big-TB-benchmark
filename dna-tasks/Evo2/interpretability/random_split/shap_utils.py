"""SHAP value computation for Evo2 DNABERTCNN zero-shot random-split models.

Adapted from ``interpretability/lineage_aware_split/shap_utils.py``. The SHAP
mechanics (DeepExplainer over a CNN that consumes token embeddings shaped
``(D, L)``) and the dataset/dedup/fingerprint helpers are unchanged. Two
things differ for the random-split setting:

- **Background/explainer selection**: instead of a drug-level fixed explainer
  plus a separate per-lineage-model background, both the background and the
  explainer are chosen together in one stratified split directly from the
  drug's deduplicated dataset -- the same method used by
  ``SD-CNN/interpretability/utils.py::compute_shap_values_strat`` (20%
  stratified background capped at 160 samples, remainder is the explainer),
  with the explainer additionally capped at 360 samples (matching
  ``lineage_aware_split``'s ``max_explain``).
- **Model selection**: instead of evaluating every held-out lineage, a single
  fold is explained per drug -- the fold with the highest validation AUC
  found in ``<model_name>_fold<N>_history.csv`` (falling back to the first
  available fold when no fold has a usable history).
"""

from __future__ import annotations

import gc
import glob
import hashlib
import json
from pathlib import Path
from time import time
from typing import Sequence

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from evo2_downstream.config import ensure_finetune_utils_on_path

ensure_finetune_utils_on_path()
import resistance_classification_train as evo2_data  # noqa: E402
from utils.token_train_utils import get_model_class  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DRUG_TO_LOCI = evo2_data.DRUG_TO_LOCI
build_label_map = evo2_data.build_label_map


class Wrapped(torch.nn.Module):
    """Unsqueeze CNN logits to the ``(B, 1)`` shape SHAP's DeepExplainer expects."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x).unsqueeze(1)


def _gene_token_length(memmap_dir: str, gene: str, embed_type: str) -> int:
    meta_path = next(Path(memmap_dir, gene).glob(f"*_{embed_type}_meta.npz"))
    return int(np.load(meta_path, allow_pickle=True)["shape"][1])


def build_full_dataset(
    drug: str,
    embed_type: str,
    memmap_dir: str,
    phenotype_label_path: str,
):
    """Build the (single- or multi-gene) Evo2 token dataset for ``drug``.

    Returns ``(dataset, label_map, per_gene_lengths, gene_names)``.
    """
    full_label_map, _ = build_label_map(phenotype_label_path, drug, prefix="full")
    loci = DRUG_TO_LOCI[drug]

    if len(loci) == 1:
        gene = loci[0]
        meta_paths = sorted(glob.glob(f"{memmap_dir}/{gene}/*_{embed_type}_meta.npz"))
        if not meta_paths:
            raise FileNotFoundError(
                f"No {embed_type} metadata found for {gene} under {memmap_dir}"
            )
        dataset = evo2_data.TokenMemmapMap(meta_paths, full_label_map)
        per_gene_len = [_gene_token_length(memmap_dir, gene, embed_type)]
        gene_names = [gene]
    else:
        gene_dirs = [f"{memmap_dir}/{gene}/" for gene in loci]
        dataset = evo2_data.MultiGeneConcatDataset(gene_dirs, full_label_map)
        per_gene_len = [_gene_token_length(memmap_dir, gene, embed_type) for gene in loci]
        gene_names = list(loci)

    return dataset, full_label_map, per_gene_len, gene_names


def dataset_sample_ids(dataset) -> list[str]:
    """Return the full_N sample IDs in dataset order without loading tensors."""
    if hasattr(dataset, "lookup"):
        return [
            str(dataset.blocks[block_index][0][row_index])
            for block_index, row_index in dataset.lookup
        ]
    if hasattr(dataset, "ids"):
        return [str(sample_id) for sample_id in dataset.ids]
    raise ValueError("Dataset type not recognized: missing lookup/ids attribute")


def dataset_labels(dataset, indices: Sequence[int]) -> np.ndarray:
    """Return integer labels for dataset indices without loading embeddings."""
    sample_ids = dataset_sample_ids(dataset)
    label_map = getattr(dataset, "label_dict", None)
    if label_map is None:
        label_map = getattr(dataset, "label_map", None)
    if label_map is None:
        raise ValueError("Dataset type not recognized: missing label_dict/label_map")
    return np.asarray(
        [int(label_map[sample_ids[int(index)]]) for index in indices], dtype=int
    )


def _sample_signature(dataset, index: int) -> tuple[bytes, int]:
    embedding, label = dataset[int(index)]
    embedding_array = np.ascontiguousarray(embedding.detach().cpu().numpy())
    return hashlib.sha256(embedding_array).digest(), int(label)


def load_or_create_fingerprints(
    dataset,
    name: str,
    out_dir: str | Path,
) -> np.ndarray:
    """Cache one compact embedding fingerprint per dataset row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_fingerprints.npy"
    if out_path.exists():
        fingerprints = np.load(out_path, mmap_mode="r")
        if fingerprints.shape != (len(dataset), 32):
            raise ValueError(
                f"Fingerprint cache shape {fingerprints.shape} does not match "
                f"dataset length {len(dataset)}: {out_path}"
            )
        print(f"[{name}] Loaded cached embedding fingerprints ({len(dataset)})")
        return fingerprints

    fingerprints = np.empty((len(dataset), 32), dtype=np.uint8)
    for index in tqdm(range(len(dataset)), desc=f"[{name} fingerprints]"):
        digest, _label = _sample_signature(dataset, index)
        fingerprints[index] = np.frombuffer(digest, dtype=np.uint8)
    np.save(out_path, fingerprints)
    print(f"[{name}] Cached {len(dataset)} embedding fingerprints")
    return np.load(out_path, mmap_mode="r")


def _cached_signature(
    fingerprints: np.ndarray,
    labels: np.ndarray,
    index: int,
) -> tuple[bytes, int]:
    return fingerprints[int(index)].tobytes(), int(labels[int(index)])


def dedup_and_save_indices(
    dataset,
    name: str,
    out_dir: str | Path,
    fingerprints: np.ndarray | None = None,
) -> list[int]:
    """Deduplicate (embedding, label) pairs and cache the resulting indices."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_dedup_indices.npy"

    if out_path.exists():
        uniq_indices = np.load(out_path).tolist()
        print(f"[{name}] Loaded cached dedup indices ({len(dataset)} -> {len(uniq_indices)})")
        return uniq_indices

    if fingerprints is None:
        fingerprints = load_or_create_fingerprints(dataset, name, out_dir)
    labels = dataset_labels(dataset, range(len(dataset)))
    uniq_indices: list[int] = []
    seen: set[tuple[bytes, int]] = set()
    for i in tqdm(range(len(dataset)), desc=f"[{name}]"):
        key = _cached_signature(fingerprints, labels, i)
        if key not in seen:
            seen.add(key)
            uniq_indices.append(i)

    np.save(out_path, uniq_indices)
    reduction = len(dataset) - len(uniq_indices)
    print(
        f"[{name}] Deduplicated {len(dataset)} -> {len(uniq_indices)} "
        f"({reduction} removed, {100.0 * reduction / len(dataset):.1f}% reduction)"
    )
    return uniq_indices


def select_background_indices(
    labels: np.ndarray,
    background_fraction: float,
    max_background: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified background/explainer split.

    Matches ``SD-CNN/interpretability/utils.py::compute_shap_values_strat``:
    a ``background_fraction`` (default 20%) stratified split is drawn first,
    then the background is capped at ``max_background`` (default 160) with a
    second stratified subsample if needed. Falls back to a plain random split
    when a class has fewer than 2 members (stratification is impossible).
    """
    if not 0 < background_fraction < 1:
        raise ValueError("background_frac must be between 0 and 1")
    if max_background < 1:
        raise ValueError("max_background must be positive")

    labels = np.asarray(labels, dtype=int).reshape(-1)
    indices = np.arange(len(labels))
    class_counts = pd.Series(labels).value_counts()
    can_stratify = len(class_counts) > 1 and int(class_counts.min()) >= 2

    if can_stratify:
        background, explanation = train_test_split(
            indices,
            train_size=background_fraction,
            stratify=labels,
            random_state=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        background_size = min(
            len(indices) - 1,
            max(1, round(background_fraction * len(indices))),
        )
        background, explanation = (
            shuffled[:background_size],
            shuffled[background_size:],
        )

    if len(background) > max_background:
        background_labels = labels[background]
        background_counts = pd.Series(background_labels).value_counts()
        if len(background_counts) > 1 and int(background_counts.min()) >= 2:
            background, _ = train_test_split(
                background,
                train_size=max_background,
                stratify=background_labels,
                random_state=seed,
            )
        else:
            background = np.random.default_rng(seed).choice(
                background, size=max_background, replace=False
            )
        mask = np.ones(len(indices), dtype=bool)
        mask[background] = False
        explanation = indices[mask]

    if len(explanation) == 0:
        explanation = np.asarray([background[-1]])
        background = background[:-1]
    return np.asarray(background), np.asarray(explanation)


def select_background_and_explainer_indices(
    dataset,
    dedup_indices: Sequence[int],
    *,
    background_fraction: float,
    max_background: int,
    max_explain: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Select a disjoint background and explainer set for the chosen fold's model.

    Both sets are drawn together from the drug's deduplicated dataset with
    the SD-CNN random-split stratified method (``select_background_indices``:
    20% background capped at 160), then the remaining explainer pool is
    additionally capped at ``max_explain`` (360) using the same stratified
    subsampling style as ``lineage_aware_split``'s fixed explainer selection.
    """
    if max_explain < 1:
        raise ValueError("max_explain must be positive")
    dedup_indices = [int(index) for index in dedup_indices]
    labels = dataset_labels(dataset, dedup_indices)
    background_offsets, explanation_offsets = select_background_indices(
        labels, background_fraction, max_background, seed
    )

    if len(explanation_offsets) > max_explain:
        explanation_labels = labels[explanation_offsets]
        class_counts = pd.Series(explanation_labels).value_counts()
        if len(class_counts) > 1 and int(class_counts.min()) >= 2:
            try:
                explanation_offsets, _ = train_test_split(
                    explanation_offsets,
                    train_size=max_explain,
                    stratify=explanation_labels,
                    random_state=seed,
                )
            except ValueError:
                explanation_offsets = np.random.default_rng(seed).choice(
                    explanation_offsets, size=max_explain, replace=False
                )
        else:
            explanation_offsets = np.random.default_rng(seed).choice(
                explanation_offsets, size=max_explain, replace=False
            )

    background_indices = [dedup_indices[int(offset)] for offset in background_offsets]
    explanation_indices = [dedup_indices[int(offset)] for offset in explanation_offsets]
    return background_indices, explanation_indices


def sample_manifest(
    dataset,
    indices: Sequence[int],
    isolate_id_map: dict[int, str],
) -> pd.DataFrame:
    """Record Evo2 sample IDs, biological isolate IDs, and labels."""
    sample_ids = dataset_sample_ids(dataset)
    indices = [int(index) for index in indices]
    selected_ids = [sample_ids[index] for index in indices]
    source_rows = [int(sample_id.removeprefix("full_")) for sample_id in selected_ids]
    return pd.DataFrame(
        {
            "row_id": [isolate_id_map.get(source_row, "") for source_row in source_rows],
            "sample_id": selected_ids,
            "label": dataset_labels(dataset, indices),
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fold selection (best validation AUC, with a fallback for missing history)
# ──────────────────────────────────────────────────────────────────────────────


def _fold_number(fold_dir_name: str) -> str:
    return fold_dir_name.removeprefix("fold_")


def discover_folds(
    model_dir: str | Path,
    drug: str,
    embed_type: str,
    model_name: str = "DNABERTCNN",
    model_filename: str = "auto",
    model_seed: str = "42",
) -> dict[str, tuple[str, Path, str]]:
    """Return ``{fold: (seed, model_path, checkpoint_name)}`` for every fold
    directory that has a usable checkpoint.

    When ``model_filename`` is ``"auto"``, the final saved checkpoint
    (``<model_name>.pt``) is preferred over the early-stopping checkpoint
    (``<model_name>_best_model.pt``) -- matching the checkpoint-selection
    convention used for the zero-shot random-split evaluation jobs.
    """
    saved_models_dir = (
        Path(model_dir) / drug / "saved_models" / "evo2" / embed_type / drug / f"seed_{model_seed}"
    )
    if not saved_models_dir.is_dir():
        raise FileNotFoundError(f"Training output for {drug} not found: {saved_models_dir}")

    if model_filename == "auto":
        candidates = (f"{model_name}.pt", f"{model_name}_best_model.pt")
    else:
        candidates = (model_filename,)

    folds: dict[str, tuple[str, Path, str]] = {}
    for fold_dir in sorted(saved_models_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold = _fold_number(fold_dir.name)
        for candidate in candidates:
            model_path = fold_dir / candidate
            if model_path.is_file():
                folds[fold] = (model_seed, model_path, candidate)
                break

    if not folds:
        raise FileNotFoundError(
            f"No fold checkpoints found for {drug} below {saved_models_dir} "
            f"(tried {', '.join(candidates)})"
        )
    return folds


def _fold_sort_key(fold: str) -> tuple[int, int | str]:
    return (0, int(fold)) if fold.isdigit() else (1, fold)


def _history_best_val_auc(
    model_dir: str | Path,
    drug: str,
    embed_type: str,
    fold: str,
    model_name: str,
    model_seed: str,
) -> float | None:
    """Return the maximum ``val_auc`` in a fold's training history, or None."""
    history_path = (
        Path(model_dir)
        / drug
        / "classification_results"
        / "evo2"
        / embed_type
        / drug
        / f"seed_{model_seed}"
        / f"{model_name}_fold{fold}_history.csv"
    )
    if not history_path.is_file():
        return None
    history = pd.read_csv(history_path)
    if "val_auc" not in history.columns or history["val_auc"].dropna().empty:
        return None
    return float(history["val_auc"].max())


def select_best_fold(
    model_dir: str | Path,
    drug: str,
    embed_type: str,
    model_name: str = "DNABERTCNN",
    model_filename: str = "auto",
    model_seed: str = "42",
) -> dict:
    """Pick the fold with the highest validation AUC for ``drug``.

    Falls back to the first available fold (lowest fold number) when no
    fold has a training history with a usable ``val_auc`` column.
    """
    folds = discover_folds(model_dir, drug, embed_type, model_name, model_filename, model_seed)

    scored = []
    for fold, (seed_used, model_path, checkpoint_name) in folds.items():
        best_val_auc = _history_best_val_auc(
            model_dir, drug, embed_type, fold, model_name, seed_used
        )
        scored.append((fold, seed_used, model_path, checkpoint_name, best_val_auc))

    with_auc = [item for item in scored if item[-1] is not None]
    if with_auc:
        with_auc.sort(key=lambda item: (-item[-1], _fold_sort_key(item[0])))
        fold, seed_used, model_path, checkpoint_name, best_val_auc = with_auc[0]
        reason = (
            f"best val_auc={best_val_auc:.4f} among {len(with_auc)}/{len(scored)} "
            "fold(s) with usable training history"
        )
    else:
        scored.sort(key=lambda item: _fold_sort_key(item[0]))
        fold, seed_used, model_path, checkpoint_name, best_val_auc = scored[0]
        reason = (
            f"no fold history with a usable val_auc column found for {drug}; "
            f"falling back to fold {fold}"
        )

    return {
        "fold": fold,
        "seed": seed_used,
        "model_path": str(model_path),
        "model_filename": checkpoint_name,
        "best_val_auc": best_val_auc,
        "reason": reason,
        "folds_considered": sorted(folds, key=_fold_sort_key),
    }


def save_fold_selection(path: str | Path, selection: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)


def load_fold_selection(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_fold_model(
    model_path: str | Path,
    in_dim: int,
    seq_len: int,
    model_name: str = "DNABERTCNN",
):
    """Load the DNABERTCNN checkpoint for the chosen random-split fold."""
    model = get_model_class(model_name=model_name, in_dim=in_dim, seq_len=seq_len, device=DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_shap_for_model(
    model,
    full_dataset,
    background_indices: Sequence[int],
    explanation_indices: Sequence[int],
    per_gene_lengths: list[int],
    gene_names: list[str],
    batch_size: int = 4,
    background_batch_size: int = 16,
    device: str = DEVICE,
) -> pd.DataFrame:
    """Compute per-sample token-level SHAP importance for the chosen fold's model.

    Identical mechanics to
    ``lineage_aware_split/shap_utils.py::compute_shap_for_lineage_model``
    (mirrors ``DNABERT-2/interpretability/utils/shap_utils.py::shap_per_residue``).
    Returns a DataFrame with columns ``sample_idx``, ``label``,
    ``importance_full`` and (for multi-gene drugs) ``importance_<gene>``.
    """
    model = model.to(device).eval()
    bg_idx = [int(index) for index in background_indices]
    samp_idx = [int(index) for index in explanation_indices]
    if not bg_idx:
        raise ValueError("SHAP background cannot be empty")
    if not samp_idx:
        raise ValueError("SHAP explainer set cannot be empty")
    if batch_size < 1 or background_batch_size < 1:
        raise ValueError("SHAP batch sizes must be positive")
    print(f"Using background size {len(bg_idx)} from the deduplicated drug dataset")
    first_background, _ = full_dataset[bg_idx[0]]
    background = torch.empty(
        (len(bg_idx), *first_background.shape), dtype=first_background.dtype
    )
    background[0].copy_(first_background)
    for position, index in enumerate(bg_idx[1:], start=1):
        background[position].copy_(full_dataset[index][0])
    del first_background

    E = len(samp_idx)
    print(
        f"Explaining the fixed set of {E} samples in streaming batches "
        f"of {batch_size}; background batches = {background_batch_size}"
    )

    start_time = time()
    out: dict[str, list] = {
        "sample_idx": [],
        "label": [],
        "importance_full": [],
    }
    for gene in gene_names:
        out[f"importance_{gene}"] = []
    cuts = np.cumsum([0] + per_gene_lengths)

    for start in range(0, E, batch_size):
        chunk_indices = samp_idx[start : start + batch_size]
        chunk = torch.stack([full_dataset[i][0] for i in chunk_indices]).to(device)
        weighted_shap = None
        for background_start in range(0, len(bg_idx), background_batch_size):
            background_stop = min(
                background_start + background_batch_size, len(bg_idx)
            )
            background_chunk = background[background_start:background_stop].to(device)
            explainer = shap.DeepExplainer(Wrapped(model), [background_chunk])
            sv_chunk = np.asarray(
                explainer.shap_values([chunk], check_additivity=False)[0],
                dtype=np.float32,
            )
            chunk_weight = background_stop - background_start
            if weighted_shap is None:
                weighted_shap = sv_chunk * chunk_weight
            else:
                weighted_shap += sv_chunk * chunk_weight
            del explainer, background_chunk, sv_chunk
            torch.cuda.empty_cache()
        if weighted_shap is None:
            raise AssertionError("No SHAP background chunks were processed")
        weighted_shap /= len(bg_idx)
        importance = np.abs(weighted_shap).sum(axis=1)
        out["sample_idx"].extend(chunk_indices)
        out["label"].extend(int(full_dataset[i][1]) for i in chunk_indices)
        out["importance_full"].extend(importance)
        for gene_index, gene in enumerate(gene_names):
            out[f"importance_{gene}"].extend(
                importance[
                    :,
                    cuts[gene_index] : cuts[gene_index + 1],
                ]
            )
        del chunk, weighted_shap, importance
        torch.cuda.empty_cache()
        gc.collect()
    print(f"SHAP computation completed in {time() - start_time:.2f} seconds")

    return pd.DataFrame(out)


def rank_features_by_shap(shap_df: pd.DataFrame, gene_names: list[str]) -> pd.DataFrame:
    """Rank ``"{gene}_{position}"`` features by max |SHAP| across samples.

    Matches ``DNABERT-2/interpretability/run_shap_interpret.py``'s
    ``rank_positions_by_shap`` + ``rank_all_genes_for_drug``: importance for a
    position is the maximum absolute SHAP value observed across the explained
    isolates, and per-gene rankings are merged and re-sorted.
    """
    rows = []
    for gene in gene_names:
        col = f"importance_{gene}"
        if col not in shap_df.columns:
            continue
        values = np.stack([np.asarray(v).squeeze() for v in shap_df[col]], axis=0)
        max_abs_shap = np.abs(values).max(axis=0)
        for position, value in enumerate(max_abs_shap):
            rows.append({"gene": gene, "position": int(position), "max_abs_shap": float(value)})

    ranked = (
        pd.DataFrame(rows)
        .sort_values(
            ["max_abs_shap", "gene", "position"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    return ranked
