"""SHAP value computation for Evo2 DNABERTCNN lineage-aware models.

Adapted from ``DNABERT-2/interpretability/utils/shap_utils.py`` and
``utils/model_utils.py``. The SHAP mechanics (DeepExplainer over a CNN that
consumes token embeddings shaped ``(D, L)``) are unchanged. The only real
differences are input-data specific:

- Evo2 embeddings are loaded through the vendored
  ``finetuning/modules`` dataset classes (``TokenMemmapMap`` /
  ``MultiGeneConcatDataset``) instead of the DNABERT-2 memmap loaders.
- Evo2 tokenizes one nucleotide per token (no BPE merging), so a SHAP
  importance at token position ``i`` of gene ``g`` already corresponds
  directly to the WHO/VCF gapped-alignment feature ``"{g}_{i}"`` -- no
  tokenizer offset-mapping step (as DNABERT-2 needs) is required.
- For the lineage-aware setting, a specific per-lineage model checkpoint is
  loaded (instead of one model per drug).
"""

from __future__ import annotations

import gc
import glob
import hashlib
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


def _sample_signatures(dataset, indices: Sequence[int]) -> list[tuple[bytes, int]]:
    """Return compact (embedding, phenotype) fingerprints in index order."""
    return [_sample_signature(dataset, int(index)) for index in indices]


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
    """Match Regression's stratified background/explainer split."""
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


def select_fixed_explainer_indices(
    dataset,
    dedup_indices: Sequence[int],
    *,
    background_fraction: float,
    max_background: int,
    max_explain: int,
    seed: int,
) -> list[int]:
    """Select one capped drug-level explainer from deduplicated samples."""
    if max_explain < 1:
        raise ValueError("max_explain must be positive")
    dedup_indices = [int(index) for index in dedup_indices]
    labels = dataset_labels(dataset, dedup_indices)
    _, explanation_offsets = select_background_indices(
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
    return [dedup_indices[int(offset)] for offset in explanation_offsets]


def select_lineage_background_indices(
    dataset,
    training_indices: Sequence[int],
    explanation_indices: Sequence[int],
    *,
    max_background: int,
    seed: int,
    fingerprints: np.ndarray | None = None,
) -> tuple[list[int], int]:
    """Select a unique training-only background disjoint from the explainer."""
    if max_background < 1:
        raise ValueError("max_background must be positive")

    sample_ids = dataset_sample_ids(dataset)
    training_indices = [int(index) for index in training_indices]
    explanation_indices = [int(index) for index in explanation_indices]
    explanation_ids = {sample_ids[index] for index in explanation_indices}
    all_labels = dataset_labels(dataset, range(len(dataset)))

    def signature(index: int) -> tuple[bytes, int]:
        if fingerprints is None:
            return _sample_signature(dataset, index)
        return _cached_signature(fingerprints, all_labels, index)

    explanation_signatures = {signature(index) for index in explanation_indices}

    seen: set[tuple[bytes, int]] = set()
    unique_indices: list[int] = []
    for index in training_indices:
        sample_signature = signature(index)
        if (
            sample_ids[index] in explanation_ids
            or sample_signature in explanation_signatures
            or sample_signature in seen
        ):
            continue
        seen.add(sample_signature)
        unique_indices.append(index)

    candidate_count = len(unique_indices)
    if candidate_count == 0:
        raise ValueError(
            "No unique training background samples remain after excluding "
            "the fixed explainer"
        )

    if candidate_count > max_background:
        labels = dataset_labels(dataset, unique_indices)
        offsets = np.arange(candidate_count)
        class_counts = pd.Series(labels).value_counts()
        can_stratify = len(class_counts) > 1 and int(class_counts.min()) >= 2
        if can_stratify:
            try:
                selected_offsets, _ = train_test_split(
                    offsets,
                    train_size=max_background,
                    stratify=labels,
                    random_state=seed,
                )
            except ValueError:
                selected_offsets = np.random.default_rng(seed).choice(
                    offsets, size=max_background, replace=False
                )
        else:
            selected_offsets = np.random.default_rng(seed).choice(
                offsets, size=max_background, replace=False
            )
        background_indices = [
            unique_indices[int(offset)] for offset in selected_offsets
        ]
    else:
        background_indices = unique_indices

    background_signatures = [signature(index) for index in background_indices]
    if len(background_signatures) != len(set(background_signatures)):
        raise AssertionError(
            "Background contains duplicate embedding/phenotype signatures"
        )
    if set(background_signatures) & explanation_signatures:
        raise AssertionError("Background and explainer signatures overlap")
    if not set(background_indices).issubset(set(training_indices)):
        raise AssertionError("Background contains rows outside the model training set")
    if {
        sample_ids[index] for index in background_indices
    } & explanation_ids:
        raise AssertionError("Background and explainer sample IDs overlap")
    if len(background_indices) > max_background:
        raise AssertionError("Background exceeds max_background")

    return background_indices, candidate_count


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


def load_lineage_model(
    model_path: str | Path,
    in_dim: int,
    seq_len: int,
    model_name: str = "DNABERTCNN",
):
    """Load the DNABERTCNN checkpoint trained for one held-out lineage."""
    model = get_model_class(model_name=model_name, in_dim=in_dim, seq_len=seq_len, device=DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_shap_for_lineage_model(
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
    """Compute per-sample token-level SHAP importance for one lineage's model.

    Mirrors ``DNABERT-2/interpretability/utils/shap_utils.py::shap_per_residue``.
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
    print(f"Using background size {len(bg_idx)} from the model training set")
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
