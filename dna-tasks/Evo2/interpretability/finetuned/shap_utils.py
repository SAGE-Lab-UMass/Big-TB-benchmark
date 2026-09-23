"""SHAP utilities for lineage-aware finetuned Evo2 models."""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
EVO2_DIR = THIS_DIR.parents[1]
FINETUNE_DIR = EVO2_DIR / "finetuning" / "lineage_holdout"
for path in (EVO2_DIR, FINETUNE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evo2_embed_gen.model.evo2_model import Evo2Embedder, Evo2ModelConfig  # noqa: E402
from finetuning.lineage_holdout.train_evo2_lora import (  # noqa: E402
    Evo2LoRAClassifier,
    load_adapter_and_classifier,
)
from utils.lineage_split import load_isolate_id_map, load_lineage_map  # noqa: E402
from utils.lora_data import (  # noqa: E402
    apply_lineage_split,
    create_validation_split,
    load_sequences_from_fasta,
)

try:
    from finetuning.modules.dataloader.locus_order import DRUG_TO_LOCI  # noqa: E402
except ImportError:  # pragma: no cover
    from dataloader.locus_order import DRUG_TO_LOCI  # type: ignore  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SequenceTable:
    isolate_ids: list[str]
    sequences: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.sequences[int(index)], int(self.labels[int(index)])


class ClassifierWrapper(torch.nn.Module):
    """Expose the CNN head as a SHAP-compatible tensor model."""

    def __init__(self, classifier: torch.nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classifier_dtype = next(self.classifier.parameters()).dtype
        if x.dtype != classifier_dtype:
            x = x.to(dtype=classifier_dtype)
        return self.classifier(x).unsqueeze(1)


def _gene_token_length(fasta_dir: Path, gene: str) -> int:
    gene_sequences = load_sequences_from_fasta(gene, fasta_dir)
    if not gene_sequences:
        raise ValueError(f"No sequences loaded for {gene} from {fasta_dir}")
    return len(next(iter(gene_sequences.values())))


def build_sequence_table(
    drug: str,
    geno_pheno_csv: str | Path,
    fasta_dir: str | Path,
) -> tuple[SequenceTable, list[int], list[str]]:
    fasta_dir = Path(fasta_dir)
    df_labels = pd.read_csv(geno_pheno_csv, index_col=0, low_memory=False)
    if drug not in df_labels.columns:
        raise ValueError(f"Drug {drug} not found in {geno_pheno_csv}")

    gene_names = list(DRUG_TO_LOCI[drug])
    gene_sequences = {gene: load_sequences_from_fasta(gene, fasta_dir) for gene in gene_names}
    per_gene_lengths = [_gene_token_length(fasta_dir, gene) for gene in gene_names]

    isolate_ids: list[str] = []
    sequences: list[str] = []
    labels: list[int] = []
    for isolate_id in df_labels.index:
        isolate_id_str = str(isolate_id)
        parts = []
        for gene in gene_names:
            sequence = gene_sequences[gene].get(isolate_id_str)
            if sequence is None:
                parts = []
                break
            parts.append(sequence)
        if not parts:
            continue

        label_val = df_labels.loc[isolate_id, drug]
        if label_val not in ["R", "S", 0, 1]:
            continue
        isolate_ids.append(isolate_id_str)
        sequences.append("".join(parts))
        labels.append(1 if label_val in ["S", 1] else 0)

    table = SequenceTable(isolate_ids=isolate_ids, sequences=sequences, labels=labels)
    print(f"{drug}: loaded {len(table)} labelled sequences across genes {gene_names}")
    print(f"{drug}: R={(np.asarray(labels) == 0).sum()} S={(np.asarray(labels) == 1).sum()}")
    return table, per_gene_lengths, gene_names


def dataset_labels(dataset: SequenceTable, indices: Iterable[int]) -> np.ndarray:
    return np.asarray([dataset.labels[int(index)] for index in indices], dtype=int)


def _sample_signature(dataset: SequenceTable, index: int) -> tuple[bytes, int]:
    sequence, label = dataset[index]
    return hashlib.sha256(sequence.encode("ascii", errors="ignore")).digest(), int(label)


def load_or_create_fingerprints(dataset: SequenceTable, name: str, out_dir: str | Path) -> np.ndarray:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_sequence_fingerprints.npy"
    if out_path.exists():
        fingerprints = np.load(out_path, mmap_mode="r")
        if fingerprints.shape != (len(dataset), 32):
            raise ValueError(
                f"Fingerprint cache shape {fingerprints.shape} does not match dataset length {len(dataset)}: {out_path}"
            )
        print(f"[{name}] Loaded cached sequence fingerprints ({len(dataset)})")
        return fingerprints

    fingerprints = np.empty((len(dataset), 32), dtype=np.uint8)
    for index in tqdm(range(len(dataset)), desc=f"[{name} fingerprints]"):
        digest, _label = _sample_signature(dataset, index)
        fingerprints[index] = np.frombuffer(digest, dtype=np.uint8)
    np.save(out_path, fingerprints)
    print(f"[{name}] Cached {len(dataset)} sequence fingerprints")
    return np.load(out_path, mmap_mode="r")


def _cached_signature(fingerprints: np.ndarray, labels: np.ndarray, index: int) -> tuple[bytes, int]:
    return fingerprints[int(index)].tobytes(), int(labels[int(index)])


def dedup_and_save_indices(
    dataset: SequenceTable,
    name: str,
    out_dir: str | Path,
    fingerprints: np.ndarray | None = None,
) -> list[int]:
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
    seen: set[tuple[bytes, int]] = set()
    uniq_indices: list[int] = []
    for index in tqdm(range(len(dataset)), desc=f"[{name}]"):
        key = _cached_signature(fingerprints, labels, index)
        if key not in seen:
            seen.add(key)
            uniq_indices.append(index)
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
        background_size = min(len(indices) - 1, max(1, round(background_fraction * len(indices))))
        background, explanation = shuffled[:background_size], shuffled[background_size:]

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
            background = np.random.default_rng(seed).choice(background, size=max_background, replace=False)
        mask = np.ones(len(indices), dtype=bool)
        mask[background] = False
        explanation = indices[mask]

    if len(explanation) == 0:
        explanation = np.asarray([background[-1]])
        background = background[:-1]
    return np.asarray(background), np.asarray(explanation)


def select_fixed_explainer_indices(
    dataset: SequenceTable,
    dedup_indices: Sequence[int],
    *,
    background_fraction: float,
    max_background: int,
    max_explain: int,
    seed: int,
) -> list[int]:
    if max_explain < 1:
        raise ValueError("max_explain must be positive")
    dedup_indices = [int(index) for index in dedup_indices]
    labels = dataset_labels(dataset, dedup_indices)
    _background_offsets, explanation_offsets = select_background_indices(
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
    dataset: SequenceTable,
    training_indices: Sequence[int],
    explanation_indices: Sequence[int],
    *,
    max_background: int,
    seed: int,
    fingerprints: np.ndarray | None = None,
) -> tuple[list[int], int]:
    if max_background < 1:
        raise ValueError("max_background must be positive")

    training_indices = [int(index) for index in training_indices]
    explanation_indices = [int(index) for index in explanation_indices]
    explanation_ids = {dataset.isolate_ids[index] for index in explanation_indices}
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
            dataset.isolate_ids[index] in explanation_ids
            or sample_signature in explanation_signatures
            or sample_signature in seen
        ):
            continue
        seen.add(sample_signature)
        unique_indices.append(index)

    candidate_count = len(unique_indices)
    if candidate_count == 0:
        raise ValueError("No unique training background samples remain after excluding the fixed explainer")

    if candidate_count > max_background:
        labels = dataset_labels(dataset, unique_indices)
        offsets = np.arange(candidate_count)
        class_counts = pd.Series(labels).value_counts()
        if len(class_counts) > 1 and int(class_counts.min()) >= 2:
            try:
                selected_offsets, _ = train_test_split(
                    offsets,
                    train_size=max_background,
                    stratify=labels,
                    random_state=seed,
                )
            except ValueError:
                selected_offsets = np.random.default_rng(seed).choice(offsets, size=max_background, replace=False)
        else:
            selected_offsets = np.random.default_rng(seed).choice(offsets, size=max_background, replace=False)
        background_indices = [unique_indices[int(offset)] for offset in selected_offsets]
    else:
        background_indices = unique_indices

    background_signatures = [signature(index) for index in background_indices]
    if len(background_signatures) != len(set(background_signatures)):
        raise AssertionError("Background contains duplicate sequence/phenotype signatures")
    if set(background_signatures) & explanation_signatures:
        raise AssertionError("Background and explainer signatures overlap")
    if {dataset.isolate_ids[index] for index in background_indices} & explanation_ids:
        raise AssertionError("Background and explainer sample IDs overlap")
    if len(background_indices) > max_background:
        raise AssertionError("Background exceeds max_background")
    return background_indices, candidate_count


def sample_manifest(dataset: SequenceTable, indices: Sequence[int]) -> pd.DataFrame:
    indices = [int(index) for index in indices]
    return pd.DataFrame(
        {
            "row_id": [dataset.isolate_ids[index] for index in indices],
            "sample_id": [dataset.isolate_ids[index] for index in indices],
            "label": dataset_labels(dataset, indices),
        }
    )


def discover_eligible_lineages(
    model_dir: str | Path,
    drug: str,
    require_test_metrics: bool = True,
) -> list[tuple[str, Path]]:
    drug_root = Path(model_dir) / drug / "final"
    if not drug_root.is_dir():
        raise FileNotFoundError(f"Finetuned output for {drug} not found: {drug_root}")

    eligible: list[tuple[str, Path]] = []
    for lineage_dir in sorted(drug_root.glob("heldout_lineage_*")):
        if not lineage_dir.is_dir():
            continue
        lineage = lineage_dir.name.removeprefix("heldout_lineage_")
        checkpoint_dir = lineage_dir / "best"
        required = [
            checkpoint_dir / "training_config.json",
            checkpoint_dir / "lora_adapter.pt",
            checkpoint_dir / "classifier_head.pt",
        ]
        if require_test_metrics:
            required.append(lineage_dir / "test_metrics.json")
        if all(path.is_file() for path in required):
            eligible.append((lineage, checkpoint_dir))

    def sort_key(item: tuple[str, Path]) -> tuple[int, int | str]:
        lineage = item[0]
        return (0, int(lineage)) if lineage.isdigit() else (1, lineage)

    eligible.sort(key=sort_key)
    if not eligible:
        raise ValueError(f"No eligible finetuned held-out lineages for {drug} below {drug_root}")
    return eligible


def reconstruct_model_training_indices(
    dataset: SequenceTable,
    drug: str,
    heldout_lineage: str,
    geno_pheno_csv: str | Path,
    lineage_csv: str | Path,
    val_frac: float,
    seed: int,
) -> list[int]:
    isolate_id_map = load_isolate_id_map(str(geno_pheno_csv))
    lineage_map = load_lineage_map(str(lineage_csv))
    train_indices, _test_indices = apply_lineage_split(
        dataset.isolate_ids,
        dataset.sequences,
        dataset.labels,
        str(heldout_lineage),
        isolate_id_map,
        lineage_map,
    )
    train_indices, _val_indices = create_validation_split(
        train_indices,
        dataset.labels,
        val_frac=val_frac,
        random_seed=seed,
    )
    return [int(index) for index in train_indices]


def load_finetuned_model(checkpoint_dir: str | Path, device: str = DEVICE) -> Evo2LoRAClassifier:
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "training_config.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    evo2_config = Evo2ModelConfig(
        model_name=metadata.get("base_evo2_model", "evo2_7b"),
        layer_name=metadata.get("evo2_layer", "blocks.20.mlp.l3"),
        max_length=int(metadata.get("max_length", metadata.get("seq_len", 5000))),
        pad_char=metadata.get("pad_char", "N"),
        use_kernels=bool(metadata.get("use_kernels", False)),
    )
    evo2_embedder = Evo2Embedder(evo2_config)
    lora_config = metadata["lora"]
    model = Evo2LoRAClassifier(
        evo2_embedder,
        lora_config,
        seq_len=int(metadata.get("seq_len", evo2_config.max_length)),
        hidden_dim=int(metadata["hidden_dim"]) if metadata.get("hidden_dim") is not None else None,
        enable_gradient_checkpointing=False,
    )
    load_adapter_and_classifier(model, checkpoint_dir, map_location=device)
    model.to(device)
    model.eval()
    return model


def _hidden_dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported hidden dtype: {name}")


def materialize_hidden_inputs(
    model: Evo2LoRAClassifier,
    dataset: SequenceTable,
    indices: Sequence[int],
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: str,
    use_amp: bool = True,
) -> torch.Tensor:
    indices = [int(index) for index in indices]
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc="Materializing hidden inputs"):
            batch_indices = indices[start : start + batch_size]
            sequences = [dataset.sequences[index] for index in batch_indices]
            with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.startswith("cuda"), dtype=torch.bfloat16):
                hidden = model._generate_embeddings(sequences)
            hidden = hidden.transpose(1, 2).detach().to(dtype=dtype).cpu()
            chunks.append(hidden)
    return torch.cat(chunks, dim=0)


def compute_shap_for_finetuned_model(
    model: Evo2LoRAClassifier,
    dataset: SequenceTable,
    background_indices: Sequence[int],
    explanation_indices: Sequence[int],
    per_gene_lengths: list[int],
    gene_names: list[str],
    *,
    hidden_batch_size: int = 1,
    shap_batch_size: int = 1,
    background_batch_size: int = 8,
    hidden_dtype: str = "float16",
    device: str = DEVICE,
) -> pd.DataFrame:
    bg_idx = [int(index) for index in background_indices]
    exp_idx = [int(index) for index in explanation_indices]
    if not bg_idx:
        raise ValueError("SHAP background cannot be empty")
    if not exp_idx:
        raise ValueError("SHAP explainer set cannot be empty")

    dtype = _hidden_dtype_from_name(hidden_dtype)
    print(f"Materializing {len(bg_idx)} background hidden tensors ({hidden_dtype})")
    background = materialize_hidden_inputs(
        model,
        dataset,
        bg_idx,
        batch_size=hidden_batch_size,
        dtype=dtype,
        device=device,
    )

    wrapped_classifier = ClassifierWrapper(model.classifier).to(device).eval()
    cuts = np.cumsum([0] + per_gene_lengths)
    out: dict[str, list] = {"sample_idx": [], "label": [], "importance_full": []}
    for gene in gene_names:
        out[f"importance_{gene}"] = []

    start_time = time()
    for start in range(0, len(exp_idx), shap_batch_size):
        chunk_indices = exp_idx[start : start + shap_batch_size]
        explain_hidden = materialize_hidden_inputs(
            model,
            dataset,
            chunk_indices,
            batch_size=hidden_batch_size,
            dtype=dtype,
            device=device,
        ).to(device)

        weighted_shap = None
        for background_start in range(0, len(bg_idx), background_batch_size):
            background_stop = min(background_start + background_batch_size, len(bg_idx))
            background_chunk = background[background_start:background_stop].to(device)
            explainer = shap.DeepExplainer(wrapped_classifier, [background_chunk])
            sv_chunk = np.asarray(
                explainer.shap_values([explain_hidden], check_additivity=False)[0],
                dtype=np.float32,
            )
            chunk_weight = background_stop - background_start
            weighted_shap = sv_chunk * chunk_weight if weighted_shap is None else weighted_shap + sv_chunk * chunk_weight
            del explainer, background_chunk, sv_chunk
            torch.cuda.empty_cache()

        if weighted_shap is None:
            raise AssertionError("No SHAP background chunks were processed")
        weighted_shap /= len(bg_idx)
        importance = np.abs(weighted_shap).sum(axis=1)
        out["sample_idx"].extend(chunk_indices)
        out["label"].extend(int(dataset.labels[index]) for index in chunk_indices)
        out["importance_full"].extend(importance)
        for gene_index, gene in enumerate(gene_names):
            out[f"importance_{gene}"].extend(
                importance[:, cuts[gene_index] : cuts[gene_index + 1]]
            )
        del explain_hidden, weighted_shap, importance
        torch.cuda.empty_cache()
        gc.collect()

    print(f"SHAP computation completed in {time() - start_time:.2f} seconds")
    return pd.DataFrame(out)


def rank_features_by_shap(shap_df: pd.DataFrame, gene_names: list[str]) -> pd.DataFrame:
    rows = []
    for gene in gene_names:
        col = f"importance_{gene}"
        if col not in shap_df.columns:
            continue
        values = np.stack([np.asarray(value).squeeze() for value in shap_df[col]], axis=0)
        max_abs_shap = np.abs(values).max(axis=0)
        for position, value in enumerate(max_abs_shap):
            rows.append({"gene": gene, "position": int(position), "max_abs_shap": float(value)})
    return (
        pd.DataFrame(rows)
        .sort_values(["max_abs_shap", "gene", "position"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def select_explainer_shard(explanation_indices: Iterable[int], shard_index: int, shard_count: int) -> list[int]:
    explanation_indices = [int(index) for index in explanation_indices]
    if shard_count < 1:
        raise ValueError("explainer shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"explainer shard index {shard_index} is outside [0, {shard_count})")
    shard = explanation_indices[shard_index::shard_count]
    if not shard:
        raise ValueError(f"Explainer shard {shard_index} is empty for {len(explanation_indices)} samples")
    return shard
