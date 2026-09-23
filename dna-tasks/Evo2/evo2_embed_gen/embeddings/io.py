"""Output helpers for Evo2 embedding batches."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _atomic_save(path: str, array: np.ndarray) -> None:
    """Write a NumPy array completely before exposing the final batch filename."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def save_batch(
    save_path: str,
    data_partition: str,
    batch_index: int,
    embeddings: np.ndarray,
    phenotypes: np.ndarray,
    isolate_ids: list[str],
) -> None:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    _atomic_save(
        os.path.join(save_path, f"zs_{data_partition}_embeddings_batch_{batch_index}.npy"),
        embeddings,
    )
    _atomic_save(
        os.path.join(save_path, f"zs_{data_partition}_res_phenotypes_batch_{batch_index}.npy"),
        phenotypes.astype(np.int16, copy=False),
    )
    _atomic_save(
        os.path.join(save_path, f"zs_{data_partition}_isolate_ids_batch_{batch_index}.npy"),
        np.asarray(isolate_ids, dtype=object),
    )


def write_metadata(save_path: str, metadata: dict[str, Any]) -> None:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _batch_index(path: str) -> int:
    """Extract the numeric batch index from a batch file path for correct numeric sort order."""
    import re
    m = re.search(r"_batch_(\d+)\.npy$", path)
    if not m:
        raise ValueError(f"Could not parse batch index from path: {path}")
    return int(m.group(1))


def stack_final_phenotypes(embed_dir: str, data_partition: str, gene: str) -> None:
    phenotype_files = sorted(
        glob.glob(os.path.join(embed_dir, gene, f"zs_{data_partition}_res_phenotypes_batch_*.npy")),
        key=_batch_index,
    )
    if not phenotype_files:
        raise FileNotFoundError(f"No phenotype batch files found in {os.path.join(embed_dir, gene)}")

    first_phenotypes = np.load(phenotype_files[0])
    total_samples = sum(np.load(path, mmap_mode="r").shape[0] for path in phenotype_files)
    final_phenotypes = np.empty((total_samples, first_phenotypes.shape[1]), dtype=np.int16)

    current_index = 0
    for phenotype_file in phenotype_files:
        batch = np.load(phenotype_file)
        batch_size = batch.shape[0]
        final_phenotypes[current_index : current_index + batch_size] = batch
        current_index += batch_size

    save_path = os.path.join(embed_dir, f"zs_{data_partition}_stacked_phenotypes.npz")
    np.savez_compressed(save_path, phenotypes=final_phenotypes)
    print(f"Stacked phenotypes saved at {save_path}, shape: {final_phenotypes.shape}")
