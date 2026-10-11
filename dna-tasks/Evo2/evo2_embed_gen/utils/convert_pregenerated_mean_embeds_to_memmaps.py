"""Convert pre-generated mean_dim/mean_seq embedding .npy files to memmaps for downstream training.

This script is used when mean_dim/mean_seq embeddings have already been generated
(e.g., via generate_evo2_embeddings.sh) and need to be converted to the memmap format
expected by downstream training jobs.

Unlike prepare_memmaps.py (which pools token memmaps), this script converts
already-aggregated .npy batch files directly to memmaps.
"""

from __future__ import annotations

import argparse
import gc
import glob
import re
from pathlib import Path

import numpy as np

from evo2_downstream.config import memmap_root
from evo2_embed_gen.data.locus_order import locus_order


BATCH_INDEX_RE = re.compile(r"_batch_(\d+)\.npy$")


def _batch_key(path: str) -> int:
    match = BATCH_INDEX_RE.search(path)
    if not match:
        raise ValueError(f"Could not parse batch index from {path}")
    return int(match.group(1))


def _batch_offset_map(npy_files: list[str]) -> dict[str, int]:
    """Map each batch file to its global sample offset in the original order."""
    offsets: dict[str, int] = {}
    sample_offset = 0
    for npy_path in npy_files:
        batch = np.load(npy_path, mmap_mode="r")
        offsets[npy_path] = sample_offset
        sample_offset += int(batch.shape[0])
    return offsets


def parse_gene_list(genes: str) -> list[str]:
    if genes.strip().lower() == "all":
        return list(locus_order)
    requested = [gene.strip() for gene in genes.split(",") if gene.strip()]
    missing = [gene for gene in requested if gene not in locus_order]
    if missing:
        raise ValueError(f"Unknown gene(s): {missing}. Valid genes: {locus_order}")
    return requested


def convert_mean_embed_batches_to_memmaps(
    raw_embed_root: Path,
    output_root: Path,
    gene: str,
    embed_type: str,
    prefix: str = "full",
) -> None:
    """Convert pre-generated mean_dim/mean_seq `.npy` batches into the memmap layout.
    
    Args:
        raw_embed_root: Path to directory containing <gene>/<prefix>_embeddings_batch_*.npy files
        output_root: Path to output directory for memmaps
        gene: Gene name (e.g., 'embA', 'inhA')
        embed_type: Either 'mean_dim' or 'mean_seq'
        prefix: Identifier prefix used for downstream labels (default: 'full')
    """
    if embed_type not in {"mean_dim", "mean_seq"}:
        raise ValueError(f"embed_type must be 'mean_dim' or 'mean_seq', got {embed_type}")

    src_glob = raw_embed_root / gene / f"zs_{prefix}_embeddings_batch_*.npy"
    npy_files = sorted(glob.glob(str(src_glob)), key=_batch_key)
    if not npy_files:
        raise FileNotFoundError(f"No {embed_type} batch files found for {gene} at {src_glob}")

    batch_offsets = _batch_offset_map(npy_files)

    gene_out_dir = output_root / gene
    gene_out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Converting {embed_type} gene {gene} with {len(npy_files)} batches"
    )

    for npy_path in npy_files:
        batch_path = Path(npy_path)
        base_name = batch_path.stem
        mmap_out = gene_out_dir / f"{base_name}_{embed_type}.mmap"
        meta_out = gene_out_dir / f"{base_name}_{embed_type}_meta.npz"

        if mmap_out.exists() and meta_out.exists():
            print(f"[INFO] Skipping existing {embed_type} memmap: {base_name}")
            continue

        batch = np.load(batch_path, mmap_mode="r")
        
        # Validate shape: should be (N, L, D) for mean_seq or (N, L, 1) for mean_dim
        if batch.ndim != 3:
            raise ValueError(
                f"Expected {embed_type} batch shape (N, L, D) for {batch_path}, got {batch.shape}"
            )

        # Convert to float16 for storage efficiency
        embeddings = batch.astype("float16", copy=False)
        
        # Write memmap
        mm = np.memmap(mmap_out, mode="w+", dtype="float16", shape=embeddings.shape)
        mm[:] = embeddings
        mm.flush()
        del mm

        # Create metadata
        batch_size = int(embeddings.shape[0])
        sample_offset = batch_offsets[npy_path]
        identifiers = np.array(
            [f"{prefix}_{idx:06d}" for idx in range(sample_offset, sample_offset + batch_size)],
            dtype="<U32",
        )

        np.savez_compressed(
            meta_out,
            shape=np.asarray(embeddings.shape, dtype=np.int64),
            mmap_path=mmap_out.name,
            identifier=identifiers,
        )

        print(f"[INFO] Converted {base_name}: shape {embeddings.shape} -> {mmap_out.name}")
        del embeddings
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pre-generated mean_dim/mean_seq embeddings to memmaps for downstream training"
    )
    parser.add_argument("--genes", type=str, default="all", help="Comma-separated gene list or 'all'")
    parser.add_argument(
        "--embed_types",
        nargs="+",
        default=["mean_dim", "mean_seq"],
        choices=["mean_dim", "mean_seq"],
        help="Embedding types to convert (pre-generated .npy files)",
    )
    parser.add_argument("--prefix", type=str, default="full", help="Identifier prefix used for downstream labels")
    parser.add_argument(
        "--mean_dim_raw_embed_root",
        type=str,
        default="/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings/zero-shot/mean_dim/layer20/full",
        help="Path to directory containing pre-generated mean_dim .npy batches",
    )
    parser.add_argument(
        "--mean_seq_raw_embed_root",
        type=str,
        default="/scratch/workspace/saishradhamo_umass_edu-big-tb/evo2/embeddings/zero-shot/mean_seq/layer20/full",
        help="Path to directory containing pre-generated mean_seq .npy batches",
    )
    parser.add_argument(
        "--mean_dim_memmap_root",
        type=str,
        default=str(memmap_root("mean_dim")),
        help="Output path for mean_dim memmaps",
    )
    parser.add_argument(
        "--mean_seq_memmap_root",
        type=str,
        default=str(memmap_root("mean_seq")),
        help="Output path for mean_seq memmaps",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    genes = parse_gene_list(args.genes)

    if "mean_dim" in args.embed_types:
        mean_dim_raw_root = Path(args.mean_dim_raw_embed_root)
        mean_dim_out_root = Path(args.mean_dim_memmap_root)
        for gene in genes:
            convert_mean_embed_batches_to_memmaps(
                mean_dim_raw_root,
                mean_dim_out_root,
                gene,
                embed_type="mean_dim",
                prefix=args.prefix,
            )

    if "mean_seq" in args.embed_types:
        mean_seq_raw_root = Path(args.mean_seq_raw_embed_root)
        mean_seq_out_root = Path(args.mean_seq_memmap_root)
        for gene in genes:
            convert_mean_embed_batches_to_memmaps(
                mean_seq_raw_root,
                mean_seq_out_root,
                gene,
                embed_type="mean_seq",
                prefix=args.prefix,
            )


if __name__ == "__main__":
    main(build_parser().parse_args())
