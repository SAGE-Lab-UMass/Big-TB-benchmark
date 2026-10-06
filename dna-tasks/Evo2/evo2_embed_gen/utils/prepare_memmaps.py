"""Prepare generated Evo2 embedding batches for downstream consumers."""

from __future__ import annotations

import argparse
import gc
import glob
import re
from pathlib import Path

import numpy as np

from evo2_downstream.config import DEFAULT_PHENOTYPE_LABEL_PATH, RAW_TOKEN_EMBED_ROOT, memmap_root
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


def convert_token_batches_to_memmaps(
    raw_embed_root: Path,
    output_root: Path,
    gene: str,
    prefix: str = "full",
    num_shards: int = 1,
    shard_index: int = 0,
) -> None:
    """Convert raw token `.npy` batches into the memmap layout expected downstream."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    src_glob = raw_embed_root / gene / f"zs_{prefix}_embeddings_batch_*.npy"
    npy_files = sorted(glob.glob(str(src_glob)), key=_batch_key)
    if not npy_files:
        raise FileNotFoundError(f"No token batch files found for {gene} at {src_glob}")

    batch_offsets = _batch_offset_map(npy_files)
    shard_files = [path for index, path in enumerate(npy_files) if index % num_shards == shard_index]
    if not shard_files:
        print(f"[INFO] No batches assigned to shard {shard_index}/{num_shards} for gene {gene}")
        return

    gene_out_dir = output_root / gene
    gene_out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Converting gene {gene} shard {shard_index + 1}/{num_shards} "
        f"with {len(shard_files)} of {len(npy_files)} total batches"
    )

    for npy_path in shard_files:
        batch_path = Path(npy_path)
        base_name = batch_path.stem
        mmap_out = gene_out_dir / f"{base_name}_token.mmap"
        meta_out = gene_out_dir / f"{base_name}_token_meta.npz"

        if mmap_out.exists() and meta_out.exists():
            continue

        batch = np.load(batch_path, mmap_mode="r")
        if batch.ndim != 4 or batch.shape[1] != 1:
            raise ValueError(f"Expected token batch shape (N, 1, L, D) for {batch_path}, got {batch.shape}")

        tokens = batch[:, 0].astype("float16", copy=False)
        mm = np.memmap(mmap_out, mode="w+", dtype="float16", shape=tokens.shape)
        mm[:] = tokens
        mm.flush()
        del mm

        batch_size = int(tokens.shape[0])
        sample_offset = batch_offsets[npy_path]
        identifiers = np.array(
            [f"{prefix}_{idx:06d}" for idx in range(sample_offset, sample_offset + batch_size)],
            dtype="<U32",
        )

        np.savez_compressed(
            meta_out,
            shape=np.asarray(tokens.shape, dtype=np.int64),
            mmap_path=mmap_out.name,
            identifier=identifiers,
        )

        del tokens
        gc.collect()


def project_gene_to_mean(
    token_memmap_root: Path,
    output_root: Path,
    gene: str,
    mean_method: str,
) -> None:
    """Project token memmaps to `mean_dim` or `mean_seq` memmaps."""
    if mean_method not in {"mean_dim", "mean_seq"}:
        raise ValueError(f"Unsupported mean method: {mean_method}")

    token_meta_paths = sorted((token_memmap_root / gene).glob("*_token_meta.npz"))
    if not token_meta_paths:
        raise FileNotFoundError(f"No token meta files found for {gene} in {token_memmap_root / gene}")

    gene_out_dir = output_root / gene
    gene_out_dir.mkdir(parents=True, exist_ok=True)

    for meta_path in token_meta_paths:
        meta_in = np.load(meta_path, allow_pickle=True)
        mmap_path = meta_path.with_name(meta_path.name.replace("_meta.npz", ".mmap"))
        mm_in = np.memmap(mmap_path, dtype="float16", mode="r", shape=tuple(meta_in["shape"]))

        base_name = meta_path.stem.replace("_token_meta", "")
        out_mmap = gene_out_dir / f"{base_name}_{mean_method}.mmap"
        out_meta = gene_out_dir / f"{base_name}_{mean_method}_meta.npz"

        if out_mmap.exists() and out_meta.exists():
            continue

        if mean_method == "mean_dim":
            out_shape = (mm_in.shape[0], mm_in.shape[1], 1)
            mm_out = np.memmap(out_mmap, dtype="float16", mode="w+", shape=out_shape)
            mm_out[:] = mm_in.astype("float32").mean(axis=2, keepdims=True).astype("float16")
        else:
            out_shape = (mm_in.shape[0], 1, mm_in.shape[2])
            mm_out = np.memmap(out_mmap, dtype="float16", mode="w+", shape=out_shape)
            mm_out[:] = mm_in.astype("float32").mean(axis=1, keepdims=True).astype("float16")

        mm_out.flush()
        np.savez_compressed(
            out_meta,
            identifier=meta_in["identifier"],
            shape=np.asarray(out_shape, dtype=np.int64),
            mmap_path=out_mmap.name,
        )

        del mm_out
        del mm_in
        gc.collect()


def validate_phenotype_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Phenotype stack not found at {path}. "
            "Expected the Evo2 embedding run to produce zs_full_stacked_phenotypes.npz."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Evo2 embedding memmaps for downstream training")
    parser.add_argument("--genes", type=str, default="all", help="Comma-separated gene list or 'all'")
    parser.add_argument(
        "--embed_types",
        nargs="+",
        default=["token", "mean_dim", "mean_seq"],
        choices=["token", "mean_dim", "mean_seq"],
        help="Embedding representations to materialize for downstream training",
    )
    parser.add_argument("--prefix", type=str, default="full", help="Identifier prefix used for downstream labels")
    parser.add_argument("--raw_embed_root", type=str, default=str(RAW_TOKEN_EMBED_ROOT))
    parser.add_argument("--token_memmap_root", type=str, default=str(memmap_root("token")))
    parser.add_argument("--mean_dim_memmap_root", type=str, default=str(memmap_root("mean_dim")))
    parser.add_argument("--mean_seq_memmap_root", type=str, default=str(memmap_root("mean_seq")))
    parser.add_argument("--phenotype_label_path", type=str, default=str(DEFAULT_PHENOTYPE_LABEL_PATH))
    parser.add_argument("--num_shards", type=int, default=1, help="Split each gene's batches across N jobs")
    parser.add_argument("--shard_index", type=int, default=0, help="Zero-based shard index to process")
    return parser


def main(args: argparse.Namespace) -> None:
    genes = parse_gene_list(args.genes)
    raw_embed_root = Path(args.raw_embed_root)
    token_root = Path(args.token_memmap_root)
    mean_dim_root = Path(args.mean_dim_memmap_root)
    mean_seq_root = Path(args.mean_seq_memmap_root)

    validate_phenotype_file(Path(args.phenotype_label_path))

    if "token" in args.embed_types or "mean_dim" in args.embed_types or "mean_seq" in args.embed_types:
        for gene in genes:
            convert_token_batches_to_memmaps(
                raw_embed_root,
                token_root,
                gene,
                prefix=args.prefix,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
            )

    if "mean_dim" in args.embed_types:
        for gene in genes:
            project_gene_to_mean(token_root, mean_dim_root, gene, mean_method="mean_dim")

    if "mean_seq" in args.embed_types:
        for gene in genes:
            project_gene_to_mean(token_root, mean_seq_root, gene, mean_method="mean_seq")


if __name__ == "__main__":
    main(build_parser().parse_args())
