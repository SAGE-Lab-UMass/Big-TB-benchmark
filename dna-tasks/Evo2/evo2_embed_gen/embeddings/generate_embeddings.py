"""Generate Evo2 embeddings for the same MTB data used by DNABERT2."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re

import numpy as np
import torch
import tqdm

from evo2_embed_gen.data.dataset import DataConfig, make_loader, parse_gene_list
from evo2_embed_gen.embeddings.io import save_batch, stack_final_phenotypes, write_metadata
from evo2_embed_gen.model.evo2_model import Evo2Embedder, Evo2ModelConfig


EVO2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("EVO2_DATA_DIR", EVO2_DIR / "data" / "multidrug_classification" / "training"))
DEFAULT_PHENOTYPE_FILE = Path(
    os.environ.get("EVO2_PHENOTYPE_FILE", DEFAULT_DATA_DIR / "phenotype" / "master_resistance_table.csv")
)
DEFAULT_GENOTYPE_DIR = Path(
    os.environ.get("EVO2_GENOTYPE_INPUT_DIRECTORY", EVO2_DIR / "data" / "aligned")
)
DEFAULT_EMBED_DIR = Path(os.environ.get("EVO2_EMBED_ROOT", EVO2_DIR / "embeddings"))


def main(args: argparse.Namespace) -> None:
    genes = parse_gene_list(args.genes)
    n_gpu = torch.cuda.device_count()
    print(f"{n_gpu} GPUs available")
    torch.cuda.empty_cache()

    data_config = DataConfig(
        datapath=args.datapath,
        full_dataname=args.full_dataname,
        train_dataname=args.train_dataname,
        val_dataname=args.val_dataname,
        phenotype_file=args.phenotype_file,
        genotype_input_directory=args.genotype_input_directory,
        test_split=args.test_split,
        full_batch_size=args.full_batch_size,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_isolates=args.max_isolates,
    )
    model_config = Evo2ModelConfig(
        model_name=args.model_name,
        local_path=args.local_path,
        layer_name=args.layer_name,
        max_length=args.max_length,
        pad_char=args.pad_char,
        use_kernels=args.use_kernels,
        save_dtype=args.save_dtype,
    )

    print("Loading Evo2 model")
    embedder = Evo2Embedder(model_config)
    print("Evo2 model loaded")

    if args.is_single_gene_algo:
        full_loader = make_loader(data_config, args.is_single_gene_algo, load_train=True, n_gpu=n_gpu)
        generate_embeddings_for_loader(
            full_loader,
            embedder,
            args.embed_dir,
            genes,
            args.drug,
            args.embed_type,
            is_single_gene=True,
            data_partition="full",
            args=args,
        )
        if args.stack_phenotypes:
            stack_final_phenotypes(args.embed_dir, data_partition="full", gene=genes[0])
        return

    train_loader = make_loader(data_config, args.is_single_gene_algo, load_train=True, n_gpu=n_gpu)
    generate_embeddings_for_loader(
        train_loader,
        embedder,
        args.embed_dir,
        genes,
        args.drug,
        args.embed_type,
        is_single_gene=False,
        data_partition="train",
        args=args,
    )
    del train_loader
    torch.cuda.empty_cache()

    val_loader = make_loader(data_config, args.is_single_gene_algo, load_train=False, n_gpu=n_gpu)
    generate_embeddings_for_loader(
        val_loader,
        embedder,
        args.embed_dir,
        genes,
        args.drug,
        args.embed_type,
        is_single_gene=False,
        data_partition="val",
        args=args,
    )
    torch.cuda.empty_cache()


def generate_embeddings_for_loader(
    data_loader,
    embedder: Evo2Embedder,
    embed_dir: str,
    genes: list[str],
    drug: str,
    embed_type: str,
    is_single_gene: bool,
    data_partition: str,
    args: argparse.Namespace,
) -> None:
    Path(embed_dir).mkdir(parents=True, exist_ok=True)

    gene_order = list(data_loader.dataset.gene_names)
    selected_gene_indices = get_selected_gene_indices(gene_order, genes, is_single_gene)
    selected_gene_names = [gene_order[i].replace(".fasta", "") for i in selected_gene_indices]

    save_path = os.path.join(embed_dir, selected_gene_names[0]) if is_single_gene else embed_dir
    Path(save_path).mkdir(parents=True, exist_ok=True)
    write_metadata(
        save_path,
        {
            "model_name": args.model_name,
            "local_path": args.local_path,
            "layer_name": args.layer_name,
            "max_length": args.max_length,
            "embed_type": embed_type,
            "save_dtype": args.save_dtype,
            "data_partition": data_partition,
            "is_single_gene_algo": is_single_gene,
            "requested_genes": genes,
            "selected_gene_names": selected_gene_names,
            "all_gene_order": gene_order,
            "drug": drug,
            "drug_order": list(data_loader.dataset.drug_names),
            "phenotype_file": args.phenotype_file,
            "genotype_input_directory": args.genotype_input_directory,
        },
    )

    resume_from = 0
    if args.resume:
        resume_from = find_resume_batch(save_path, data_partition)
        print(f"Resuming {selected_gene_names[0]} at batch {resume_from}")

    for batch_index, batch in enumerate(tqdm.tqdm(data_loader)):
        if batch_index < resume_from:
            if batch_index >= max(0, resume_from - args.resume_validation_batches):
                validate_saved_batch(
                    save_path,
                    data_partition,
                    batch_index,
                    batch,
                    args.max_length,
                    args.save_dtype,
                )
                print(f"Validated existing batch {batch_index}")
            continue

        batch_embeddings = []
        phenotypes = collect_phenotypes(batch)

        for gene_index in selected_gene_indices:
            sequence_key = f"gene_seq_{gene_index + 1}"
            gene_embeddings = embedder.embed_sequences(batch[sequence_key], embed_type)
            batch_embeddings.append(gene_embeddings)
            torch.cuda.empty_cache()

        if embed_type == "token":
            embeddings = np.stack(batch_embeddings, axis=1)
        else:
            embeddings = np.stack(batch_embeddings, axis=1)

        print(f"Batch {batch_index}: embeddings shape {embeddings.shape}; phenotypes shape {phenotypes.shape}")
        save_batch(
            save_path,
            data_partition=data_partition,
            batch_index=batch_index,
            embeddings=embeddings,
            phenotypes=phenotypes,
            isolate_ids=batch["isolate_id"],
        )

    print(f"All {data_partition} batches saved in {save_path}")


def find_resume_batch(save_path: str, data_partition: str) -> int:
    batch_kinds = ("embeddings", "res_phenotypes", "isolate_ids")
    index_sets = []
    for kind in batch_kinds:
        pattern = os.path.join(save_path, f"zs_{data_partition}_{kind}_batch_*.npy")
        indices = set()
        for path in glob.glob(pattern):
            match = re.search(r"_batch_(\d+)\.npy$", path)
            if match:
                indices.add(int(match.group(1)))
        index_sets.append(indices)

    complete_indices = set.intersection(*index_sets)
    resume_from = 0
    while resume_from in complete_indices:
        resume_from += 1

    completed_after_gap = sorted(index for index in complete_indices if index > resume_from)
    if completed_after_gap:
        raise RuntimeError(
            f"Non-contiguous completed batches in {save_path}; first gap is {resume_from}, "
            f"but later completed batches exist: {completed_after_gap[:5]}"
        )
    return resume_from


def validate_saved_batch(
    save_path: str,
    data_partition: str,
    batch_index: int,
    batch: dict,
    max_length: int,
    save_dtype: str,
) -> None:
    embedding_path = os.path.join(
        save_path, f"zs_{data_partition}_embeddings_batch_{batch_index}.npy"
    )
    phenotype_path = os.path.join(
        save_path, f"zs_{data_partition}_res_phenotypes_batch_{batch_index}.npy"
    )
    isolate_path = os.path.join(
        save_path, f"zs_{data_partition}_isolate_ids_batch_{batch_index}.npy"
    )

    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    phenotypes = np.load(phenotype_path, allow_pickle=False)
    isolate_ids = np.load(isolate_path, allow_pickle=True).astype(str).tolist()
    expected_phenotypes = collect_phenotypes(batch)
    expected_isolate_ids = [str(identifier) for identifier in batch["isolate_id"]]
    expected_prefix = (len(expected_isolate_ids), 1, max_length)

    if embeddings.shape[:3] != expected_prefix or embeddings.ndim != 4:
        raise ValueError(
            f"Invalid embedding shape for batch {batch_index}: {embeddings.shape}; "
            f"expected ({expected_prefix[0]}, 1, {max_length}, hidden_size)"
        )
    if embeddings.dtype != np.dtype(save_dtype):
        raise ValueError(
            f"Invalid embedding dtype for batch {batch_index}: {embeddings.dtype}; "
            f"expected {save_dtype}"
        )
    if not np.isfinite(embeddings).all() or not np.any(embeddings):
        raise ValueError(f"Embedding payload failed finite/nonzero validation for batch {batch_index}")
    if not np.array_equal(phenotypes, expected_phenotypes):
        raise ValueError(f"Phenotypes do not match source data for batch {batch_index}")
    if isolate_ids != expected_isolate_ids:
        raise ValueError(f"Isolate IDs do not match source data for batch {batch_index}")


def get_selected_gene_indices(gene_order: list[str], genes: list[str], is_single_gene: bool) -> list[int]:
    if not is_single_gene and not genes:
        return list(range(len(gene_order)))

    selected_indices = []
    for gene in genes:
        matches = [
            index
            for index, gene_name in enumerate(gene_order)
            if gene_name.replace(".fasta", "") == gene or gene_name.startswith(gene)
        ]
        if not matches:
            raise ValueError(f"Gene {gene!r} not found in gene order: {gene_order}")
        selected_indices.append(matches[0])

    if is_single_gene and len(selected_indices) != 1:
        raise ValueError("--is_single_gene_algo expects exactly one gene in --genes")
    return selected_indices


def collect_phenotypes(batch: dict) -> np.ndarray:
    drug_keys = sorted(
        (key for key in batch if key.startswith("res_phenotype_drug_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    phenotypes = torch.stack([batch[key] for key in drug_keys], dim=1)
    return phenotypes.cpu().numpy().astype(np.int16, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Evo2 embeddings for MTB genotype data")
    parser.add_argument("--model_name", type=str, default="evo2_7b", help="Evo2 checkpoint name")
    parser.add_argument("--local_path", type=str, default=None, help="Optional local Evo2 .pt checkpoint path")
    parser.add_argument("--layer_name", type=str, default="blocks.20.mlp.l3", help="Evo2 layer to extract")
    parser.add_argument("--use_kernels", action="store_true", help="Enable optional Vortex Triton kernels")
    parser.add_argument("--max_length", type=int, default=5000, help="DNABERT-compatible max token length")
    parser.add_argument("--pad_char", type=str, default="N", help="Right-padding character masked out of pooling")
    parser.add_argument(
        "--embed_type",
        type=str,
        default="token",
        choices=["mean_dim", "mean_seq", "token"],
        help="Embedding reduction to save",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Saved embedding dtype",
    )

    parser.add_argument("--full_batch_size", type=int, default=2, help="Batch size for single-gene full data")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for train data")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Batch size for validation data")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--max_isolates", type=int, default=None, help="Optional cap for a quick test run")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train/validation split ratio")

    parser.add_argument(
        "--datapath",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory for generated genotype/phenotype CSV files",
    )
    parser.add_argument("--full_dataname", type=str, default="geno_pheno_full_combined.csv")
    parser.add_argument("--train_dataname", type=str, default="geno_pheno_train_combined.csv")
    parser.add_argument("--val_dataname", type=str, default="geno_pheno_val_combined.csv")
    parser.add_argument(
        "--phenotype_file",
        type=str,
        default=str(DEFAULT_PHENOTYPE_FILE),
        help="Phenotype table used by DNABERT2",
    )
    parser.add_argument(
        "--genotype_input_directory",
        type=str,
        default=str(DEFAULT_GENOTYPE_DIR),
        help="Aligned genotype FASTA directory used by DNABERT2",
    )
    parser.add_argument(
        "--embed_dir",
        type=str,
        default=str(DEFAULT_EMBED_DIR),
        help="Root directory for Evo2 embedding outputs",
    )
    parser.add_argument("--drug", type=str, default="RIFAMPICIN", help="Metadata label matching DNABERT CLI")
    parser.add_argument("--genes", type=str, default="rpoB", help="Comma-separated gene names")
    parser.add_argument("--is_single_gene_algo", action="store_true", help="Use full dataset for one gene")
    parser.add_argument("--stack_phenotypes", action="store_true", help="Stack phenotype batches after full run")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue after the contiguous set of complete existing batches",
    )
    parser.add_argument(
        "--resume_validation_batches",
        type=int,
        default=3,
        help="Number of trailing existing batches to validate before resuming",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
