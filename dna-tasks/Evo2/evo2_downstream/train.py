"""Train downstream per-drug models on Evo2 embeddings with DNABERT-matched logic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evo2_downstream.config import (
    DEFAULT_PHENOTYPE_LABEL_PATH,
    EVO2_DRUG_INDEX,
    classification_output_root,
    ensure_dnabert_transfer_learn_on_path,
    memmap_root,
    saved_model_root,
)


ensure_dnabert_transfer_learn_on_path()
import resistance_classification_train as dnabert_train  # noqa: E402


def enforce_evo2_drug_index_mapping(phenotype_label_path: str) -> None:
    """Patch DNABERT trainer's DRUG_INDEX to match the Evo2 NPZ column ordering."""
    if not hasattr(dnabert_train, "DRUG_INDEX"):
        raise AttributeError("DNABERT trainer missing DRUG_INDEX symbol required for Evo2 mapping enforcement")

    dnabert_train.DRUG_INDEX = dict(EVO2_DRUG_INDEX)

    phenotypes = np.load(phenotype_label_path)["phenotypes"]
    if phenotypes.ndim != 2:
        raise ValueError(f"Expected phenotype matrix to be 2D, got shape={phenotypes.shape}")
    if phenotypes.shape[1] != len(EVO2_DRUG_INDEX):
        raise ValueError(
            "Phenotype column count does not match EVO2_DRUG_INDEX length: "
            f"{phenotypes.shape[1]} vs {len(EVO2_DRUG_INDEX)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train downstream resistance classifiers on Evo2 embeddings")
    parser.add_argument("--model_name", type=str, default="DNABERTCNN")
    parser.add_argument("--embed_type", type=str, default="token", choices=["token", "mean_dim", "mean_seq", "pca"])
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--saved_embed_memmap_dir", type=str, default=None)
    parser.add_argument("--phenotype_label_path", type=str, default=str(DEFAULT_PHENOTYPE_LABEL_PATH))
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--freeze_bias_frac", type=float, default=0.25)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--data_loader_workers", type=int, default=0)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--pca_components", type=int, default=10)
    parser.add_argument("--use_pca", action="store_true")
    
    # Early stopping parameters (matching SD-CNN defaults)
    parser.add_argument("--early_stopping_min_epochs", type=int, default=5,
                        help="Minimum epochs before early stopping can trigger")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Number of epochs to wait without improvement before stopping")
    parser.add_argument("--early_stopping_min_relative_improvement", type=float, default=1e-3,
                        help="Minimum relative improvement threshold (default: 0.001 = 0.1%%)")
    parser.add_argument("--early_stopping_smoothing_window", type=int, default=3,
                        help="Window size for smoothing training loss")
    parser.add_argument("--use_validation_early_stopping", action="store_true",
                        help="Use validation-AUC early stopping instead of training-loss (default: False, uses training-loss)")
    
    return parser


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.saved_embed_memmap_dir is None:
        args.saved_embed_memmap_dir = str(memmap_root(args.embed_type))
    if args.output_path is None:
        args.output_path = str(classification_output_root(args.embed_type))
    if args.saved_model_path is None:
        args.saved_model_path = str(saved_model_root(args.embed_type))

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(args.saved_model_path).mkdir(parents=True, exist_ok=True)
    return args


def main(args: argparse.Namespace) -> None:
    args = apply_defaults(args)
    enforce_evo2_drug_index_mapping(args.phenotype_label_path)
    dnabert_train.main(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
