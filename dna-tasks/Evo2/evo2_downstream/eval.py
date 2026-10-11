"""Evaluate downstream per-drug Evo2 classifiers trained via random split."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evo2_downstream.config import (
    DEFAULT_PHENOTYPE_LABEL_PATH,
    EVO2_DRUG_INDEX,
    classification_output_root,
    ensure_dnabert_transfer_learn_on_path,
    memmap_root,
    saved_model_root,
    threshold_root,
)

ensure_dnabert_transfer_learn_on_path()

from finetuning.modules.dataloader.locus_order import DRUG_TO_LOCI
from finetuning.modules.resistance_classification_train import (
    MeanMemmapMap,
    MeanMultiGeneConcatDataset,
    MultiGeneConcatDataset,
    PcaMemmapMap,
    PcaMultiGeneConcatDataset,
    TokenMemmapMap,
    stratified_split_dataset,
)
from finetuning.modules.utils.classification_metric_utils import ThresholdValue
from finetuning.modules.utils.token_train_utils import (
    calculate_single_drug_threshold,
    calculate_test_metrics_single_drug,
    evaluate,
    get_model_class,
)


def build_label_map(label_file: str, drug: str, prefix: str = "full"):
    drug_index = EVO2_DRUG_INDEX[drug]
    label_np_file = np.load(label_file)
    labels = label_np_file["phenotypes"]
    if labels.ndim != 2:
        raise ValueError(f"Expected phenotype matrix to be 2D, got shape={labels.shape}")
    if labels.shape[1] != len(EVO2_DRUG_INDEX):
        raise ValueError(
            "Phenotype column count does not match EVO2_DRUG_INDEX length: "
            f"{labels.shape[1]} vs {len(EVO2_DRUG_INDEX)}"
        )

    drug_labels = labels[:, drug_index]
    print(f"Loading labels from: {label_file}")
    print(f"Building label map for drug: {drug} (Evo2 index {drug_index})")
    print(f"Total samples for drug {drug}: {len(drug_labels)} (including missing labels)")
    valid_indices = np.where(drug_labels != -1)[0]
    drug_labels = drug_labels[valid_indices]
    print(f"Total samples for drug {drug}: {len(drug_labels)} (valid labels)")
    label_map = {
        f"{prefix}_{sample_index:06d}": float(drug_labels[label_index])
        for label_index, sample_index in enumerate(valid_indices)
    }
    return label_map, drug_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate downstream resistance classifiers on Evo2 embeddings"
    )
    parser.add_argument("--model_name", type=str, default="DNABERTCNN")
    parser.add_argument("--saved_model_name", type=str, default="DNABERTCNN")
    parser.add_argument(
        "--embed_type",
        type=str,
        default="token",
        choices=["token", "mean_dim", "mean_seq", "pca"],
    )
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--saved_embed_memmap_dir", type=str, default=None)
    parser.add_argument(
        "--phenotype_label_path",
        type=str,
        default=str(DEFAULT_PHENOTYPE_LABEL_PATH),
    )
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--threshold_dir", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument(
        "--pca_components",
        type=int,
        default=10,
        help="Component count in precomputed PCA embedding files",
    )
    parser.add_argument("--max_length", type=int, default=5000)
    return parser


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.saved_embed_memmap_dir is None:
        args.saved_embed_memmap_dir = str(memmap_root(args.embed_type))
    if args.output_path is None:
        args.output_path = str(classification_output_root(args.embed_type))
    if args.saved_model_path is None:
        args.saved_model_path = str(saved_model_root(args.embed_type))
    if args.threshold_dir is None:
        args.threshold_dir = str(threshold_root(args.embed_type))

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(args.saved_model_path).mkdir(parents=True, exist_ok=True)
    Path(args.threshold_dir).mkdir(parents=True, exist_ok=True)
    return args


def _load_dataset(args) -> Tuple[Subset, Subset, int, int]:
    full_label_map, _ = build_label_map(
        args.phenotype_label_path, args.drug, prefix="full"
    )

    loci = DRUG_TO_LOCI[args.drug]
    if len(loci) == 1:
        if args.embed_type == "pca":
            meta_pattern = (
                f"{args.saved_embed_memmap_dir}/{loci[0]}"
                f"/*_pc{args.pca_components}_meta.npz"
            )
        else:
            meta_pattern = (
                f"{args.saved_embed_memmap_dir}/{loci[0]}"
                f"/*_{args.embed_type}_meta.npz"
            )
        meta_paths = sorted(glob.glob(meta_pattern))
        if not meta_paths:
            raise FileNotFoundError(f"No embedding metadata matched {meta_pattern}")
        if args.embed_type == "token":
            full_dataset = TokenMemmapMap(meta_paths, full_label_map)
        elif args.embed_type == "pca":
            full_dataset = PcaMemmapMap(
                meta_paths, full_label_map, k=args.pca_components
            )
        else:
            full_dataset = MeanMemmapMap(
                meta_paths, full_label_map, embed_type=args.embed_type
            )
    else:
        gene_dirs = [f"{args.saved_embed_memmap_dir}/{gene}/" for gene in loci]
        if args.embed_type == "token":
            full_dataset = MultiGeneConcatDataset(gene_dirs, full_label_map)
        elif args.embed_type == "pca":
            full_dataset = PcaMultiGeneConcatDataset(
                gene_dirs, full_label_map, k=args.pca_components
            )
        else:
            full_dataset = MeanMultiGeneConcatDataset(
                gene_dirs, full_label_map, embed_type=args.embed_type
            )

    embeds, _ = full_dataset[0]
    model_dim, model_seq_len = embeds.shape

    train_idx, test_idx, _, _ = stratified_split_dataset(
        full_dataset=full_dataset,
        label_dict=full_label_map,
        test_size=args.test_split,
        seed=args.random_seed,
    )
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    return train_dataset, test_dataset, model_dim, model_seq_len


def _load_or_compute_threshold(model, train_loader, args, device):
    threshold_path = Path(args.threshold_dir) / args.drug / f"seed_{args.random_seed}"
    if args.fold is not None:
        threshold_path = threshold_path / f"fold_{args.fold}"
    threshold_path = threshold_path / "threshold.txt"

    if threshold_path.exists():
        for line in threshold_path.read_text().splitlines():
            if line.startswith("Threshold:"):
                threshold = float(line.split(":", 1)[1].strip())
                print(f"Loaded threshold {threshold:.4f} from {threshold_path}")
                return threshold
        raise ValueError(f"No Threshold entry found in {threshold_path}")

    y_train, y_train_pred = evaluate(model, train_loader, device)
    threshold = calculate_single_drug_threshold(
        y_train.ravel(),
        y_train_pred.ravel(),
        get_threshold_val=ThresholdValue(),
    )
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(
        f"Drug: {args.drug}\n"
        f"Threshold: {threshold}\n"
        f"Embed type: {args.embed_type}\n"
        f"Random seed: {args.random_seed}\n"
        "Prediction scale: probability\n"
    )
    print(f"Threshold saved to: {threshold_path}")
    return threshold


def main(args: argparse.Namespace) -> None:
    apply_defaults(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{torch.cuda.device_count()} GPUs available to use!")
    print(f"[eval] drug={args.drug} fold={args.fold} seed={args.random_seed}")

    train_dataset, test_dataset, model_dim, model_seq_len = _load_dataset(args)
    print(
        f"[eval] train={len(train_dataset)} test={len(test_dataset)} "
        f"D={model_dim} L={model_seq_len}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.val_batch_size, shuffle=False
    )

    model_path = Path(args.saved_model_path) / args.drug / f"seed_{args.random_seed}"
    if args.fold is not None:
        model_path = model_path / f"fold_{args.fold}"
    model_path = model_path / f"{args.saved_model_name}.pt"

    print(f"[eval] Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model = get_model_class(
        model_name=args.model_name,
        in_dim=model_dim,
        seq_len=model_seq_len,
        device=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    threshold = _load_or_compute_threshold(model, train_loader, args, device)

    print("\nEvaluating on test data...")
    y_test, y_test_pred = evaluate(model, test_loader, device)
    test_results = calculate_test_metrics_single_drug(
        y_test.ravel(),
        y_test_pred.ravel(),
        threshold,
        drug_name=args.drug,
        model_type=f"Evo2-{args.saved_model_name}",
    )

    output_seed_path = Path(args.output_path) / args.drug / f"seed_{args.random_seed}"
    if args.fold is not None:
        output_seed_path = output_seed_path / f"fold_{args.fold}"
    output_seed_path.mkdir(parents=True, exist_ok=True)
    test_results_file = output_seed_path / f"test_set_auc_{args.drug}.csv"
    test_results.to_csv(test_results_file, index=False)
    print(f"\nTest results saved to: {test_results_file}")
    print(test_results)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main(build_parser().parse_args())
