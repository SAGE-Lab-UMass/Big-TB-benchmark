"""Evaluate Evo2 downstream classifiers on a held-out MTB lineage."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
EVO2_DIR = THIS_DIR.parents[1]
sys.path.insert(0, str(EVO2_DIR))
sys.path.insert(0, str(THIS_DIR))

import evo2_downstream.train as evo2_train  # noqa: E402
from evo2_downstream.config import EVO2_DRUG_INDEX  # noqa: E402
from lineage_split import (  # noqa: E402
    MAJOR_LINEAGES,
    load_isolate_id_map,
    load_lineage_map,
    make_lineage_aware_split_fn,
)

from evo2_downstream.config import ensure_dnabert_transfer_learn_on_path  # noqa: E402

ensure_dnabert_transfer_learn_on_path()
import resistance_classification_train as evo2_data  # noqa: E402
from utils.classification_metric_utils import ThresholdValue  # noqa: E402
from utils.token_train_utils import (  # noqa: E402
    calculate_single_drug_threshold,
    calculate_test_metrics_single_drug,
    evaluate,
    get_model_class,
)

_DATA_DIR = EVO2_DIR / "data" / "multidrug_classification" / "training"
_GENO_PHENO_CSV = _DATA_DIR / "geno_pheno_full_combined.csv"
_LINEAGE_CSV = EVO2_DIR.parents[1] / "BIG_TB_isolates_with_lineages.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = evo2_train.build_parser()
    parser.description = "Evaluate Evo2 downstream classifiers on a leave-one-lineage-out test split"
    parser.add_argument("--saved_model_name", type=str, default="DNABERTCNN")
    parser.add_argument("--threshold_dir", type=str, default=None)
    parser.add_argument(
        "--heldout-lineage",
        dest="heldout_lineage",
        required=True,
        choices=list(MAJOR_LINEAGES),
        help="Major MTB lineage (1-4) to evaluate as the held-out test set",
    )
    parser.add_argument(
        "--geno-pheno-csv",
        dest="geno_pheno_csv",
        default=str(_GENO_PHENO_CSV),
        help="Path to geno_pheno_full_combined.csv (maps full_N to isolate ID)",
    )
    parser.add_argument(
        "--lineage-csv",
        dest="lineage_csv",
        default=str(_LINEAGE_CSV),
        help="Path to BIG_TB_isolates_with_lineages.csv",
    )
    parser.add_argument(
        "--min-class-count",
        dest="min_class_count",
        type=int,
        default=50,
        help="Minimum samples required in each class (train_S, train_R, test_S, test_R)",
    )
    return parser


def _load_dataset(args):
    full_label_map, drug_index = evo2_data.build_label_map(
        args.phenotype_label_path, args.drug, prefix="full"
    )
    if drug_index != EVO2_DRUG_INDEX[args.drug]:
        raise ValueError(
            f"Drug-index mismatch for {args.drug}: loaded {drug_index}, "
            f"expected Evo2 index {EVO2_DRUG_INDEX[args.drug]}"
        )

    loci = evo2_data.DRUG_TO_LOCI[args.drug]
    if len(loci) == 1:
        meta_paths = sorted(
            glob.glob(
                f"{args.saved_embed_memmap_dir}/{loci[0]}/*_{args.embed_type}_meta.npz"
            )
        )
        if not meta_paths:
            raise FileNotFoundError(
                f"No {args.embed_type} metadata found for {loci[0]} under "
                f"{args.saved_embed_memmap_dir}"
            )
        if args.embed_type == "token":
            full_dataset = evo2_data.TokenMemmapMap(meta_paths, full_label_map)
        elif args.embed_type == "pca":
            full_dataset = evo2_data.PcaMemmapMap(
                meta_paths, full_label_map, k=args.pca_components
            )
        else:
            full_dataset = evo2_data.MeanMemmapMap(
                meta_paths, full_label_map, embed_type=args.embed_type
            )
    else:
        gene_dirs = [f"{args.saved_embed_memmap_dir}/{gene}/" for gene in loci]
        if args.embed_type == "token":
            full_dataset = evo2_data.MultiGeneConcatDataset(gene_dirs, full_label_map)
        elif args.embed_type == "pca":
            full_dataset = evo2_data.PcaMultiGeneConcatDataset(
                gene_dirs, full_label_map, k=args.pca_components
            )
        else:
            full_dataset = evo2_data.MeanMultiGeneConcatDataset(
                gene_dirs, full_label_map, embed_type=args.embed_type
            )

    embeds, _ = full_dataset[0]
    model_dim, model_seq_len = embeds.shape
    print(f"[lineage_holdout_eval] loci={loci}")
    print(
        f"[lineage_holdout_eval] Evo2 input shape: "
        f"(D={model_dim}, L={model_seq_len})"
    )
    return full_dataset, model_dim, model_seq_len, full_label_map


def _predict_probabilities(model, dataloader, model_name, device):
    labels, predictions = evaluate(model, dataloader, device)
    if model_name in {"DNABERTCNN", "DNABERTMLP"}:
        predictions = torch.sigmoid(torch.from_numpy(predictions)).numpy()
    return labels, predictions


def _load_or_compute_threshold(model, dataloader, args, device):
    threshold_path = (
        Path(args.threshold_dir)
        / args.drug
        / f"seed_{args.random_seed}"
        / "threshold.txt"
    )
    if threshold_path.exists():
        for line in threshold_path.read_text().splitlines():
            if line.startswith("Threshold:"):
                threshold = float(line.split(":", 1)[1].strip())
                print(f"Loaded threshold {threshold:.4f} from {threshold_path}")
                return threshold
        raise ValueError(f"No Threshold entry found in {threshold_path}")

    y_train, y_train_pred = _predict_probabilities(
        model, dataloader, args.model_name, device
    )
    threshold = calculate_single_drug_threshold(
        y_train.ravel(), y_train_pred.ravel(), get_threshold_val=ThresholdValue()
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
    args = evo2_train.apply_defaults(args)
    evo2_train.enforce_evo2_drug_index_mapping(args.phenotype_label_path)

    heldout = str(args.heldout_lineage)
    lineage_tag = f"heldout_lineage_{heldout}"
    args.output_path = str(Path(args.output_path) / lineage_tag)
    args.saved_model_path = str(Path(args.saved_model_path) / lineage_tag)
    if args.threshold_dir is None:
        args.threshold_dir = str(
            EVO2_DIR
            / "training_output"
            / "zero_shot"
            / "lineage_aware_holdout"
            / args.drug
            / "saved_parameters"
            / "evo2"
            / args.embed_type
        )
    args.threshold_dir = str(Path(args.threshold_dir) / lineage_tag)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(args.threshold_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{torch.cuda.device_count()} GPUs available to use!")
    print(f"[lineage_holdout_eval] drug={args.drug} heldout_lineage={heldout}")
    print(f"[lineage_holdout_eval] output_path={args.output_path}")
    print(f"[lineage_holdout_eval] saved_model_path={args.saved_model_path}")

    full_dataset, model_dim, model_seq_len, full_label_map = _load_dataset(args)

    isolate_id_map = load_isolate_id_map(args.geno_pheno_csv)
    lineage_map = load_lineage_map(args.lineage_csv)
    lineage_split_fn = make_lineage_aware_split_fn(
        heldout,
        isolate_id_map,
        lineage_map,
        min_class_count=args.min_class_count,
    )
    train_idx, test_idx, train_labels, test_labels = lineage_split_fn(full_dataset, full_label_map)

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print(f"\nTraining samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Training set: {np.sum(train_labels == 0)} R, {np.sum(train_labels == 1)} S")
    print(f"Test set: {np.sum(test_labels == 0)} R, {np.sum(test_labels == 1)} S")

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False)

    seed_path = Path(args.saved_model_path) / args.drug / f"seed_{args.random_seed}"
    model_path = seed_path / f"{args.saved_model_name}.pt"
    print(f"Loading model from {model_path}...\n")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    checkpoint_dim = state_dict["inp_project.weight"].shape[1]
    if checkpoint_dim != model_dim:
        raise ValueError(
            f"Evo2/checkpoint input-dimension mismatch: dataset D={model_dim}, "
            f"checkpoint D={checkpoint_dim}"
        )

    model = get_model_class(
        model_name=args.model_name,
        in_dim=model_dim,
        seq_len=model_seq_len,
        device=device,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"[lineage_holdout_eval] Checkpoint matches Evo2 input "
        f"(D={model_dim}, L={model_seq_len})"
    )

    threshold = _load_or_compute_threshold(model, train_dataloader, args, device)

    print("\nEvaluating on held-out lineage test data...")
    y_test, y_test_pred = _predict_probabilities(
        model, test_dataloader, args.model_name, device
    )
    test_results = calculate_test_metrics_single_drug(
        y_test.ravel(),
        y_test_pred.ravel(),
        threshold,
        drug_name=args.drug,
        model_type=f"Evo2-{args.saved_model_name}",
    )

    output_seed_path = Path(args.output_path) / args.drug / f"seed_{args.random_seed}"
    output_seed_path.mkdir(parents=True, exist_ok=True)
    test_results_file = output_seed_path / f"test_set_auc_{args.drug}.csv"
    test_results.to_csv(test_results_file, index=False)

    print(f"\nTest results saved to: {test_results_file}")
    print(test_results)
    print("\nLineage holdout evaluation complete!")


if __name__ == "__main__":
    main(build_parser().parse_args())
