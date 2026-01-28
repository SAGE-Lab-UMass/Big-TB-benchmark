"""
Evaluation script for downstream drug resistance classification model.

This module evaluates a trained DNABERT-CNN model on drug resistance classification
tasks using stratified train/test splits and threshold-based decision making.
"""

import argparse
import os
import glob

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from downstream_cnn_model import *
from dataloader.dataloader import (
    TokenMemmapMap,
    MeanMemmapMap,
    PcaMemmapMap,
    MultiGeneConcatDataset,
    MeanMultiGeneConcatDataset,
    PcaMultiGeneConcatDataset
)
from dataloader.locus_order import DRUG_TO_LOCI
from utils.embed_gen_utils import *
from utils.classification_metric_utils import *
from utils.token_train_utils import (
    evaluate,
    calculate_single_drug_threshold,
    calculate_test_metrics_single_drug,
    get_model_class
)


# Mapping of drug names to their index in the phenotype matrix (fallback only)
DRUG_INDEX = {
    'ISONIAZID': 0,
    'RIFAMPICIN': 1,
    'ETHAMBUTOL': 2,
    'PYRAZINAMIDE': 3,
    'STREPTOMYCIN': 4,
    'KANAMYCIN': 5,
    'AMIKACIN': 6,
    'CAPREOMYCIN': 7,
    'LEVOFLOXACIN': 8,
    'MOXIFLOXACIN': 9,
    'ETHIONAMIDE': 10
}


def _infer_drug_index(label_file, drug):
    """
    Automatically infer the drug index from the phenotype file.

    Attempts to find the drug index by checking:
    1. If the NPZ file contains drug names metadata
    2. If the drug exists in DRUG_INDEX (fallback to hardcoded mapping)

    Args:
        label_file (str): Path to NPZ file containing phenotype data.
        drug (str): Drug name to find the index for.

    Returns:
        int: Column index of the drug in the phenotype matrix.

    Raises:
        KeyError: If drug index cannot be inferred from available data.
    """
    label_data = np.load(label_file, allow_pickle=True)

    # Try to get drug names from NPZ metadata
    if "drug_names" in label_data:
        drug_names = label_data["drug_names"]
        if drug in drug_names:
            return int(np.where(drug_names == drug)[0][0])

    # Fall back to hardcoded DRUG_INDEX mapping
    if drug in DRUG_INDEX:
        return DRUG_INDEX[drug]

    # If all else fails, raise an error
    raise KeyError(
        f"Could not infer drug index for '{drug}'. "
        f"Available methods: NPZ metadata, hardcoded DRUG_INDEX. "
        f"Consider adding '{drug}' to DRUG_INDEX or NPZ metadata."
    )


def stratified_split_dataset(full_dataset, label_dict, test_size=0.2, seed=42):
    """
    Perform stratified train/test split for single- or multi-gene datasets.

    Automatically uses label_dict keys for alignment and works without loading
    all embeddings into memory.

    Args:
        full_dataset: Dataset object with lookup/ids attribute.
        label_dict (dict): Mapping of sample IDs to resistance labels (0=resistant, 1=sensitive).
        test_size (float): Fraction of data to use for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_indices, test_indices, train_labels, test_labels)

    Raises:
        ValueError: If dataset type not recognized (missing lookup/ids attribute).
    """
    print("\n[Split] Performing stratified train/test split...")
    print(f"Dataset size: {len(full_dataset)} | Label map: {len(label_dict)}")

    # Extract ordered sample IDs from the dataset
    if hasattr(full_dataset, "lookup"):  # e.g., TokenMemmapMap
        seq_ids = [full_dataset.blocks[bidx][0][ridx] for bidx, ridx in full_dataset.lookup]
    elif hasattr(full_dataset, "ids"):  # e.g., MultiGeneConcatDataset
        seq_ids = full_dataset.ids
    else:
        raise ValueError("Dataset type not recognized: missing lookup/ids attribute")

    # Filter samples present in label_dict
    valid_ids = [sid for sid in seq_ids if sid in label_dict]
    labels = np.array([label_dict[sid] for sid in valid_ids], dtype=float)
    print(f"Valid samples found: {len(valid_ids)}")

    # Convert sample IDs to dataset indices
    id_to_idx = {sid: idx for idx, sid in enumerate(seq_ids)}
    valid_indices = np.array([id_to_idx[sid] for sid in valid_ids])

    # Stratified split using sklearn
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(valid_indices, labels))

    train_indices = valid_indices[train_idx]
    test_indices = valid_indices[test_idx]

    y_train, y_test = labels[train_idx], labels[test_idx]

    return train_indices, test_indices, y_train, y_test


def build_label_map(label_file, drug, prefix="full"):
    """
    Build a label map for a specific drug from phenotype data.

    Creates a mapping from sample identifiers to binary phenotype labels
    (0 = resistant, 1 = sensitive), filtering out invalid labels (-1).
    Automatically infers the drug index from available metadata.

    Args:
        label_file (str): Path to NPZ file containing phenotype matrix.
                          Expected key: "phenotypes" with shape (num_samples, num_drugs).
                          Optional keys: "drug_names" or "drugs" for metadata.
        drug (str): Drug name to extract labels for.
        prefix (str): Prefix for sample identifiers (default: "full").
                      Output IDs will be formatted as "{prefix}_{index:06d}".

    Returns:
        tuple: (label_map, drug_index)
            - label_map (dict): Maps sample IDs to binary phenotype labels.
            - drug_index (int): Column index of the drug in the phenotype matrix.

    Raises:
        KeyError: If drug index cannot be inferred from available data sources.
    """
    print(f"Loading labels from: {label_file}")

    # Automatically infer drug index
    drug_index = _infer_drug_index(label_file, drug)

    # Load phenotype data
    label_data = np.load(label_file)
    phenotypes = label_data["phenotypes"]  # Shape: (num_samples, num_drugs)
    drug_labels = phenotypes[:, drug_index]

    print(f"Drug: {drug} (column index {drug_index})")
    print(f"Total samples (including missing labels): {len(drug_labels)}")

    # Filter out invalid labels (-1)
    valid_indices = np.where(drug_labels != -1)[0]
    valid_labels = drug_labels[valid_indices]
    print(f"Valid samples with labels: {len(valid_labels)}")

    # Build sample ID to label mapping
    label_map = {
        f"{prefix}_{i:06d}": float(valid_labels[j])
        for j, i in enumerate(valid_indices)
    }

    return label_map, drug_index


def load_dataset_and_model(args, device):
    """
    Load dataset and get model dimensions.

    Args:
        args: Argument namespace with dataset configuration.
        device: PyTorch device.

    Returns:
        tuple: (full_dataset, model_dim, model_seq_len)
    """
    memmap_dir = args.saved_embed_memmap_dir
    prefix = "full"

    print(f"Loading dataset for drug: {args.drug}")
    print(f"Embedding type: {args.embed_type}")

    # Build label map for the drug
    full_label_map, drug_index = build_label_map(args.phenotype_label_path, args.drug, prefix=prefix)

    # Load dataset based on number of genes
    if len(DRUG_TO_LOCI[args.drug]) == 1:
        print(f"\nSingle gene drug {args.drug} selected, using per token embeddings.")
        gene = DRUG_TO_LOCI[args.drug][0]
        print(f"Using gene: {gene}")

        # Load meta file paths
        meta_paths = sorted(glob.glob(f"{memmap_dir}/{gene}/*_{args.embed_type}_meta.npz"))
        print(f"\nFound {len(meta_paths)} meta files")

        # Construct dataset based on embedding type
        if args.embed_type == 'token':
            print(f"Using {args.embed_type} embeddings")
            full_dataset = TokenMemmapMap(meta_paths, full_label_map)
        elif args.embed_type == 'pca':
            print(f"Using {args.embed_type} embeddings with (k={args.pca_components})")
            full_dataset = PcaMemmapMap(meta_paths, full_label_map, k=args.pca_components)
        else:
            print(f"Using {args.embed_type} embeddings")
            full_dataset = MeanMemmapMap(meta_paths, full_label_map, embed_type=args.embed_type)

    else:
        print(f"\nMultiple gene drug {args.drug} selected, concatenating gene embeddings.")
        loci = DRUG_TO_LOCI[args.drug]
        gene_memmap_dirs = [f"{memmap_dir}/{gene}/" for gene in loci]

        # Construct multi-gene dataset based on embedding type
        if args.embed_type == 'token':
            print(f"Using {args.embed_type} embeddings")
            full_dataset = MultiGeneConcatDataset(gene_memmap_dirs, full_label_map)
        elif args.embed_type == 'pca':
            print(f"Using {args.embed_type} embeddings with (k={args.pca_components})")
            full_dataset = PcaMultiGeneConcatDataset(gene_memmap_dirs, full_label_map, k=args.pca_components)
        else:
            print(f"Using {args.embed_type} embeddings")
            full_dataset = MeanMultiGeneConcatDataset(gene_memmap_dirs, full_label_map, embed_type=args.embed_type)

        print(f"Concatenated embedding shape: {full_dataset[0][0].shape} (D, L)")

    embeds, _ = full_dataset[0]
    model_dim = embeds.shape[0]
    model_seq_len = embeds.shape[1]

    print(f"\nModel dimensions: D={model_dim}, L={model_seq_len}")
    print(f"Total dataset size: {len(full_dataset)}")

    return full_dataset, model_dim, model_seq_len, full_label_map


def load_or_compute_threshold(model, train_dataloader, threshold_dir, drug, random_seed, device):
    """
    Load existing threshold or compute from training data.

    Args:
        model: Trained model.
        train_dataloader: DataLoader for training data.
        threshold_dir (str): Directory containing threshold files.
        drug (str): Drug name.
        random_seed (int): Random seed used for training.
        device: PyTorch device.

    Returns:
        float: Decision threshold.
    """
    print("\nHandling threshold calculation/loading...")

    seed_path = os.path.join(threshold_dir, f"{drug}/seed_{random_seed}")
    os.makedirs(seed_path, exist_ok=True)
    threshold_file = os.path.join(seed_path, "threshold.txt")

    print(f"Threshold file path: {threshold_file}")

    # Load existing threshold if available
    if os.path.exists(threshold_file):
        print(f"Loading existing threshold from {threshold_file}")
        with open(threshold_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Threshold:"):
                    threshold = float(line.split(":")[1].strip())
        print(f"Loaded threshold: {threshold}")
        return threshold

    # Calculate threshold from training data
    print(f"Threshold not found at {threshold_file}. Calculating from training data...")
    print("Predicting on training data to get AUC thresholds...")
    y_train, y_train_pred = evaluate(model, train_dataloader, device)

    threshold = calculate_single_drug_threshold(
        y_train.ravel(),
        y_train_pred.ravel(),
        get_threshold_val=ThresholdValue()
    )

    print(f"Calculated threshold: {threshold}")

    # Save threshold to file
    with open(threshold_file, 'w') as f:
        f.write(f"Drug: {drug}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Random seed: {random_seed}\n")
    print(f"Threshold saved to: {threshold_file}")

    return threshold


def main(args):
    """
    Main evaluation pipeline.

    Loads model, performs stratified train/test split, computes threshold,
    and evaluates on test set, saving results.

    Args:
        args: Argument namespace from argparse.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"\n{n_gpu} GPUs available to use!")

    # Load dataset and get dimensions
    full_dataset, model_dim, model_seq_len, full_label_map = load_dataset_and_model(args, device)

    # Stratified train/test split
    train_idx, test_idx, train_labels, test_labels = stratified_split_dataset(
        full_dataset=full_dataset,
        label_dict=full_label_map,
        test_size=args.test_split,
        seed=args.random_seed
    )

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print(f"\nTraining samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Training set: {np.sum(train_labels == 0)} R, {np.sum(train_labels == 1)} S")
    print(f"Test set: {np.sum(test_labels == 0)} R, {np.sum(test_labels == 1)} S")

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False)

    # Load model
    print("\nLoading trained model...")
    model = get_model_class(
        model_name=args.saved_model_name,
        in_dim=model_dim,
        seq_len=model_seq_len,
        device=device
    )

    # Load model checkpoint
    seed_path = os.path.join(args.saved_model_path, f"{args.drug}/seed_{args.random_seed}/fold_4")
    model_path = os.path.join(seed_path, f"{args.saved_model_name}.pt")

    print(f"Loading model from {model_path}...\n")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load or compute threshold
    threshold = load_or_compute_threshold(
        model, train_dataloader, args.threshold_dir, args.drug, args.random_seed, device
    )

    # Evaluate on test data
    print("\nEvaluating on test data...")
    y_test, y_test_pred = evaluate(model, test_dataloader, device)
    test_results = calculate_test_metrics_single_drug(
        y_test.ravel(),
        y_test_pred.ravel(),
        threshold,
        drug_name=args.drug
    )

    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    seed_path = os.path.join(args.output_path, f"{args.drug}/seed_{args.random_seed}")
    os.makedirs(seed_path, exist_ok=True)
    test_results_file = os.path.join(seed_path, f"test_set_auc_{args.drug}.csv")
    test_results.to_csv(test_results_file)

    print(f"\nTest results saved to: {test_results_file}")
    print(test_results)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate downstream resistance task model'
    )

    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        default="DNABERTCNN",
        help='Model name'
    )
    parser.add_argument(
        '--saved_model_name',
        type=str,
        default='dnabert-mdcnn_cv_split_0.pt',
        help='Name of the saved model'
    )

    # Data paths
    parser.add_argument(
        '--saved_embed_memmap_dir',
        type=str,
        default='/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/full',
        help='Saved embeds directory containing memmaps'
    )
    parser.add_argument(
        '--phenotype_label_path',
        type=str,
        default='training_output/zero_shot/token_embeddings_5000/dnabert2/zs_train_stacked_phenotypes.npz',
        help='Path to the phenotype labels npz file'
    )
    parser.add_argument(
        '--saved_model_path',
        type=str,
        default='training_output/transfer_learn/saved_models/dnabert2',
        help='Directory containing saved model'
    )
    parser.add_argument(
        '--threshold_dir',
        type=str,
        default='training_output/zero_shot/saved_parameters/dnabert2/token_embeds',
        help='Path to threshold values directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='training_output/transfer_learn/classification_results',
        help='Directory to save results'
    )

    # Data configuration
    parser.add_argument(
        '--max_length',
        type=int,
        default=5000,
        help='Max length of tokens'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=128,
        help='Batch size used for training dataset'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=128,
        help='Batch size used for validation dataset'
    )
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.2,
        help='Test split ratio'
    )

    # Embedding configuration
    parser.add_argument(
        '--embed_type',
        type=str,
        default='token',
        help="Type of embedding to use. Options: 'token', 'mean', 'pca'"
    )
    parser.add_argument(
        '--pca_components',
        type=int,
        default=10,
        help='Number of PCA components to keep if embed_type is pca'
    )

    # Drug and random seed
    parser.add_argument(
        '--drug',
        type=str,
        default='ISONIAZID',
        help='Drug to use for classification'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for train/test split'
    )

    args = parser.parse_args()
    main(args)
