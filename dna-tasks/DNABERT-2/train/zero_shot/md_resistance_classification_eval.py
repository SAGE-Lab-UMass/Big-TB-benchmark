"""
Multi-Drug Antimicrobial Resistance Classification Evaluation

This module evaluates a trained multi-drug resistance classification model
on test data. It loads pre-trained embeddings, computes decision thresholds
from training data, and evaluates performance on a held-out test set.

Key capabilities:
- Load embeddings and phenotypes from NPZ files
- Filter samples with valid phenotype labels
- Compute optimal decision thresholds per drug
- Evaluate multi-drug resistance predictions
- Generate comprehensive performance metrics
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from downstream_cnn_model import *
from utils.embed_gen_utils import *
from utils.classification_metric_utils import *
from utils.multigene_train_utils import *


def load_embeddings_and_labels(embed_path):
    """
    Load embeddings and phenotype labels from NPZ file.

    Args:
        embed_path (str): Path to NPZ file containing embeddings and phenotypes.

    Returns:
        tuple: (embeddings, labels)
            - embeddings: Shape (num_samples, num_genes, embedding_dim)
            - labels: Shape (num_samples, num_drugs)
    """
    data = np.load(embed_path)
    embeddings = data['embeddings']
    labels = data['phenotypes']

    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded phenotypes shape: {labels.shape}")

    return embeddings, labels


def filter_valid_samples(embeddings, labels):
    """
    Filter out samples with missing phenotypes across all drugs.

    Keeps only samples that have at least one valid resistance label
    (i.e., not all -1 values for all drugs).

    Args:
        embeddings (np.ndarray): Shape (num_samples, num_genes, embedding_dim)
        labels (np.ndarray): Shape (num_samples, num_drugs)

    Returns:
        tuple: (embeddings_filtered, labels_filtered)
            - embeddings_filtered: Shape (num_filtered_samples, num_genes, embedding_dim)
            - labels_filtered: Shape (num_filtered_samples, num_drugs)
    """
    print("\nFiltering isolates with at least 1 resistance status across all drugs...")

    num_drugs = labels.shape[-1]
    valid_indices = np.where(labels.sum(axis=1) != -num_drugs)[0]

    embeddings_filtered = embeddings[valid_indices, :, :]
    labels_filtered = labels[valid_indices, :]

    print(f"Filtered to {len(valid_indices)} valid samples")

    return embeddings_filtered, labels_filtered


def prepare_tensors_and_dataloaders(train_embeddings, train_labels, test_embeddings, test_labels, batch_size):
    """
    Convert embeddings and labels to tensors and create dataloaders.

    Args:
        train_embeddings (np.ndarray): Training embeddings of shape (num_train, num_genes, embedding_dim)
        train_labels (np.ndarray): Training labels of shape (num_train, num_drugs)
        test_embeddings (np.ndarray): Test embeddings of shape (num_test, num_genes, embedding_dim)
        test_labels (np.ndarray): Test labels of shape (num_test, num_drugs)
        batch_size (int): Batch size for dataloaders.

    Returns:
        tuple: (train_dataloader, test_dataloader, train_embeddings_tensor, test_embeddings_tensor)
    """
    # Convert to tensors and permute for CNN input format (batch_size, embedding_dim, num_genes)
    train_embeddings_tensor = torch.tensor(train_embeddings).permute(0, 2, 1)
    train_labels_tensor = torch.tensor(train_labels)

    test_embeddings_tensor = torch.tensor(test_embeddings).permute(0, 2, 1)
    test_labels_tensor = torch.tensor(test_labels)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, train_embeddings_tensor, test_embeddings_tensor


def main(args):
    """
    Main evaluation pipeline.

    Loads pre-trained model and embeddings, computes per-drug decision thresholds
    from training data, and evaluates performance on test data.

    Args:
        args: Command-line arguments with evaluation configuration.
    """
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"\n{n_gpu} GPUs available to use!")

    print("=" * 70)
    print("Multi-Drug Resistance Classification Evaluation")
    print("=" * 70)

    # Load embeddings and labels
    print("\nLoading training embeddings and labels...")
    train_embeddings, train_labels = load_embeddings_and_labels(
        os.path.join(args.saved_embed_dir, args.train_embed_name)
    )

    print("\nLoading test embeddings and labels...")
    test_embeddings, test_labels = load_embeddings_and_labels(
        os.path.join(args.saved_embed_dir, args.val_embed_name)
    )

    # Filter samples with valid phenotypes
    train_embeddings, train_labels = filter_valid_samples(train_embeddings, train_labels)
    test_embeddings, test_labels = filter_valid_samples(test_embeddings, test_labels)

    print(f"\nTrain set: {train_embeddings.shape[0]} samples")
    print(f"Test set: {test_embeddings.shape[0]} samples")

    # Prepare tensors and dataloaders
    print("\nPreparing tensors and dataloaders...")
    train_dataloader, test_dataloader, _, _ = prepare_tensors_and_dataloaders(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        batch_size=args.train_batch_size
    )

    # Load model
    print("\nLoading trained model...")
    model = MDDNABERTCNN(dropout_rate=0)
    model = model.to(device)

    saved_path = os.path.join(args.saved_model_path, args.saved_model_name)
    print(f"Loading model from: {saved_path}")
    model.load_state_dict(torch.load(saved_path, weights_only=True))

    # Compute thresholds from training data
    print("\nComputing decision thresholds from training data...")
    auc_threshold = ThresholdValue()
    y_train, y_train_pred = evaluate(model, train_dataloader, device)
    auc_thresholds, drug_to_threshold = calculate_auc_thresholds(y_train, y_train_pred, auc_threshold)

    print("Per-drug thresholds computed:")
    for drug, threshold in drug_to_threshold.items():
        print(f"  {drug}: {threshold:.4f}")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    y_test, y_test_pred = evaluate(model, test_dataloader, device)
    test_results = calculate_test_auc(y_test, y_test_pred, drug_to_threshold)

    # Save results
    print("\nSaving results...")
    os.makedirs(args.output_path, exist_ok=True)
    results_file = os.path.join(args.output_path, "test_set_auc.csv")
    test_results.to_csv(results_file)
    print(f"Results saved to: {results_file}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)
    print("\nTest Results:")
    print(test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate multi-drug antimicrobial resistance classification model'
    )

    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        default="zhihan1996/DNABERT-S",
        help='Model name for reference'
    )

    # Data paths
    parser.add_argument(
        '--saved_embed_dir',
        type=str,
        default='training_output/transfer_learn/embeddings',
        help='Directory containing saved embeddings'
    )
    parser.add_argument(
        '--train_embed_name',
        type=str,
        default='zs_train_embeddings_phenotypes.npz',
        help='Name of training embeddings file'
    )
    parser.add_argument(
        '--val_embed_name',
        type=str,
        default='zs_val_embeddings_phenotypes.npz',
        help='Name of validation/test embeddings file'
    )

    # Model loading
    parser.add_argument(
        '--saved_model_path',
        type=str,
        default='training_output/transfer_learn/saved_models',
        help='Directory containing saved model checkpoints'
    )
    parser.add_argument(
        '--saved_model_name',
        type=str,
        default='dnabert-mdcnn_cv_split_0.pt',
        help='Name of the saved model checkpoint'
    )

    # Data configuration
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=128,
        help='Batch size for data loading'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=128,
        help='Batch size for validation (not used in eval)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=5000,
        help='Maximum sequence length'
    )

    # Output
    parser.add_argument(
        '--output_path',
        type=str,
        default='training_output/transfer_learn/classification_results',
        help='Directory to save evaluation results'
    )

    # Other options
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.2,
        help='Test split ratio (for reference only)'
    )

    args = parser.parse_args()
    main(args)
