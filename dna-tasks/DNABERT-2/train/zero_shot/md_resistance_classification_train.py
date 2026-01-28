"""
Multi-Drug Antimicrobial Resistance Classification Training

This module trains deep learning models for multi-drug antimicrobial resistance
prediction using DNABERT-2 embeddings. It supports loading embeddings from various
formats (NPZ, HDF5) and includes utilities for gene selection and embedding processing.

Key capabilities:
- Multi-drug resistance prediction across all antimicrobial drugs
- Flexible embedding loading from NPZ and HDF5 formats
- Optional PCA-based dimensionality reduction
- Gene-specific embedding selection
- GPU-accelerated 5-fold cross-validation training
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from downstream_cnn_model import *
from dataloader.locus_order import DRUGS
from utils.embed_gen_utils import *
from utils.classification_metric_utils import *
from utils.token_train_utils import *


def load_embeddings_and_labels_h5(embed_path, batch_size=128):
    """
    Efficiently load embeddings and labels from HDF5 file.

    Loads data in batches without loading the full file into memory, filtering
    out samples with missing phenotypes across all drugs.

    Args:
        embed_path (str): Path to HDF5 file containing embeddings and phenotypes.
        batch_size (int): Number of samples to load per batch.

    Returns:
        tuple: (embeddings_filtered, labels_filtered)
            - embeddings_filtered: Shape (num_filtered_samples, num_genes, seq_len), dtype=float32
            - labels_filtered: Shape (num_filtered_samples, num_drugs)
    """
    all_embeddings = []
    all_labels = []

    with h5py.File(embed_path, 'r') as h5f:
        embeddings_dset = h5f["embeddings"]
        labels_dset = h5f["phenotypes"]

        total_samples = labels_dset.shape[0]
        num_drugs = labels_dset.shape[1]
        print(f"Total samples: {total_samples}")
        print("Embeddings shape:", embeddings_dset.shape)
        print("Labels shape:", labels_dset.shape)

        for start in tqdm.tqdm(range(0, total_samples, batch_size), desc="Filtering valid samples"):
            end = min(start + batch_size, total_samples)

            batch_emb = embeddings_dset[start:end]
            batch_lbl = labels_dset[start:end]

            # Keep samples with at least one valid phenotype
            mask = np.sum(batch_lbl == -1, axis=1) != num_drugs
            if np.any(mask):
                all_embeddings.append(batch_emb[mask])
                all_labels.append(batch_lbl[mask])

    embeddings_filtered = np.concatenate(all_embeddings, axis=0)
    labels_filtered = np.concatenate(all_labels, axis=0)

    print("Filtered embeddings shape:", embeddings_filtered.shape)
    print("Filtered labels shape:", labels_filtered.shape)
    return embeddings_filtered.astype(np.float32), labels_filtered


def concatenate_gene_embeddings(embeddings, use_pca=False, pca_components=10):
    """
    Concatenate embeddings across genes with optional PCA dimensionality reduction.

    Combines embeddings from multiple genes into a single representation,
    optionally applying PCA to reduce the dimensionality.

    Args:
        embeddings (torch.Tensor): Input embeddings of shape (num_samples, embedding_dim, seq_len, num_genes).
        use_pca (bool): If True, apply PCA to reduce dimensionality.
        pca_components (int): Number of PCA components to keep (only used if use_pca=True).

    Returns:
        torch.Tensor: Concatenated embeddings.
            - If use_pca=True: Shape (num_samples, pca_components, seq_len * num_genes)
            - If use_pca=False: Shape (num_samples, embedding_dim, seq_len * num_genes)
    """
    if use_pca:
        B, D, L, G = embeddings.shape  # (batch, dim, seq_len, num_genes)

        # Permute to (batch, seq_len, num_genes, dim)
        embeddings = embeddings.permute(0, 2, 3, 1)

        # Flatten to (batch * seq_len * num_genes, dim)
        flattened = embeddings.reshape(-1, D).cpu().numpy()

        # Fit PCA and transform
        pca = PCA(n_components=pca_components)
        reduced = pca.fit_transform(flattened)  # (batch * seq_len * num_genes, pca_components)

        # Reshape back to (batch, seq_len * num_genes, pca_components)
        reduced = torch.from_numpy(reduced).float().reshape(B, L * G, pca_components)

        # Permute to (batch, pca_components, seq_len * num_genes)
        concatenated_embeddings = reduced.permute(0, 2, 1)
    else:
        # Combine seq_len and gene dimensions
        concatenated_embeddings = embeddings.reshape(embeddings.size(0), embeddings.size(1), -1)

    return concatenated_embeddings


def main(args):
    """
    Main training pipeline for multi-drug resistance classification.

    Orchestrates the full training workflow:
    1. Setup GPU environment
    2. Load embeddings and phenotype labels from HDF5
    3. Convert to tensors and compute statistics
    4. Create dataset and dataloaders
    5. Initialize loss function and metrics
    6. Execute 5-fold cross-validation training

    Args:
        args: Command-line arguments with training configuration.
    """
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"\n{n_gpu} GPUs available to use!")

    print("=" * 70)
    print("Multi-Drug Antimicrobial Resistance Classification Training")
    print("=" * 70)

    # Load embeddings and labels from HDF5
    print("\nLoading embeddings and labels...")
    train_embeddings, train_labels = load_embeddings_and_labels_h5(
        args.embed_path_h5,
        batch_size=args.train_batch_size
    )
    print(f"Train embeddings shape: {train_embeddings.shape}")  # (num_samples, num_genes, seq_len)
    print(f"Train labels shape: {train_labels.shape}")  # (num_samples, num_drugs)

    # Convert to tensors
    train_embeddings = torch.tensor(train_embeddings).permute(0, 2, 1)
    train_labels = torch.tensor(train_labels)

    # Compute and display embedding statistics
    min_value = train_embeddings.min().item()
    max_value = train_embeddings.max().item()
    mean_value = train_embeddings.mean().item()
    std_value = train_embeddings.std().item()

    print(f"\nEmbedding Statistics:")
    print(f"  Min Value:  {min_value:.6f}")
    print(f"  Max Value:  {max_value:.6f}")
    print(f"  Mean Value: {mean_value:.6f}")
    print(f"  Std Dev:    {std_value:.6f}")

    # Create dataset and dataloader
    print("\nCreating dataset and dataloader...")
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    model_seq_len = train_embeddings.shape[2]
    model_name = "DNABERTCNN"

    # Initialize loss function and metrics
    print("\nInitializing loss function and metrics...")
    criterion = MaskedMultiWeightedBCE()
    acc_metric = MaskedWeightedAccuracy()
    auc_threshold = ThresholdValue()

    # 5-fold cross-validation training
    print("\nStarting 5-fold cross-validation training...")
    print(f"  Model: {model_name}")
    print(f"  Sequence length: {model_seq_len}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Train batch size: {args.train_batch_size}")
    print(f"  Val batch size: {args.val_batch_size}")
    print("=" * 70)

    trained_model = train_kfold_mod(
        train_dataset,
        DRUGS,
        criterion,
        args.learning_rate,
        args.weight_decay,
        acc_metric,
        auc_threshold,
        output_path=args.output_path,
        saved_model_path=args.saved_model_path,
        model_name=model_name,
        model_seq_len=model_seq_len,
        k_folds=5,
        epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        random_seed=args.random_seed,
        device=device
    )

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print(f"Model checkpoints saved to: {args.saved_model_path}")
    print(f"Results saved to: {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train multi-drug antimicrobial resistance classification model'
    )

    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        default="zhihan1996/DNABERT-S",
        help='Model name (for reference; actual model set to DNABERTCNN in main)'
    )

    # Data paths
    parser.add_argument(
        '--saved_embed_dir',
        type=str,
        default='training_output/transfer_learn/embeddings',
        help='Directory containing saved embeddings'
    )
    parser.add_argument(
        '--embed_path_h5',
        type=str,
        default='training_output/zero_shot/token_embeddings_5000/dnabert2/RIFAMPICIN/zs_train_FIRST_embed_pheno.h5',
        help='Path to HDF5 file with embeddings and phenotypes'
    )

    # Training hyperparameters
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=128,
        help='Batch size for validation'
    )
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.2,
        help='Test split ratio'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay for optimizer'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--freeze_bias_frac',
        type=float,
        default=0.25,
        help='Fraction of training for which to freeze bias'
    )

    # Output paths
    parser.add_argument(
        '--output_path',
        type=str,
        default='training_output/transfer_learn/classification_results',
        help='Directory to save training results'
    )
    parser.add_argument(
        '--saved_model_path',
        type=str,
        default='training_output/transfer_learn/saved_models',
        help='Directory to save model checkpoints'
    )

    # Other options
    parser.add_argument(
        '--random_seed',
        type=int,
        default=1,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--use_pca',
        action='store_true',
        help='Use PCA for dimensionality reduction'
    )
    parser.add_argument(
        '--pca_components',
        type=int,
        default=10,
        help='Number of PCA components to keep'
    )

    args = parser.parse_args()
    main(args)
