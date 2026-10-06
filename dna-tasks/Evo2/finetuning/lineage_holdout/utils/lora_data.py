"""Data loading for LoRA fine-tuning.

This module integrates with the existing FASTA loading pipeline to create
sequence datasets for end-to-end training.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset

from evo2_downstream.config import ensure_finetune_utils_on_path

try:
    from finetuning.modules.dataloader.locus_order import DRUG_TO_LOCI
except ImportError:  # pragma: no cover - legacy fallback path
    ensure_finetune_utils_on_path()
    from dataloader.locus_order import DRUG_TO_LOCI  # type: ignore  # noqa: E402


def load_sequences_from_fasta(gene_name: str, data_dir: Path) -> dict[str, str]:
    """Load sequences from a gene FASTA file.
    
    Args:
        gene_name: Gene name (e.g., 'rpoB')
        data_dir: Directory containing aligned FASTA files
        
    Returns:
        Dict mapping isolate_id -> sequence
    """
    fasta_pattern = data_dir / f"{gene_name}*.fasta"
    fasta_files = list(glob.glob(str(fasta_pattern)))
    
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA file found for gene {gene_name} in {data_dir}")
    
    fasta_file = fasta_files[0]
    sequences = {}
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        isolate_id = record.id
        sequence = str(record.seq)
        sequences[isolate_id] = sequence
    
    return sequences


def concatenate_gene_sequences(isolate_id: str, gene_names: list[str], gene_sequences: dict[str, dict[str, str]]) -> str:
    """Concatenate sequences from multiple genes for one isolate.
    
    Args:
        isolate_id: Isolate identifier
        gene_names: List of gene names in order
        gene_sequences: Dict of {gene_name: {isolate_id: sequence}}
        
    Returns:
        Concatenated sequence
    """
    parts = []
    for gene in gene_names:
        if isolate_id in gene_sequences[gene]:
            parts.append(gene_sequences[gene][isolate_id])
        else:
            # Missing gene - skip this isolate
            return None
    return "".join(parts)


def load_drug_sequences_and_labels(
    drug: str,
    geno_pheno_csv: Path,
    fasta_dir: Path,
    prefix: str = "full",
) -> tuple[list[str], list[str], list[int]]:
    """Load sequences and labels for a specific drug.
    
    Args:
        drug: Drug name (e.g., 'ISONIAZID')
        geno_pheno_csv: Path to geno_pheno_full_combined.csv
        fasta_dir: Directory containing gene FASTA files
        prefix: Isolate ID prefix (e.g., 'full')
        
    Returns:
        isolate_ids, sequences, labels
    """
    # Load phenotype labels
    df_labels = pd.read_csv(geno_pheno_csv, index_col=0, low_memory=False)
    
    # Filter for drug
    if drug not in df_labels.columns:
        raise ValueError(f"Drug {drug} not found in {geno_pheno_csv}")
    
    # Get gene loci for this drug
    gene_names = DRUG_TO_LOCI[drug]
    
    # Load sequences for each gene
    print(f"Loading sequences for {len(gene_names)} genes: {gene_names}")
    gene_sequences = {}
    for gene in gene_names:
        gene_sequences[gene] = load_sequences_from_fasta(gene, fasta_dir)
        print(f"  {gene}: {len(gene_sequences[gene])} isolates")
    
    # Build dataset
    isolate_ids = []
    sequences = []
    labels = []
    
    for isolate_id in df_labels.index:
        # Convert isolate ID to match FASTA IDs
        isolate_id_str = str(isolate_id)
        fasta_id = f"{prefix}_{isolate_id_str}" if prefix and not isolate_id_str.startswith(prefix) else isolate_id_str

        # Check if isolate has sequence data for all genes
        concat_seq = concatenate_gene_sequences(fasta_id, gene_names, gene_sequences)
        if concat_seq is None:
            continue
        
        # Get label
        label_val = df_labels.loc[isolate_id, drug]
        if label_val not in ["R", "S", 0, 1]:
            continue  # Skip missing labels
        
        # Convert label: R=0 (resistant), S=1 (susceptible)
        label = 1 if label_val in ["S", 1] else 0
        
        isolate_ids.append(fasta_id)
        sequences.append(concat_seq)
        labels.append(label)
    
    print(f"\nLoaded {len(sequences)} isolates for drug {drug}")
    num_resistant = sum(1 for l in labels if l == 0)
    num_susceptible = sum(1 for l in labels if l == 1)
    print(f"  Resistant (R): {num_resistant}")
    print(f"  Susceptible (S): {num_susceptible}")
    
    return isolate_ids, sequences, labels


def apply_lineage_split(
    isolate_ids: list[str],
    sequences: list[str],
    labels: list[int],
    heldout_lineage: str,
    isolate_id_map: dict[str, str],
    lineage_map: dict[str, str],
) -> tuple[list[int], list[int]]:
    """Apply lineage-aware train/test split.
    
    Args:
        isolate_ids: List of isolate IDs
        sequences: List of sequences
        labels: List of labels
        heldout_lineage: Lineage to hold out (e.g., '2')
        isolate_id_map: Maps full_N -> actual isolate ID
        lineage_map: Maps isolate ID -> lineage
        
    Returns:
        train_indices, test_indices
    """
    train_indices = []
    test_indices = []

    for idx, fasta_id in enumerate(isolate_ids):
        # The FASTA IDs may already be the real isolate IDs, or may be full_N memmap style.
        if fasta_id in lineage_map:
            actual_id = fasta_id
        elif fasta_id in isolate_id_map:
            actual_id = isolate_id_map[fasta_id]
        else:
            # Try memmap-style full_N -> row index mapping
            base_id = fasta_id.split("_", 1)[-1] if "_" in fasta_id else fasta_id
            try:
                row_idx = int(base_id)
                actual_id = isolate_id_map.get(row_idx)
            except ValueError:
                actual_id = None
            if actual_id is None:
                print(f"Warning: {fasta_id} not found in isolate_id_map")
                continue

        # Get lineage
        # Match the existing lineage-aware baseline semantics: isolates without
        # a lineage annotation stay in training rather than being silently
        # dropped.  Only the explicitly held-out lineage goes to test.
        lineage = lineage_map.get(actual_id)
        if lineage is None:
            print(f"Warning: {actual_id} has no lineage annotation; keeping it in training")
            train_indices.append(idx)
        elif lineage == heldout_lineage:
            test_indices.append(idx)
        else:
            train_indices.append(idx)
    
    return train_indices, test_indices


def create_validation_split(
    train_indices: list[int],
    labels: list[int],
    val_frac: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Create stratified validation split from training indices.
    
    Args:
        train_indices: Training indices
        labels: Full label list
        val_frac: Fraction for validation
        random_seed: Random seed
        
    Returns:
        new_train_indices, val_indices
    """
    from sklearn.model_selection import train_test_split
    
    # Get labels for train indices
    train_labels = [labels[i] for i in train_indices]
    
    # Stratified split
    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=val_frac,
        stratify=train_labels,
        random_state=random_seed,
    )
    
    return list(train_idx), list(val_idx)


class SequenceDataset(Dataset):
    """Dataset that returns raw DNA sequences + labels for on-the-fly embedding."""
    
    def __init__(self, sequences: list[str], labels: list[int], indices: list[int] | None = None):
        """
        Args:
            sequences: Full list of sequences
            labels: Full list of labels
            indices: Optional subset of indices to use
        """
        if indices is not None:
            self.sequences = [sequences[i] for i in indices]
            self.labels = [labels[i] for i in indices]
        else:
            self.sequences = sequences
            self.labels = labels
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.sequences[idx], self.labels[idx]


def collate_sequences(batch: list[tuple[str, int]]) -> tuple[list[str], torch.Tensor]:
    """Collate function that keeps sequences as strings."""
    sequences, labels = zip(*batch)
    return list(sequences), torch.tensor(labels, dtype=torch.float32)
