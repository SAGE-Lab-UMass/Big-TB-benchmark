"""
Data loading utilities for multi-gene, multi-drug MTB dataset.

This module provides PyTorch Dataset classes for loading DNA sequences and
resistance phenotypes for multiple genes and drugs, supporting various
embedding compression formats (token-level, PCA, mean embeddings).
"""

import os
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as util_data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from dataloader.dataloader_utils import make_geno_pheno_pkl
from dataloader.locus_order import DRUG_INDEX, DRUGS as drugs

class MultigeneMultidrugSamples(Dataset):
    """Dataset for multi-gene, multi-drug resistance phenotypes."""

    def __init__(self, sequences, res_phenotypes, gene_names, drug_names, num_genes):
        """
        Initialize the dataset.

        Args:
            sequences: List of length num_genes, each element is a list of sequences
                      for one gene across all isolates.
            res_phenotypes: List of length num_drugs, each element is a list of resistance
                           phenotypes for one drug across all isolates.
            gene_names: List of gene names.
            drug_names: List of drug names.
            num_genes: Expected number of genes.
        """
        assert len(sequences) == num_genes

        self.sequences = sequences
        self.res_phenotypes = res_phenotypes
        self.gene_names = gene_names
        self.drug_names = drug_names

    def __len__(self) -> int:
        """Return the number of isolates in the dataset."""
        return len(self.res_phenotypes[0])

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve an isolate at the specified index.

        Args:
            idx: Index of the isolate to retrieve.

        Returns:
            Dictionary with:
            - gene_seq_1, gene_seq_2, ...: DNA sequences for each gene
            - res_phenotype_drug_1, res_phenotype_drug_2, ...: Resistance labels
            - gene_order: List of gene names
            - drug_order: List of drug names
        """
        item = {
            **{
                f'gene_seq_{i+1}': self.sequences[i][idx]
                for i in range(len(self.sequences))
            },
            **{
                f'res_phenotype_drug_{i+1}': self.res_phenotypes[i][idx]
                for i in range(len(drugs))
            },
            'gene_order': self.gene_names,
            'drug_order': self.drug_names
        }
        return item


def multi_gene_multi_drug_loader_csv(args, load_train: bool = True, n_gpu: int = 1) -> util_data.DataLoader:
    """
    Load multi-gene, multi-drug data from CSV and create a DataLoader.

    Assumes CSV format: DNA sequences (with .fasta suffix) and drug phenotypes.

    Args:
        args: Configuration object with datapath, train_dataname, val_dataname attributes.
        load_train: If True, load training data; otherwise load validation data.
        n_gpu: Number of GPUs for batch size scaling.

    Returns:
        PyTorch DataLoader with MultigeneMultidrugSamples dataset.
    """
    delimiter = ","

    _ = create_multidrug_classification_data(args, delimiter)

    # Load the data from CSV
    csv_filename = args.train_dataname if load_train else args.val_dataname
    print(f"Loading data from {csv_filename}...")

    with open(os.path.join(args.datapath, csv_filename)) as csvfile:
        reader = list(csv.reader(csvfile, delimiter=delimiter))
        headers = reader[0]
        data = reader[1:]

    # Identify gene columns (ending with .fasta)
    fasta_columns = [i for i, header in enumerate(headers) if header.endswith(".fasta")]
    gene_names = [header for header in headers if header.endswith(".fasta")]
    print(f"Number of genes: {len(fasta_columns)}")
    print(f"Number of drugs: {len(drugs)}\n")

    # Extract sequences
    sequences = [[row[i] for row in data] for i in fasta_columns]
    print(f"Sequences shape: {np.array(sequences).shape}\n")

    # Identify drug columns and extract resistance phenotypes
    drug_columns = [i for i, header in enumerate(headers) if header in drugs]
    drug_names = [header for header in headers if header in drugs]

    res_phenotypes = [[row[i] for row in data] for i in drug_columns]
    print(f"Resistance phenotypes shape: {np.array(res_phenotypes).shape}\n")

    # Convert resistance labels to numeric values
    print("Converting resistance labels to numeric values...")
    resistance_categories = {'R': 0, 'S': 1, '-1.0': -1, '-1': -1}
    res_phenotypes_label = [
        [resistance_categories[res] for res in res_phenotype]
        for res_phenotype in res_phenotypes
    ]
    print(f"Converted phenotypes shape: {np.array(res_phenotypes_label).shape}\n")

    # Create dataset and loader
    print("Creating MultigeneMultidrugSamples dataset...")
    dataset = MultigeneMultidrugSamples(
        sequences, res_phenotypes_label, gene_names=gene_names,
        drug_names=drug_names, num_genes=len(fasta_columns)
    )

    batch_size = args.train_batch_size if load_train else args.val_batch_size
    loader = util_data.DataLoader(
        dataset, batch_size=batch_size * n_gpu, shuffle=False, num_workers=4 * n_gpu
    )
    print("Done!\n")

    return loader


def create_multidrug_classification_data(args, delimiter: str) -> pd.DataFrame:
    """Create and split multi-drug classification data."""
    geno_pheno_df = create_genotype_phenotype_csv(args, delimiter)
    split_data_into_train_val_sets(args, geno_pheno_df)
    return geno_pheno_df


def split_data_into_train_val_sets(
    args, geno_pheno_df: pd.DataFrame, split_type: str = 'other'
) -> None:
    """
    Split data into training and validation sets.

    Args:
        args: Configuration object.
        geno_pheno_df: Genotype-phenotype DataFrame.
        split_type: Either 'custom' or 'other' for different splitting strategies.
    """
    geno_pheno_df = geno_pheno_df.reset_index(drop=True)

    if split_type == 'custom':
        train_indices = geno_pheno_df.query("category=='set1_original_10202'").index
        train_data = geno_pheno_df.loc[train_indices]
        test_indices = geno_pheno_df.query("category!='set1_original_10202'").index
        val_data = geno_pheno_df.loc[test_indices]
    else:
        print("Splitting data into train/validation sets (80/20 ratio)...")
        all_indices = geno_pheno_df.index
        train_indices, val_indices = train_test_split(
            all_indices, test_size=args.test_split, random_state=42
        )
        train_data = geno_pheno_df.loc[train_indices]
        val_data = geno_pheno_df.loc[val_indices]

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    del geno_pheno_df

    # Filter out samples with all missing resistance values
    print("Filtering rows with -1 (missing) labels for all drugs...")
    train_data = train_data[train_data[drugs].apply(lambda x: (x != -1).any(), axis=1)]
    val_data = val_data[val_data[drugs].apply(lambda x: (x != -1).any(), axis=1)]
    print("Done!\n")

    print(f"Training set size after filtering: {len(train_data)}")
    print(f"Validation set size after filtering: {len(val_data)}")

    # Save to CSV
    train_data.to_csv(os.path.join(args.datapath, args.train_dataname), index=False)
    val_data.to_csv(os.path.join(args.datapath, args.val_dataname), index=False)


def create_genotype_phenotype_csv(args, delimiter: str) -> pd.DataFrame:
    """
    Load or create the genotype-phenotype CSV file.

    Args:
        args: Configuration object.
        delimiter: CSV delimiter character.

    Returns:
        DataFrame with genotype-phenotype data.
    """
    data_path = os.path.join(args.datapath, args.pkl_file)

    if os.path.isfile(data_path):
        print("Genotype-phenotype file already exists, loading...")
    else:
        print("Creating genotype-phenotype file...")
        make_geno_pheno_pkl(args)

    print("Reading genotype-phenotype CSV...")
    geno_pheno_df = pd.read_csv(data_path, delimiter=delimiter)
    print("Done!\n")

    return geno_pheno_df


def build_label_map(label_file: str, drug: str, prefix: str = "train") -> tuple:
    """
    Build a label map from numpy phenotype file.

    Args:
        label_file: Path to .npz file containing phenotypes array.
        drug: Drug name to extract labels for.
        prefix: Prefix for sample IDs (e.g., 'train', 'val').

    Returns:
        Tuple of (label_map dict, drug_index int).
    """
    print(f"Loading labels from: {label_file}")
    print("**WARNING: DOUBLE CHECK THE ORDER OF THE DRUGS IN THE PHENOTYPE FILE WITH DRUG_INDEX MAPPING!**")
    drug_index = DRUG_INDEX[drug]

    label_np_file = np.load(label_file)
    labels = label_np_file["phenotypes"]  # shape: (num_samples, num_drugs)
    drug_labels = labels[:, drug_index]

    print(f"Building label map for drug: {drug} (index {drug_index})")
    print(f"Total samples (including missing): {len(drug_labels)}")

    # Keep only valid (non -1) labels
    valid_indices = np.where(drug_labels != -1)[0]
    drug_labels = drug_labels[valid_indices]
    print(f"Total valid samples: {len(drug_labels)}")

    # Build label map with sample IDs
    label_map = {
        f"{prefix}_{i:06d}": float(drug_labels[j])
        for j, i in enumerate(valid_indices)
    }

    return label_map, drug_index


class TokenMemmapMap(Dataset):
    """
    Dataset for token-level embeddings stored as memmap files.

    Loads full token embeddings (D, L) for each sequence.
    """

    def __init__(self, meta_paths: list, label_dict: dict):
        """
        Initialize token memmap dataset.

        Args:
            meta_paths: List of paths to _meta.npz files.
            label_dict: Dictionary mapping sequence IDs to labels.
        """
        self.blocks = []
        self.lookup = []
        self.label_dict = label_dict

        for bidx, meta_path in enumerate(meta_paths):
            meta = np.load(meta_path, allow_pickle=True)
            ids = meta["identifier"]
            shape = tuple(meta["shape"])
            mmap_path = meta_path.replace("_meta.npz", ".mmap")
            mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
            self.blocks.append((ids, mm))

            # Keep only rows with labels
            for r in range(shape[0]):
                seq_id = ids[r]
                if seq_id in self.label_dict:
                    self.lookup.append((bidx, r))

    def __len__(self) -> int:
        """Return number of labeled samples."""
        return len(self.lookup)

    def __getitem__(self, idx: int) -> tuple:
        """Return (embeddings, label) for sample at index."""
        bidx, ridx = self.lookup[idx]
        ids, mm = self.blocks[bidx]
        seq_id = ids[ridx]
        x = torch.from_numpy(mm[ridx].astype("float32")).t()  # (D, L)
        y = torch.tensor(self.label_dict[seq_id], dtype=torch.float32)
        return x, y


class PcaMemmapMap(Dataset):
    """
    Dataset for PCA-compressed token embeddings.

    Loads PCA-compressed embeddings of shape (K, L).
    """

    def __init__(self, meta_paths: list, label_dict: dict, k: int):
        """
        Initialize PCA memmap dataset.

        Args:
            meta_paths: List of paths to _pc{k}_meta.npz files.
            label_dict: Dictionary mapping sequence IDs to labels.
            k: Number of PCA components.
        """
        self.blocks = []
        self.lookup = []
        self.k = k
        self.label_dict = label_dict

        for bidx, meta_path in enumerate(meta_paths):
            meta = np.load(meta_path, allow_pickle=True)
            ids = meta["identifier"]
            shape = tuple(meta["shape"])
            mmap_path = meta_path.replace(f"_pc{self.k}_meta.npz", f"_pc{self.k}.mmap")
            mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
            self.blocks.append((ids, mm))

            # Keep only rows with labels
            for r in range(shape[0]):
                seq_id = ids[r]
                if seq_id in self.label_dict:
                    self.lookup.append((bidx, r))

    def __len__(self) -> int:
        """Return number of labeled samples."""
        return len(self.lookup)

    def __getitem__(self, idx: int) -> tuple:
        """Return (PCA embeddings, label) for sample at index."""
        bidx, ridx = self.lookup[idx]
        ids, mm = self.blocks[bidx]
        seq_id = ids[ridx]
        x = torch.from_numpy(mm[ridx].astype("float32")).t()  # (K, L)
        y = torch.tensor(self.label_dict[seq_id], dtype=torch.float32)
        return x, y


class MeanMemmapMap(Dataset):
    """
    Dataset for mean-pooled embeddings.

    Loads mean-pooled embeddings of shape (1, L) or (L, 1).
    """

    def __init__(self, meta_paths: list, label_dict: dict, embed_type: str = 'mean_seq'):
        """
        Initialize mean memmap dataset.

        Args:
            meta_paths: List of paths to _{embed_type}_meta.npz files.
            label_dict: Dictionary mapping sequence IDs to labels.
            embed_type: Type of embedding ('mean_seq' or 'mean_dim').
        """
        self.blocks = []
        self.lookup = []
        self.embed_type = embed_type
        self.label_dict = label_dict

        for bidx, meta_path in enumerate(meta_paths):
            meta = np.load(meta_path, allow_pickle=True)
            ids = meta["identifier"]
            shape = tuple(meta["shape"])
            mmap_path = meta_path.replace(f"_{embed_type}_meta.npz", f"_{embed_type}.mmap")
            mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
            self.blocks.append((ids, mm))

            # Keep only rows with labels
            for r in range(shape[0]):
                seq_id = ids[r]
                if seq_id in self.label_dict:
                    self.lookup.append((bidx, r))

    def __len__(self) -> int:
        """Return number of labeled samples."""
        return len(self.lookup)

    def __getitem__(self, idx: int) -> tuple:
        """Return (mean embeddings, label) for sample at index."""
        bidx, ridx = self.lookup[idx]
        ids, mm = self.blocks[bidx]
        seq_id = ids[ridx]
        x = torch.from_numpy(mm[ridx].astype("float32"))
        if self.embed_type != 'mean_dim':
            x = x.t()  # Transpose to (1, L)
        y = torch.tensor(self.label_dict[seq_id], dtype=torch.float32)
        return x, y


class MultiGeneConcatDataset(Dataset):
    """
    Concatenates per-gene token embeddings for multiple genes.

    Returns concatenated embeddings of shape (D, L_gene1 + L_gene2 + ...).
    """

    def __init__(self, gene_dirs: list, label_map: dict):
        """
        Initialize multi-gene dataset.

        Args:
            gene_dirs: List of directories containing per-gene memmap files.
            label_map: Dictionary mapping sequence IDs to labels.
        """
        self.gene_dirs = gene_dirs
        self.label_map = label_map
        self.blocks = {}

        for gene_dir in gene_dirs:
            metas = sorted(Path(gene_dir).glob("*_meta.npz"))
            gene = Path(gene_dir).name
            blks = []

            for meta_path in metas:
                meta = np.load(meta_path, allow_pickle=True)
                ids = meta["identifier"].astype(str)
                shape = tuple(meta["shape"])
                mmap_path = str(meta_path).replace("_meta.npz", ".mmap")
                mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
                blks.append((ids, mm))

            self.blocks[gene] = blks

        # Extract common IDs across all genes
        id_sets = [set(id_ for blk in self.blocks[Path(g).name] for id_ in blk[0]) for g in gene_dirs]
        common_ids = set.intersection(*id_sets)

        # Keep only IDs with labels
        valid_ids = set(self.label_map.keys())
        self.ids = sorted(common_ids & valid_ids)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.ids)

    def _row(self, gene: str, ident: str) -> torch.Tensor:
        """Get embedding for a specific gene and sample ID."""
        gene = Path(gene).name
        for ids, mm in self.blocks[gene]:
            w = np.where(ids == ident)[0]
            if w.size:
                return torch.from_numpy(mm[w[0]].astype("float32")).t()
        raise KeyError(f"{ident} missing in {gene}")

    def __getitem__(self, idx: int) -> tuple:
        """Return (concatenated embeddings, label) for sample at index."""
        ident = self.ids[idx]
        parts = [self._row(gene, ident) for gene in self.gene_dirs]
        x = torch.cat(parts, dim=1)  # Concatenate along length dimension
        y = torch.tensor(self.label_map[ident], dtype=torch.float32)
        return x, y


class PcaMultiGeneConcatDataset(Dataset):
    """
    Concatenates PCA-compressed per-gene embeddings for multiple genes.

    Returns concatenated embeddings of shape (K, L_gene1 + L_gene2 + ...).
    """

    def __init__(self, gene_dirs: list, label_map: dict, k: int):
        """
        Initialize PCA multi-gene dataset.

        Args:
            gene_dirs: List of directories containing per-gene PCA memmap files.
            label_map: Dictionary mapping sequence IDs to labels.
            k: Number of PCA components.
        """
        self.gene_dirs = gene_dirs
        self.k = k
        self.label_map = label_map
        self.blocks = {}

        for gene_dir in gene_dirs:
            metas = sorted(Path(gene_dir).glob(f"*_pc{k}_meta.npz"))
            gene = Path(gene_dir).name

            if not metas:
                raise FileNotFoundError(f"No *_pc{k}_meta.npz found for {gene}")

            blks = []
            for meta_path in metas:
                meta = np.load(meta_path, allow_pickle=True)
                ids = meta["identifier"].astype(str)
                shape = tuple(meta["shape"])
                mmap_path = str(meta_path).replace("_meta.npz", ".mmap")
                mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
                blks.append((ids, mm))

            self.blocks[gene] = blks

        # Extract common IDs across all genes
        id_sets = [set(id_ for blk in self.blocks[Path(g).name] for id_ in blk[0]) for g in gene_dirs]
        common_ids = set.intersection(*id_sets)

        # Keep only IDs with labels
        valid_ids = set(self.label_map.keys())
        self.ids = sorted(common_ids & valid_ids)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.ids)

    def _row(self, gene: str, ident: str) -> torch.Tensor:
        """Get PCA embedding for a specific gene and sample ID."""
        gene = Path(gene).name
        for ids, mm in self.blocks[gene]:
            w = np.where(ids == ident)[0]
            if w.size:
                return torch.from_numpy(mm[w[0]].astype("float32")).t()
        raise KeyError(f"{ident} missing in {gene}")

    def __getitem__(self, idx: int) -> tuple:
        """Return (concatenated PCA embeddings, label) for sample at index."""
        ident = self.ids[idx]
        parts = [self._row(gene, ident) for gene in self.gene_dirs]
        x = torch.cat(parts, dim=1)  # Concatenate along length dimension
        y = torch.tensor(self.label_map[ident], dtype=torch.float32)
        return x, y


class MeanMultiGeneConcatDataset(Dataset):
    """
    Concatenates mean-pooled per-gene embeddings for multiple genes.

    Returns concatenated embeddings of shape (1, L_gene1 + L_gene2 + ...).
    """

    def __init__(self, gene_dirs: list, label_map: dict, embed_type: str = 'mean_seq'):
        """
        Initialize mean multi-gene dataset.

        Args:
            gene_dirs: List of directories containing per-gene mean embeddings.
            label_map: Dictionary mapping sequence IDs to labels.
            embed_type: Type of embedding ('mean_seq' or 'mean_dim').
        """
        self.gene_dirs = gene_dirs
        self.label_map = label_map
        self.embed_type = embed_type
        self.blocks = {}

        for gene_dir in gene_dirs:
            metas = sorted(Path(gene_dir).glob(f"*_{embed_type}_meta.npz"))
            gene = Path(gene_dir).name

            if not metas:
                raise FileNotFoundError(f"No *_{embed_type}_meta.npz found for {gene}")

            blks = []
            for meta_path in metas:
                meta = np.load(meta_path, allow_pickle=True)
                ids = meta["identifier"].astype(str)
                shape = tuple(meta["shape"])
                mmap_path = Path(str(meta_path).replace(f"_{embed_type}_meta.npz", f"_{embed_type}.mmap"))
                mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
                blks.append((ids, mm))

            self.blocks[gene] = blks

        # Extract common IDs across all genes
        id_sets = [set(id_ for blk in self.blocks[Path(g).name] for id_ in blk[0]) for g in gene_dirs]
        common_ids = set.intersection(*id_sets)

        # Keep only IDs with labels
        valid_ids = set(self.label_map.keys())
        self.ids = sorted(common_ids & valid_ids)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.ids)

    def _row(self, gene: str, ident: str) -> torch.Tensor:
        """Get mean embedding for a specific gene and sample ID."""
        gene = Path(gene).name
        for ids, mm in self.blocks[gene]:
            w = np.where(ids == ident)[0]
            if w.size:
                x = torch.from_numpy(mm[w[0]].astype("float32"))
                if self.embed_type != 'mean_dim':
                    x = x.t()  # Transpose to (1, L)
                return x
        raise KeyError(f"{ident} missing in {gene}")

    def __getitem__(self, idx: int) -> tuple:
        """Return (concatenated mean embeddings, label) for sample at index."""
        ident = self.ids[idx]
        parts = [self._row(gene, ident) for gene in self.gene_dirs]
        x = torch.cat(parts, dim=1)  # Concatenate along length dimension
        y = torch.tensor(self.label_map[ident], dtype=torch.float32)
        return x, y
