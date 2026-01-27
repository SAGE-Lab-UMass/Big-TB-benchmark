"""
SHAP Value Computation for DNABERT CNN Models

This module computes SHapley Additive exPlanations (SHAP) values for trained
CNN models that use DNABERT embeddings. It supports both single-gene and
multi-gene scenarios with memory-efficient batched computation.
Key Features:
    - Token-level importance scoring for genomic sequences
    - Batched computation for memory efficiency
    - Multi-gene support with per-gene importance tracking
    - Automatic sample deduplication with caching
    - Flexible embedding types (token, PCA-reduced, mean-pooled)

Example:
    >>> shap_df, labels = compute_shap_for_drug(
    ...     drug="RIFAMPICIN",
    ...     embed_type="pca",
    ...     in_dim=128
    ... )"""

import os
import glob
import math
import random
import gc
import torch
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from tqdm import tqdm

from dataloader.dataloader import *
from dataloader.locus_order import *
from utils.model_utils import *


device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Drug-to-gene mappings ────────────────────────────────────────────

single_drugs = {
    "PYRAZINAMIDE": ["pncA"],
    "KANAMYCIN": ["rrs"],
}

multi_drugs = {
    "RIFAMPICIN": ["rpoB", "rpoC"],
    "CAPREOMYCIN": ["tlyA", "rrs", "rrl"],
    "STREPTOMYCIN": ["rpsL", "rrs", "gid"],
    "ISONIAZID": ["katG", "inhA"],
    "ETHIONAMIDE": ["ethA", "ethR"],
    "AMIKACIN": ["rrs", "eis"],
    "MOXIFLOXACIN": ["gyrB", "gyrA"],
    "LEVOFLOXACIN": ["gyrB", "gyrA"],
    "ETHAMBUTOL": ["embC", "embA", "embB"],
}

all_drugs = {**single_drugs, **multi_drugs}

# ─── SHAP model wrapper ───────────────────────────────────────────────

class Wrapped(torch.nn.Module):
    """Wraps a model to ensure SHAP compatibility.
    
    SHAP's DeepExplainer expects output shape (B, 1), but models may output (B,).
    This wrapper unsqueezes the output to ensure correct dimensionality.
    
    Args:
        base (torch.nn.Module): Base model to wrap.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
    
    def forward(self, x):
        """Forward pass with output reshaping.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, *)
            
        Returns:
            torch.Tensor: Output of shape (B, 1)
        """
        return self.base(x).unsqueeze(1)

# ─── Deduplication utilities ──────────────────────────────────────

def dedup_and_save_indices(ds, name, out_dir="dedup_geno_data"):
    """Deduplicate dataset samples and cache results.
    
    Removes duplicate (sample, label) pairs using hash-based deduplication.
    Results are cached to disk for efficient reuse across runs. On subsequent
    calls with the same `name`, loads cached indices instead of recomputing.
    
    Args:
        ds: PyTorch dataset with __len__ and __getitem__ methods.
        name (str): Identifier for caching (e.g., "RIFAMPICIN_train").
        out_dir (str): Directory for storing deduplication indices.
            Default: "dedup_geno_data"
    
    Returns:
        list: Indices of unique samples (subset of range(len(ds))).
    
    Writes:
        - {out_dir}/{name}_dedup_indices.npy: Cached indices array
        - {out_dir}/dedup_log.txt: Log of deduplication statistics
    """
    print(f"[{name}] Deduplicating ...")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_dedup_indices.npy"

    # Load cached indices if available
    if out_path.exists():
        uniq_indices = np.load(out_path).tolist()
        print(f"[{name}] Loaded cached indices ({len(ds)} → {len(uniq_indices)})")
        return uniq_indices

    # Compute deduplication fresh
    uniq_indices, seen = [], set()
    for i in tqdm(range(len(ds)), desc=f"[{name}]"):
        x, y = ds[i]
        key = (x.numpy().tobytes(), int(y))
        if key not in seen:
            seen.add(key)
            uniq_indices.append(i)

    np.save(out_path, uniq_indices)
    reduction = len(ds) - len(uniq_indices)
    frac = 100.0 * reduction / len(ds)
    log_msg = (f"[{name}] Deduplicated {len(ds)} → {len(uniq_indices)} "
               f"({reduction} removed, {frac:.1f}% reduction)")
    print(log_msg)
    with open(out_dir / "dedup_log.txt", "a") as f:
        f.write(log_msg + "\n")

    return uniq_indices

# ─── SHAP core computation ────────────────────────────────────────────

def shap_per_residue(model, background_ds, full_ds,
                     background_size, explain_samples,
                     per_gene_lengths=None, gene_names=None,
                     device="cuda"):
    """Compute token-level SHAP values for CNN model predictions.
    
    Uses SHAP DeepExplainer to compute attributions at each token position.
    Processes samples in batches for memory efficiency and tracks per-gene
    importance when multi-gene models are used.
    
    Args:
        model (torch.nn.Module): Trained CNN model (should already be on device).
        background_ds: PyTorch dataset for background/reference samples.
        full_ds: PyTorch dataset for all samples (used to select validation set).
        background_size (int): Number of background samples to use.
        explain_samples (int): Number of samples to explain.
        per_gene_lengths (list, optional): Token lengths for each gene.
            If provided, importance is split by gene. Default: None
        gene_names (list, optional): Gene identifiers corresponding to 
            per_gene_lengths. Required if per_gene_lengths is provided.
        device (str): Compute device ("cuda" or "cpu"). Default: "cuda"
    
    Returns:
        pd.DataFrame: SHAP results with columns:
            - sample_idx: Indices of explained samples
            - label: True labels (0 or 1)
            - importance_full: SHAP importance per token (all genes combined)
            - importance_{gene_name}: SHAP importance per token (per gene)
    
    Note:
        Assumes datasets are already deduplicated.
    """
    model = model.to(device).eval()

    # Select background samples
    B = min(background_size, len(background_ds))
    print(f"Using background size {B} from {len(background_ds)} training samples")
    bg_idx = random.sample(range(len(background_ds)), B)
    background = torch.stack([background_ds[i][0] for i in bg_idx]).to(device)

    # Create validation set (full_ds minus background)
    all_idx = set(range(len(full_ds)))
    val_idx = list(all_idx - set(bg_idx))
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    explainer = shap.DeepExplainer(Wrapped(model), [background])

    # Select samples to explain
    E = min(explain_samples, len(val_ds))
    print(f"Explaining {E} samples from {len(val_ds)} validation samples")
    samp_idx = random.sample(range(len(val_ds)), E)
    xs = torch.stack([val_ds[i][0] for i in samp_idx]).to(device)
    ys = [val_ds[i][1] for i in samp_idx]

    # Compute SHAP values in batches
    start_time = time()
    svs = []
    for chunk in torch.split(xs, 4):
        sv_chunk = explainer.shap_values([chunk], check_additivity=False)[0]
        svs.append(sv_chunk)
        del sv_chunk
        torch.cuda.empty_cache()
        gc.collect()
    
    sv = np.concatenate(svs, axis=0)
    imp = np.abs(sv).sum(axis=1)
    
    elapsed = time() - start_time
    print(f"SHAP computation completed in {elapsed:.2f} seconds")

    out = {
        "sample_idx": samp_idx,
        "label": [int(y) for y in ys],
        "importance_full": list(imp)
    }

    if per_gene_lengths is not None:
        cuts = np.cumsum([0] + per_gene_lengths)
        for gi, g in enumerate(gene_names):
            out[f"importance_{g}"] = [imp[n, cuts[gi]:cuts[gi+1]] for n in range(E)]

    return pd.DataFrame(out)


# ─── Model loading utilities ──────────────────────────────────────────

def embeddings_root(gene, embed_dir):
    """Return the path to the embedding directory for a given gene."""
    return Path(embed_dir) / gene


def load_model(drug, gene, embed_type, in_dim, model_dir, embed_dir,
               model_name="DNABERTCNN", model_seq_len=5000, random_seed=42):
    """Load trained drug resistance prediction model.
    
    Handles both single-gene and multi-gene scenarios. Automatically
    determines gene sequence lengths from metadata files.
    
    Args:
        drug (str): Drug name (e.g., "RIFAMPICIN", "ISONIAZID")
        gene (str): Gene name for single-gene case.
        embed_type (str): Embedding type ("token", "pca", or "mean").
        in_dim (int): Input embedding dimension.
        model_dir (str): Root directory containing trained models.
        embed_dir (str): Root directory containing embedding metadata.
        model_name (str): Model class name. Default: "DNABERTCNN"
        model_seq_len (int): Sequence length for model. Default: 5000 for the SD cases
        random_seed (int): Random seed used during training. Default: 42
    
    Returns:
        tuple: (model, per_gene_lengths, gene_names, total_length)
            - model (torch.nn.Module): Loaded model in eval mode
            - per_gene_lengths (list): Token lengths for each gene
            - gene_names (list): Gene names in order
            - total_length (int): Sum of all gene lengths
    
    Raises:
        FileNotFoundError: If metadata or model files not found.
    """
    if drug in multi_drugs:
        genes = multi_drugs[drug]
        per_gene_len, gene_names = [], []
        for g in genes:
            mp = next(Path(embeddings_root(g, embed_dir)).glob("*_meta.npz"))
            Lg = int(np.load(mp, allow_pickle=True)["shape"][1])
            per_gene_len.append(Lg)
            gene_names.append(g)
        L_PAD = sum(per_gene_len)
    else:
        mp = next(Path(embeddings_root(gene, embed_dir)).glob("*_meta.npz"))
        L_PAD = int(np.load(mp, allow_pickle=True)["shape"][1])
        per_gene_len = [L_PAD]
        gene_names = [gene]

    model = get_model_class(model_name=model_name, in_dim=in_dim, 
                           seq_len=model_seq_len, device=device)

    seed_path = os.path.join(model_dir, f"{drug}/seed_{random_seed}/fold_1")
    model_path = os.path.join(seed_path, f"{model_name}.pt")

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, per_gene_len, gene_names, L_PAD

# ─── Main SHAP computation driver ─────────────────────────────────────

def compute_shap_for_drug(drug, embed_type="token", in_dim=768,
                          background_frac=0.1, explain_frac=1.0,
                          out_dir="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/interpretability/output/shap_results",
                          memmap_dir="/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps",
                          model_dir="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_models/dnabert2/token_embeds",
                          pheno_label_path="/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz",
                          random_seed=42):
    """Compute SHAP values for a drug resistance model.
    
    Main entry point for complete SHAP analysis workflow:
    1. Load embeddings and phenotype labels
    2. Train/test split
    3. Remove duplicate samples with caching
    4. Load trained model
    5. Compute token-level SHAP values
    6. Save results and generate feature labels
    
    Args:
        drug (str): Drug name (must be in SINGLE_DRUG_GENES or MULTI_DRUG_GENES).
        embed_type (str): Type of embeddings to use.
            Options: "token" (768D), "pca" (in_dim D), "mean" (pooled).
            Default: "token"
        in_dim (int): Input embedding dimension for PCA. Default: 768
        background_frac (float): Fraction of data to use for SHAP background.
            Default: 0.1
        explain_frac (float): Fraction of data to explain. Default: 1.0
        out_dir (str): Output directory for results.
        memmap_dir (str): Directory with memory-mapped embeddings.
        model_dir (str): Directory with trained models.
        pheno_label_path (str): Path to phenotype label NPZ file.
        random_seed (int): Random seed for reproducibility. Default: 42
    
    Returns:
        tuple: (shap_dataframe, gene_feature_labels)
            - shap_dataframe (pd.DataFrame): SHAP values and metadata
            - gene_feature_labels (dict): Maps gene names to feature labels
    
    Creates:
        {out_dir}/{embed_type}_{in_dim}/{drug}_dim{in_dim}_shap_all.pkl
    """
    gene = single_drugs.get(drug, [None])[0] if drug in single_drugs else None

    def make_dataset(data_prefix="train"):
        """Create dataset with appropriate embedding type."""
        label_map, _ = build_label_map(pheno_label_path, drug, prefix=data_prefix)
        if drug in multi_drugs:
            genes = multi_drugs[drug]
            gene_memmap_dirs = [f"{memmap_dir}/{g}/" for g in genes]

            if embed_type == "token":
                return MultiGeneConcatDataset(gene_memmap_dirs, label_map)
            elif embed_type == "pca":
                return PcaMultiGeneConcatDataset(gene_memmap_dirs, label_map, 
                                               n_components=in_dim)
            elif embed_type == "mean":
                return MeanMultiGeneConcatDataset(gene_memmap_dirs, label_map, 
                                                embed_type=embed_type)
        else:
            meta_paths = sorted(glob.glob(f"{memmap_dir}/{gene}/*_meta.npz"))
            print(f"Found {len(meta_paths)} meta files")

            if embed_type == "token":
                return TokenMemmapMap(meta_paths, label_map)
            elif embed_type == "pca":
                return PcaMemmapMap(meta_paths, label_map, n_components=in_dim)
            elif embed_type == "mean":
                return MeanMemmapMap(meta_paths, label_map, embed_type=in_dim)

    # Load full dataset
    full_ds = make_dataset(data_prefix="full")
    embeds, _ = full_ds[0]
    model_dim = embeds.shape[0]
    model_seq_len = embeds.shape[1]
    print(f"[{drug}] Full dataset: {len(full_ds)} samples, {model_dim} features")

    # Split into train and test
    print(f"Splitting data into train and test sets")
    train_ds, test_ds = get_train_test_split(full_ds, drug, test_size=0.2)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Deduplicate
    train_idx = dedup_and_save_indices(train_ds, f"{drug}_train")
    test_idx = dedup_and_save_indices(test_ds, f"{drug}_test")
    full_idx = dedup_and_save_indices(full_ds, f"{drug}_full")
    
    train_ds = torch.utils.data.Subset(train_ds, train_idx)
    test_ds = torch.utils.data.Subset(test_ds, test_idx)
    full_ds = torch.utils.data.Subset(full_ds, full_idx)

    out_path = Path(out_dir) / f"{embed_type}_{in_dim}"
    out_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model for {drug} ...")
    model, per_gene_len, gene_names, L_PAD = load_model(
        drug, gene, embed_type, in_dim, model_dir, memmap_dir,
        model_seq_len=model_seq_len, random_seed=random_seed)
    print(f"Model loaded. Genes: {gene_names}, Lengths: {per_gene_len}")

    # Compute SHAP
    bg_size_all = min(160, int(len(full_ds) * background_frac))
    ex_size_all = min(360, len(full_ds) - bg_size_all)
    
    shap_df_all = shap_per_residue(model, full_ds, full_ds,
                                   background_size=bg_size_all, 
                                   explain_samples=ex_size_all,
                                   per_gene_lengths=per_gene_len, 
                                   gene_names=gene_names, device=device)

    # Save results
    shap_df_all.to_pickle(out_path / f"{drug}_dim{in_dim}_shap_all.pkl", protocol=4)
    print(f"[done] {drug}: background={bg_size_all}, explain={ex_size_all} samples")

    # Map feature labels
    gene_feature_labels = {
        gene: [f"{gene}_{pos}" for pos in range(length)]
        for gene, length in zip(gene_names, per_gene_len)
    }
    return shap_df_all, gene_feature_labels


def get_train_test_split(full_ds, drug, test_size=0.2, seed=42):
    """Split dataset into train and test subsets with deterministic seeding.
    
    Args:
        full_ds: PyTorch dataset to split.
        drug (str): Drug name (for logging/clarity).
        test_size (float): Fraction of data for test set (0.0-1.0).
            Default: 0.2
        seed (int): Random seed for reproducibility. Default: 42
    
    Returns:
        tuple: (train_subset, test_subset) as torch.utils.data.Subset objects.
    """
    train_size = 1 - test_size
    train_len = math.floor(train_size * len(full_ds))
    test_len = len(full_ds) - train_len
    train_ds, test_ds = torch.utils.data.random_split(
        full_ds, [train_len, test_len], 
        generator=torch.Generator().manual_seed(seed)
    )
    return train_ds, test_ds