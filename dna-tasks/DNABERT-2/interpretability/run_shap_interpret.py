"""
SHAP-based Interpretability Analysis for DNABERT-2 Resistance Models

This script performs SHAP (SHapley Additive exPlanations) analysis to interpret
trained resistance prediction models. It computes feature importance for genomic
positions and maps them to biological mutations for causal variant discovery.

Key capabilities:
- Compute SHAP values for specified drugs
- Rank genomic positions by importance
- Map token positions to original sequences
- Identify confident mutation hits
- Integrate with VCF/WHO mapping data

Author: Saishradha Mohanty
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import transformers
from pathlib import Path

from utils.interpret_utils import get_important_features_list, map_token_positions_to_original_sequence
from utils.shap_utils import compute_shap_for_drug
from utils.map_mar import get_confident_mutation_hits


# Drug to gene loci mapping for resistance mechanisms
DRUG_TO_LOCI = {
    "ISONIAZID": ["inhA", "katG"],
    "RIFAMPICIN": ["rpoB", "rpoC"],
    "ETHAMBUTOL": ["embC", "embA", "embB"],
    "PYRAZINAMIDE": ["pncA"],
    "STREPTOMYCIN": ["rpsL", "rrs", "gid"],
    "AMIKACIN": ["rrs", "eis"],
    "KANAMYCIN": ["rrs"],
    "CAPREOMYCIN": ["rrs", "rrl", "tlyA"],
    "LEVOFLOXACIN": ["gyrB", "gyrA"],
    "MOXIFLOXACIN": ["gyrB", "gyrA"],
    "ETHIONAMIDE": ["inhA", "ethA", "ethR"],
}


def rank_positions_by_shap(shap_df, column_name):
    """
    Rank genomic positions by maximum absolute SHAP value across isolates.

    Args:
        shap_df (pd.DataFrame): DataFrame with SHAP values per position.
        column_name (str): Column name containing SHAP importance arrays.

    Returns:
        pd.DataFrame: DataFrame ranked by MaxAbsSHAP descending.
                     Columns: Token_Position, MaxAbsSHAP
    """
    # Stack SHAP values across isolates: (num_isolates, sequence_length)
    shap_values = np.stack(
        [np.asarray(v).squeeze() for v in shap_df[column_name]], axis=0
    )

    # Compute maximum absolute SHAP per position
    max_abs_shap = np.abs(shap_values).max(axis=0)

    # Create ranking dataframe
    ranking_df = pd.DataFrame(
        {"Token_Position": np.arange(len(max_abs_shap)), "MaxAbsSHAP": max_abs_shap}
    )
    ranking_df = ranking_df.sort_values("MaxAbsSHAP", ascending=False).reset_index(
        drop=True
    )

    return ranking_df


def rank_all_genes_for_drug(shap_dir, drug, loci, in_dim, top_n=100):
    """
    Rank and merge SHAP importance across all genes for a drug.

    Args:
        shap_dir (Path): Directory containing SHAP results.
        drug (str): Drug name.
        loci (list): List of gene names for this drug.
        in_dim (int): Embedding dimension (for file naming).
        top_n (int): Number of top positions to select.

    Returns:
        pd.DataFrame: Top-ranked positions across all genes with feature names.
    """
    gene_level_dfs = []

    for gene in loci:
        ranked_shap_path = shap_dir / f"{drug}_{gene}_ranked_SHAP.csv"

        if not ranked_shap_path.exists():
            print(f"  [skip] {drug} - {gene}: ranked SHAP file missing.")
            continue

        # Load and tag with gene name
        shap_df = pd.read_csv(ranked_shap_path)
        shap_df["Gene"] = gene
        gene_level_dfs.append(shap_df)

    if not gene_level_dfs:
        print(f"  [skip] {drug}: no SHAP files found for any loci.")
        return None

    # Merge all genes and select top positions
    merged_df = pd.concat(gene_level_dfs, ignore_index=True)
    merged_df = merged_df.sort_values("MaxAbsSHAP", ascending=False)
    top_df = merged_df.head(top_n).copy()

    # Create feature names as "gene_position"
    top_df["Feature_Name"] = [
        f"{gene}_{pos}"
        for gene, pos in zip(top_df["Gene"], top_df["Token_Position"])
    ]

    return top_df


def load_tokenizer(model_name, model_max_length=5000):
    """
    Load DNABERT-2 tokenizer.

    Args:
        model_name (str): Model identifier either from local pretrained model or Huggingface.
        model_max_length (int): Maximum sequence length for tokenization.

    Returns:
        transformers.PreTrainedTokenizer: Loaded tokenizer.
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    print("  Tokenizer loaded successfully!")
    return tokenizer


def main(args):
    """
    Main interpretability pipeline for SHAP-based analysis.

    Orchestrates the workflow:
    1. Compute SHAP values for specified drug(s)
    2. Rank genomic positions by importance
    3. Map token positions to original sequences
    4. Identify confident mutation hits
    5. Generate interpretability reports

    Args:
        args: Command-line arguments with configuration.
    """
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"GPUs available: {n_gpu}\n")

    print("=" * 70)
    print("SHAP-based Interpretability Analysis for Resistance Models")
    print("=" * 70)
    print(f"Drug: {args.drug}")
    print(f"Embedding type: {args.embed_type}")
    print(f"Embedding dimension: {args.in_dim}")
    print(f"Output directory: {args.output_path}")
    print("=" * 70)

    # Setup paths
    torch.cuda.empty_cache()

    shap_dir = Path(args.output_path) / f"{args.embed_type}_{args.in_dim}"
    shap_dir.mkdir(parents=True, exist_ok=True)
    shap_path = shap_dir / f"{args.drug}_dim{args.in_dim}_shap_all.pkl"

    # -----------------------------------------------
    # Step 1: Compute SHAP values
    # -----------------------------------------------
    print(f"\nStep 1: Computing SHAP values for {args.drug}...")

    if not shap_path.exists():
        print(f"  Computing SHAP and saving to {shap_path}")
        _, gene_feature_labels = compute_shap_for_drug(
            drug=args.drug,
            embed_type=args.embed_type,
            in_dim=args.in_dim,
            background_frac=args.background_frac,
            explain_frac=args.explain_frac,
            out_dir=args.output_path,
            memmap_dir=args.memmap_dir,
            model_dir=args.model_dir,
            pheno_label_path=args.pheno_label_path,
            random_seed=args.random_seed,
        )
    else:
        print(f"  SHAP values already computed at {shap_path}")

    # -----------------------------------------------
    # Step 2: Rank positions by SHAP importance
    # -----------------------------------------------
    print(f"\nStep 2: Ranking genomic positions by SHAP importance...")

    loci = DRUG_TO_LOCI.get(args.drug, [])
    if not loci:
        raise ValueError(f"Drug {args.drug} not found in DRUG_TO_LOCI mapping")

    shap_df = pd.read_pickle(shap_path)

    for gene in loci:
        col_name = f"importance_{gene}" if len(loci) > 1 else "importance_full"

        if col_name not in shap_df.columns:
            print(f"  [warn] Column {col_name} not found for {gene}")
            continue

        ranked_df = rank_positions_by_shap(shap_df, col_name)
        out_path = shap_dir / f"{args.drug}_{gene}_ranked_SHAP.csv"
        ranked_df.to_csv(out_path, index=False)
        print(f"  Saved ranked positions for {gene} to {out_path}")

    # -----------------------------------------------
    # Step 3: Get top important positions across genes
    # -----------------------------------------------
    print(f"\nStep 3: Merging and selecting top important positions...")

    top_positions_df = rank_all_genes_for_drug(
        shap_dir,
        args.drug,
        loci,
        args.in_dim,
        top_n=args.top_n_positions,
    )

    if top_positions_df is not None:
        top_pos_path = shap_dir / f"{args.drug}_top{args.top_n_positions}_pos.csv"
        top_positions_df.to_csv(top_pos_path, index=False)
        print(f"  Saved top {args.top_n_positions} positions to {top_pos_path}")
    else:
        print(f"  [warn] Could not rank positions for {args.drug}")
        return

    # -----------------------------------------------
    # Step 4: Map token positions to original sequences
    # -----------------------------------------------
    print(f"\nStep 4: Mapping token positions to original sequences...")

    tokenizer = load_tokenizer(
        model_name=args.dnabert_model,
        model_max_length=args.dnabert_model_max_len,
    )

    token_seq_mapped_path = shap_dir / f"{args.drug}_mapped_top{args.top_n_positions}_pos.csv"
    mapped_df = map_token_positions_to_original_sequence(
        tokenizer,
        top_positions_df,
        ref_seq_json_path=args.ref_seq_json_path,
    )
    mapped_df.to_csv(token_seq_mapped_path, index=False)
    print(f"  Saved mapped positions to {token_seq_mapped_path}")

    # -----------------------------------------------
    # Step 5: Extract important features
    # -----------------------------------------------
    print(f"\nStep 5: Extracting important features...")

    important_features = get_important_features_list(mapped_df)
    print(f"  Found {len(important_features)} important features")
    print(f"  Features: {important_features[:5]}..." if len(important_features) > 5 else f"  Features: {important_features}")

    # -----------------------------------------------
    # Step 6: Map to confident mutation hits
    # -----------------------------------------------
    print(f"\nStep 6: Mapping to confident mutation hits...")

    get_confident_mutation_hits(
        vcf_who_map_directory=args.vcf_who_map_dir,
        important_features=important_features,
        drug=args.drug,
        output_path=str(shap_dir),
        output_file_name=f"map_mar_{args.drug}.csv",
        embed_type="tokens",
        has_neg_strand=args.has_neg_strand,
    )
    print(f"  Confident mutation hits saved to {shap_dir}/map_mar_{args.drug}.csv")

    print("\n" + "=" * 70)
    print("Interpretability analysis completed successfully!")
    print(f"Results saved to: {shap_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform SHAP-based interpretability analysis for resistance models"
    )

    # Model and tokenizer configuration
    parser.add_argument(
        "--dnabert_model",
        type=str,
        default="pretrained_models/DNABERT2",
        help="HuggingFace model identifier for DNABERT-S",
    )
    parser.add_argument(
        "--dnabert_model_max_len",
        type=int,
        default=5000,
        help="Maximum sequence length for tokenization",
    )

    # Data paths
    parser.add_argument(
        "--memmap_dir",
        type=str,
        default="/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps",
        help="Directory containing memory-mapped embedding files",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/saved_models/dnabert2/token_embeds",
        help="Directory containing trained downstream models",
    )
    parser.add_argument(
        "--pheno_label_path",
        type=str,
        default="/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/zs_full_stacked_phenotypes.npz",
        help="Path to phenotype labels NPZ file",
    )
    parser.add_argument(
        "--ref_seq_json_path",
        type=str,
        default="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/interpretability/dataloader/ref_gene_seq.json",
        help="Path to reference gene sequences JSON file. If the refererence sequences are different, update this path accordingly or generate a new JSON file.",
    )
    parser.add_argument(
        "--vcf_who_map_dir",
        type=str,
        default="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/vcf_who_mapped_data",
        help="Directory containing VCF WHO mapping data",
    )

    # Drug and embedding configuration
    parser.add_argument(
        "--drug",
        type=str,
        default="RIFAMPICIN",
        choices=list(DRUG_TO_LOCI.keys()),
        help="Drug to compute SHAP importance for",
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        default="token",
        choices=["token", "mean", "pca"],
        help="Type of embeddings used",
    )
    parser.add_argument(
        "--in_dim",
        type=int,
        default=768,
        help="Embedding dimension (768 for token, custom for PCA)",
    )

    # SHAP computation parameters
    parser.add_argument(
        "--background_frac",
        type=float,
        default=0.1,
        help="Fraction of data to use as background for SHAP",
    )
    parser.add_argument(
        "--explain_frac",
        type=float,
        default=1.0,
        help="Fraction of data to explain with SHAP",
    )
    parser.add_argument(
        "--top_n_positions",
        type=int,
        default=100,
        help="Number of top important positions to select",
    )

    # Output paths
    parser.add_argument(
        "--output_path",
        type=str,
        default="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT-2/interpretability/output/shap_results",
        help="Directory to save SHAP analysis results",
    )

    # Mutation mapping options
    parser.add_argument(
        "--has_neg_strand",
        type=bool,
        default=True,
        help="Whether data includes negative strand mutations",
    )

    # Reproducibility
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    main(args)
