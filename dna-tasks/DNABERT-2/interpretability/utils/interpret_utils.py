"""
Integrated Gradients Attribution for CNN with DNABERT (Batched, Memory-Efficient)

This module provides utilities for computing token-level attributions using
Integrated Gradients for DNA sequence analysis with DNABERT and CNN models.
"""

import json
from typing import Callable, List

import pandas as pd


def map_token_positions_to_original_sequence(
    tokenizer: Callable,
    top_token_pos_df: pd.DataFrame,
    ref_seq_json_path: str = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/interpretability/dataloader/ref_gene_seq.json"
) -> pd.DataFrame:
    """
    Map token positions to their corresponding nucleotide positions in original sequences.

    Args:
        tokenizer: Tokenizer with offset mapping capability.
        top_token_pos_df: DataFrame with token positions and genes.
        ref_seq_json_path: Path to JSON file containing reference gene sequences.

    Returns:
        DataFrame with added columns:
        - Gene_nt_start: Nucleotide start position
        - Gene_nt_end: Nucleotide end position (inclusive)
        - Gene_nt_seq: Nucleotide sequence span
    """
    # Load reference sequences
    with open(ref_seq_json_path) as f:
        gene_seqs_dict = json.load(f)

    nt_start_list = []
    nt_end_list = []
    nt_spans = []

    for _, row in top_token_pos_df.iterrows():
        gene = row['Gene']
        token_pos = row['Token_Position']
        sequence = gene_seqs_dict[gene]

        # Tokenize with offset mapping
        encoding = tokenizer(
            sequence,
            return_offsets_mapping=True,
            truncation=True,
            return_tensors='pt'
        )

        offsets = encoding['offset_mapping'][0]

        # Map token position to nucleotide span
        if token_pos < len(offsets):
            start, end = offsets[token_pos].tolist()
            nt_start_list.append(start)
            nt_end_list.append(end - 1)  # inclusive end
            nt_spans.append(sequence[start:end])
        else:
            nt_start_list.append(None)
            nt_end_list.append(None)
            nt_spans.append("")

    # Add mapped positions to DataFrame
    top_token_pos_df["Gene_nt_start"] = nt_start_list
    top_token_pos_df["Gene_nt_end"] = nt_end_list
    top_token_pos_df["Gene_nt_seq"] = nt_spans

    return top_token_pos_df


def get_important_features_list(important_tokens_df: pd.DataFrame) -> List[tuple]:
    """
    Extract important features as a list of tuples.

    Each feature is represented as (gene_name, nt_start, nt_end) where -1 indicates
    missing values.

    Args:
        important_tokens_df: DataFrame with columns:
            - 'Gene': Gene name
            - 'Gene_nt_start': Nucleotide start position
            - 'Gene_nt_end': Nucleotide end position

    Returns:
        List of tuples: [(gene_name, start_pos, end_pos), ...]
    """
    important_features = [
        (
            row["Gene"],
            int(row["Gene_nt_start"]) if not pd.isna(row["Gene_nt_start"]) else -1,
            int(row["Gene_nt_end"]) if not pd.isna(row["Gene_nt_end"]) else -1
        )
        for _, row in important_tokens_df.iterrows()
    ]

    return important_features
