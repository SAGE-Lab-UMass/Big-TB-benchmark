# Integrated Gradients Attribution for CNN with DNABERT (Batched, Memory-Efficient)
import numpy as np
import pandas as pd
import torch
from torch import nn
from captum.attr import IntegratedGradients
from torch.amp import autocast
from tqdm import tqdm
import json
from typing import List, Callable

from utils.data_utils import *
from dataloader.locus_order import locus_order, DRUGS
import ipdb


def stack_gene_embeddings(gene_index: int, gene_embedding: torch.Tensor, fixed_embeddings: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([
        gene_embedding if i == gene_index else fixed_embeddings[i].repeat_interleave(
            gene_embedding.shape[0] // fixed_embeddings[i].shape[0], dim=0)
        for i in range(len(fixed_embeddings))
    ], dim=1)  # shape: (batch_size, num_genes, hidden_dim)

def downstream_model_forward_wrapper(
    downstream_model: nn.Module,
    gene_index: int,
    fixed_embeddings_list: List[torch.Tensor],
    attention_mask_list: torch.Tensor,
    drug_index: int,
    device: str = 'cuda'
) -> Callable:

    def forward(gene_token_embeddings: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask_list[gene_index].unsqueeze(-1).to(fixed_embeddings_list[0].device)  # (batch_size, seq_len, 1)

        # captum interpolates the gradients of the input with respect to the output, so we need to match the size
        repeat_factor = gene_token_embeddings.shape[0] // mask_expanded.shape[0]
        if repeat_factor > 1:
            mask_expanded = mask_expanded.repeat_interleave(repeat_factor, dim=0)

        gene_mean_pooled_embeddings = (gene_token_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        downstream_model_input = stack_gene_embeddings(gene_index, gene_mean_pooled_embeddings, fixed_embeddings_list)

        cnn_input = downstream_model_input.permute(0, 2, 1)  # (batch_size, hidden_dim, num_genes) for cnn
        return downstream_model(cnn_input)[:, drug_index]

    return forward

def compute_ig_for_cnn_batched(
    dnabert: nn.Module,
    downstream_model: nn.Module,
    input_ids_list: List[torch.Tensor],     # List of (num_gene, batch_size, seq_len)
    attention_mask_list: List[torch.Tensor],# List of (num_gene, batch_size, seq_len)
    max_seq_len: int,
    drug_index: int,
    batch_size: int = 1,
    device: str = 'cuda'
) -> List[List[torch.Tensor]]:
    """
    Computes token-level IG attributions for each gene for each sample in batch.

    Returns:
        List of length batch_size, each element is a list of token scores per gene.
    """
    dnabert.eval()
    downstream_model.eval()
    num_genes = len(input_ids_list)
    batch_token_scores = [[] for _ in range(batch_size)]
    global_token_importance = torch.zeros((num_genes, max_seq_len))  # shape: (num_genes, max_seq_len)
    total_samples_seen = 0

    # Step 1: Precompute pooled embeddings for all genes
    mean_pooled_embeddings_list = []
    with torch.no_grad():
        for i in range(num_genes):
            input_ids = input_ids_list[i].to(device)
            attention_mask = attention_mask_list[i].to(device) 
            
            # Mixed precision for memory efficiency
            with autocast(device_type=device):
                token_embeddings = dnabert(input_ids=input_ids, attention_mask=attention_mask)[0].detach()

            del input_ids

            mask_expanded = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            del attention_mask

            masked_tokens = token_embeddings * mask_expanded  # shape: (batch_size, seq_len, hidden_dim)
            mean_pooled_embeddings = masked_tokens.sum(dim=1) / mask_expanded.sum(dim=1) # shape: (batch_size, hidden_dim)
            mean_pooled_embeddings_list.append(mean_pooled_embeddings)  # shape: num_genes, (batch_size, hidden_dim)

    # clear memory
    torch.cuda.empty_cache()

    # Loop through genes and compute IG for one gene at a time
    for i in range(num_genes):
        with torch.no_grad():
            input_ids = input_ids_list[i].to(device)
            attention_mask = attention_mask_list[i].to(device) 

            gene_token_embeddings = dnabert(input_ids=input_ids_list[i].to(device), attention_mask=attention_mask_list[i].to(device))[0].detach()  # shape: (batch_size, seq_len, hidden_dim)

        del input_ids, attention_mask

        gene_token_embeddings.requires_grad_()

        forward_fn = downstream_model_forward_wrapper(
            downstream_model=downstream_model,
            gene_index=i,
            fixed_embeddings_list=mean_pooled_embeddings_list,
            attention_mask_list=attention_mask_list,
            drug_index=drug_index
        )
        
        ig = IntegratedGradients(forward_fn)
        gene_attributions, _ = ig.attribute(
            inputs=gene_token_embeddings,
            baselines=torch.zeros_like(gene_token_embeddings),
            return_convergence_delta=True
        )
        

        scores = gene_attributions.norm(dim=-1).detach().cpu()  # (batch_size, seq_len)

        for b in range(scores.shape[0]):
            batch_token_scores[b].append(scores[b])  # shape: [batch_size][num_genes][seq_len]

        # Track global importance and total samples
        global_token_importance[i] += scores.abs().sum(dim=0)  # sum across batch
        total_samples_seen += scores.shape[0]

        del gene_token_embeddings, gene_attributions, scores, ig
        torch.cuda.empty_cache()

    # Normalize global importance by total samples seen
    global_token_importance /= total_samples_seen

    # print("Shape summary:")
    # print(f"batch size: {len(batch_token_scores)}")
    # print(f"genes per sample: {len(batch_token_scores[0])}")
    # print(f"tokens per gene: {batch_token_scores[0][0].shape}")
    return batch_token_scores, global_token_importance


def calculate_token_attributions(dataloader, dnabert_s, downstream_model, drug_index, tokenizer, max_seq_len):
    # Initialize list to collect per-batch token attributions
    all_token_attributions = []  # list of (batch_size, num_genes, seq_len)

    for j, batch in enumerate(tqdm(dataloader, desc=f"Drug {drug_index} - Computing IG")):
        input_ids, attention_mask, _ = prepare_multigene_input_fast(batch, tokenizer, max_seq_len)

        batch_token_attributions, global_token_importance = compute_ig_for_cnn_batched(
            dnabert=dnabert_s,
            downstream_model=downstream_model,
            input_ids_list=input_ids,            # list of num_genes, (batch_size, seq_len)
            attention_mask_list=attention_mask,  # list of num_genes, (batch_size, seq_len)
            max_seq_len=max_seq_len,
            drug_index=drug_index,
            batch_size=len(input_ids[0]),
            device='cuda'
        )

        # Convert batch_token_attributions (list of lists) → tensor: (batch_size, num_genes, seq_len)
        batch_tensor = torch.stack([
            torch.stack(sample, dim=0)  
            for sample in batch_token_attributions  # for each sample in batch
        ], dim=0)  

        # Accumulate across batches
        all_token_attributions.append(batch_tensor)

        # for testing
        if j == 2:
            print(f"Batch {j}, breaking for testing...")
            return all_token_attributions, global_token_importance


def get_global_token_importance(global_token_importance, top_k = 100):
    _, seq_len = global_token_importance.shape

    # Flatten and extract top k positions
    flat_importance = global_token_importance.flatten()
    top_indices = torch.topk(flat_importance, k=top_k).indices

    # Convert flat indices back to (gene_index, token_position)
    top_positions = [(idx.item() // seq_len, idx.item() % seq_len) for idx in top_indices]
    top_scores = [flat_importance[idx].item() for idx in top_indices]

    top_token_pos_df = pd.DataFrame({
        'Gene': [locus_order[g] for g, _ in top_positions],
        'Gene_Index': [g for g, _ in top_positions],
        'Token_Position': [p for _, p in top_positions],
        'Importance_Score': top_scores
    })

    return top_token_pos_df


def map_token_positions_to_original_sequence(
    tokenizer: Callable,
    top_token_pos_df,
) -> List[dict]:
    """
    Maps each token position to its corresponding start position in the original DNA sequence.
    """

    # gene_seqs_dict maps gene name to its original DNA sequence
    # Example: gene_seqs_dict = { "rpoB": "ATGCGT...", "katG": "GCTAGC...", ... }

    with open("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/interpretability/dataloader/ref_gene_seq.json") as f:
        gene_seqs_dict = json.load(f)

    nt_start_list = []
    nt_end_list = []
    nt_spans = []

    for i, row in top_token_pos_df.iterrows():
        gene = row['Gene']
        token_pos = row['Token_Position']
        sequence = gene_seqs_dict[gene]  # original DNA sequence for the gene

        # Tokenize with offset mapping (includes special tokens)
        encoding = tokenizer(
            sequence,
            return_offsets_mapping=True,
            truncation=True,
            return_tensors='pt'
        )

        offsets = encoding['offset_mapping'][0]  # (num_tokens, 2)

        if token_pos < len(offsets):
            start, end = offsets[token_pos].tolist()
            nt_start_list.append(start)
            nt_end_list.append(end - 1)  # using inclusive end for consistency
            nt_spans.append(sequence[start:end])
        else:
            nt_start_list.append(None)
            nt_end_list.append(None)
            nt_spans.append("")

    # Add to DataFrame
    top_token_pos_df["Gene_nt_start"] = nt_start_list
    top_token_pos_df["Gene_nt_end"] = nt_end_list
    top_token_pos_df["Gene_nt_seq"] = nt_spans

    return top_token_pos_df

    
def get_important_features_list(important_tokens_df):
    """
    Extracts a list of important features from the dataframe.

    Each important feature is represented as a tuple of:
        (Gene name, nucleotide start position, nucleotide end position)

    Args:
        important_tokens_df (pd.DataFrame): DataFrame containing columns:
            - 'Gene': gene name (str)
            - 'Gene_nt_start': nucleotide start position (int or convertible to int)
            - 'Gene_nt_end': nucleotide end position (int or convertible to int)

    Returns:
        List[Tuple[str, int, int]]: List of important features as (gene, start, end) tuples.
    """
    important_features = [
        (row['Gene'], int(row['Gene_nt_start']), int(row['Gene_nt_end']))
        for _, row in important_tokens_df.iterrows()
    ]
    return important_features