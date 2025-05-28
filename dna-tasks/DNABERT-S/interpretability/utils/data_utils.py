import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

def get_tokens(batch, tokenizer, model_max_length=400):
    """
    Tokenize the input batch of DNA sequences.

    Args:
        batch (list): A list of DNA sequences.
        tokenizer: The tokenizer to use for tokenization.
        model_max_length (int): The maximum length of the input sequence to the model.
    """
    token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=model_max_length, 
                    return_tensors='pt', 
                    padding='max_length', 
                    truncation=True
                )
    
    return token_feat


def get_metadata(batch):
    gene_order = None
    drug_order = None

    # First loop: capture metadata
    for key, value in batch.items():
        if key == "gene_order":
            gene_order = value
        elif key == "drug_order":
            drug_order = value

    assert gene_order is not None, "gene_order not found in batch"
    assert drug_order is not None, "drug_order not found in batch"

    return gene_order, drug_order


def get_gene_name(key, gene_order):
    # Extract the gene index from the key: e.g., 'gene_seq_3' -> 2 (0-based)
    gene_index = int(key.split('_')[-1]) - 1
    gene_name = gene_order[gene_index][0].replace('.fasta', '')

    return gene_name


def prepare_multigene_input_fast(batch, tokenizer, model_max_length):

    input_ids_list = []
    attention_mask_list = []

    gene_order, _ = get_metadata(batch)

    # Step 1: Collect gene keys and resistance keys
    gene_keys = [key for key in batch if key.startswith('gene_seq')]
    res_keys = [key for key in batch if key.startswith('res_phenotype_drug')]


    # Step 2: Parallel tokenize sequences
    def tokenize_sequence(seq):
        token_feat = get_tokens(seq, tokenizer, model_max_length)
        return token_feat['input_ids'].unsqueeze(1), token_feat['attention_mask'].unsqueeze(1)

    with ThreadPoolExecutor() as executor:
        tokenized_results = list(executor.map(tokenize_sequence, [batch[k] for k in gene_keys]))


    # Step 3: Split tokenization results into separate lists
    for input_id, attn_mask in tokenized_results:
        input_ids_list.append(input_id)
        attention_mask_list.append(attn_mask)

    # Step 4: Concatenate tensors
    input_ids = torch.cat(input_ids_list, dim=1)          # Shape: (batch_size, num_genes, seq_len)
    attention_mask = torch.cat(attention_mask_list, dim=1)  # Shape: (batch_size, num_genes, seq_len)

    # Step 5: Concatenate resistance phenotypes
    res_phenotypes = torch.cat([batch[k].unsqueeze(1) for k in res_keys], dim=1)  # (batch_size, num_drugs)

    # Step 6: Unbind along gene axis (dim=1)
    gene_specific_input_ids = torch.unbind(input_ids, dim=1)
    gene_specific_attention_mask = torch.unbind(attention_mask, dim=1)

    return gene_specific_input_ids, gene_specific_attention_mask, res_phenotypes


