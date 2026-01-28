"""
Utilities for generating embeddings from DNABERT-2 models.

This module provides functions to tokenize DNA sequences, generate embeddings
using DNABERT-2 models, and handle multi-gene, multi-drug data loading and
stacking operations.
"""

import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import h5py
import torch
import torch.nn as nn
import transformers
from torch.amp import autocast
import tqdm

token = os.getenv("HF_AUTH_TOKEN")  # automatically loads the token from environment


def get_tokenizer_model(model_name_or_path, embed_type='zs', ft_model_path=None, model_max_length=400):
    """
    Load the model and tokenizer from the given path.

    Args:
        model_name_or_path (str): The path to the model.
        embed_type (str): Type of embedding ('zs' for zero-shot or 'ft' for fine-tuned).
        ft_model_path (str): Path to fine-tuned model file (required if embed_type='ft').
        model_max_length (int): The maximum length of the input sequence to the model.

    Returns:
        tuple: (tokenizer, model) loaded from the specified path.

    Raises:
        FileNotFoundError: If ft_model_path does not exist when embed_type='ft'.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    pretrained_model = transformers.AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_auth_token=token
    )

    if embed_type == "ft":
        if not os.path.exists(ft_model_path):
            raise FileNotFoundError(f"Model file not found at {ft_model_path}")

        state_dict = torch.load(ft_model_path)
        pretrained_model.load_state_dict(state_dict)
        print(f"Loaded fine-tuned model from {ft_model_path}")

    return tokenizer, pretrained_model


def get_tokens(batch, tokenizer, model_max_length=400):
    """
    Tokenize the input batch of DNA sequences.

    Args:
        batch (list): A list of DNA sequences.
        tokenizer: The tokenizer to use for tokenization.
        model_max_length (int): The maximum length of the input sequence to the model.

    Returns:
        dict: Tokenized features with 'input_ids' and 'attention_mask'.
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
    """
    Extract gene and drug metadata from batch dictionary.

    Args:
        batch (dict): Batch dictionary containing 'gene_order' and 'drug_order' keys.

    Returns:
        tuple: (gene_order, drug_order) lists from the batch.

    Raises:
        AssertionError: If gene_order or drug_order not found in batch.
    """
    gene_order = None
    drug_order = None

    for key, value in batch.items():
        if key == "gene_order":
            gene_order = value
        elif key == "drug_order":
            drug_order = value

    assert gene_order is not None, "gene_order not found in batch"
    assert drug_order is not None, "drug_order not found in batch"

    return gene_order, drug_order


def get_gene_name(key, gene_order):
    """
    Extract gene name from batch key and gene_order.

    Args:
        key (str): Batch key in format 'gene_seq_<index>'.
        gene_order (list): List of gene information tuples.

    Returns:
        str: Gene name with '.fasta' extension removed.
    """
    gene_index = int(key.split('_')[-1]) - 1
    gene_name = gene_order[gene_index][0].replace('.fasta', '')
    return gene_name


def calculate_phenotypes(data_loader, tokenizer, model_max_length=400, embed_dir=None):
    """
    Calculate resistance phenotypes for input DNA sequences.

    Args:
        data_loader: DataLoader providing batches of data.
        tokenizer: Tokenizer for DNA sequences.
        model_max_length (int): Maximum sequence length for tokenization.
        embed_dir (str): Directory to save intermediate phenotype checkpoints.

    Returns:
        np.ndarray: Array of shape (num_samples, num_drugs) containing phenotypes.
    """
    res_phenotypes_all_drugs = None

    for j, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            _, _, res_phenotypes = prepare_multigene_input_fast(
                batch, tokenizer, model_max_length
            )

            if j == 0:
                res_phenotypes_all_drugs = res_phenotypes
            else:
                res_phenotypes_all_drugs = torch.cat(
                    (res_phenotypes_all_drugs, res_phenotypes), dim=0
                )

            if j % 20 == 0 and j != 0:
                np.save(
                    os.path.join(embed_dir, "res_phenotypes_now.npy"),
                    np.array(res_phenotypes_all_drugs.detach().cpu())
                )

    res_phenotypes_all_drugs = np.array(res_phenotypes_all_drugs.detach().cpu())
    return res_phenotypes_all_drugs

def prepare_multigene_input_fast(batch, tokenizer, model_max_length, is_single_gene=False, gene=None):
    """
    Efficiently prepare multi-gene input using parallel tokenization.

    Args:
        batch (dict): Batch dictionary with sequences and phenotypes.
        tokenizer: Tokenizer for DNA sequences.
        model_max_length (int): Maximum sequence length.
        is_single_gene (bool): If True, return only data for specified gene.
        gene (str): Gene name to extract (required if is_single_gene=True).

    Returns:
        tuple: (gene_specific_input_ids, gene_specific_attention_mask, res_phenotypes)
    """
    input_ids_list = []
    attention_mask_list = []
    res_phenotypes_list = []

    gene_order, _ = get_metadata(batch)

    gene_keys = [key for key in batch if key.startswith('gene_seq')]
    res_keys = [key for key in batch if key.startswith('res_phenotype_drug')]

    def tokenize_sequence(seq):
        token_feat = get_tokens(seq, tokenizer, model_max_length)
        return token_feat['input_ids'].unsqueeze(1), token_feat['attention_mask'].unsqueeze(1)

    with ThreadPoolExecutor() as executor:
        tokenized_results = list(executor.map(tokenize_sequence, [batch[k] for k in gene_keys]))

    for input_id, attn_mask in tokenized_results:
        input_ids_list.append(input_id)
        attention_mask_list.append(attn_mask)

    input_ids = torch.cat(input_ids_list, dim=1)
    attention_mask = torch.cat(attention_mask_list, dim=1)
    res_phenotypes = torch.cat([batch[k].unsqueeze(1) for k in res_keys], dim=1)

    gene_specific_input_ids = torch.unbind(input_ids, dim=1)
    gene_specific_attention_mask = torch.unbind(attention_mask, dim=1)

    if is_single_gene:
        gene_index = next(
            i for i, g in enumerate(gene_order)
            if g[0].startswith(gene)
        )
        gene_specific_input_ids = [gene_specific_input_ids[gene_index]]
        gene_specific_attention_mask = [gene_specific_attention_mask[gene_index]]

    return gene_specific_input_ids, gene_specific_attention_mask, res_phenotypes


def calculate_dnaberts_embedding(data_loader, tokenizer, model, device, model_max_length=400,
                                 embed_dir=None, gene=None, drug=None, is_single_gene=False,
                                 data_partition="train", embed_type="mean_seq"):
    """
    Calculate and save DNABERT embeddings for input sequences.

    Args:
        data_loader: DataLoader for input data.
        tokenizer: Tokenizer for sequences.
        model: DNABERT model instance.
        device (str): Device to run model on ('cuda' or 'cpu').
        model_max_length (int): Maximum sequence length.
        embed_dir (str): Directory to save embeddings.
        gene (str): Gene name for single-gene mode.
        drug (str): Drug name (for file naming).
        is_single_gene (bool): If True, process only specified gene.
        data_partition (str): Data partition name ('train', 'val', 'test').
        embed_type (str): Type of embedding ('mean_dim', 'mean_seq', or 'token').

    Raises:
        ValueError: If embed_type is not one of the supported types.
    """
    if embed_type not in ["mean_dim", "mean_seq", "token"]:
        raise ValueError(f"Unsupported embed_type: {embed_type}. Supported types are 'mean_dim', 'mean_seq', 'token'.")

    if embed_type == "mean_dim":
        calculate_dnaberts_mean_dim_embedding(
            data_loader, tokenizer, model, device, model_max_length, embed_dir,
            gene, drug, is_single_gene, data_partition
        )
    elif embed_type == "mean_seq":
        calculate_dnaberts_mean_seq_embedding(
            data_loader, tokenizer, model, device, model_max_length, embed_dir,
            gene, drug, is_single_gene, data_partition
        )
    elif embed_type == "token":
        calculate_dnaberts_token_embedding(
            data_loader, tokenizer, model, device, model_max_length, embed_dir,
            gene, drug, is_single_gene, data_partition
        )


def calculate_dnaberts_mean_dim_embedding(data_loader, tokenizer, model, device, model_max_length=400,
                                         embed_dir=None, gene=None, drug=None, is_single_gene=False,
                                         data_partition="train"):
    """
    Calculate mean dimension embedding (average over hidden dimension).

    Args:
        data_loader: DataLoader for input data.
        tokenizer: Tokenizer for sequences.
        model: DNABERT model instance.
        device (str): Device to run model on.
        model_max_length (int): Maximum sequence length.
        embed_dir (str): Directory to save embeddings.
        gene (str): Gene name for single-gene mode.
        drug (str): Drug name (for file naming).
        is_single_gene (bool): If True, process only specified gene.
        data_partition (str): Data partition name.
    """
    model.eval()
    os.makedirs(embed_dir, exist_ok=True)

    save_path = os.path.join(embed_dir, gene) if is_single_gene else embed_dir
    os.makedirs(save_path, exist_ok=True)

    for j, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            input_ids, attention_mask, res_phenotypes = prepare_multigene_input_fast(
                batch, tokenizer, model_max_length, is_single_gene=is_single_gene, gene=gene
            )

            batch_embeddings = []
            for i in range(len(input_ids)):
                with autocast(device_type="cuda"):
                    dnaberts_output = model(
                        input_ids=input_ids[i].to(device),
                        attention_mask=attention_mask[i].to(device)
                    )[0]

                mask = attention_mask[i].unsqueeze(-1)
                masked_output = dnaberts_output * mask.to(device)
                sum_hidden = masked_output.sum(dim=2)
                dnaberts_output = sum_hidden / mask.to(device).sum(dim=2).clamp(min=1)

                batch_embeddings.append(dnaberts_output.cpu().numpy())

                del mask, masked_output, sum_hidden, dnaberts_output
                torch.cuda.empty_cache()

            batch_embeddings = np.stack(batch_embeddings, axis=1)
            res_phenotypes = res_phenotypes.cpu().numpy()

            np.save(
                os.path.join(save_path, f"zs_{data_partition}_embeddings_batch_{j}.npy"),
                batch_embeddings
            )
            np.save(
                os.path.join(save_path, f"zs_{data_partition}_res_phenotypes_batch_{j}.npy"),
                res_phenotypes
            )

            del input_ids, attention_mask, res_phenotypes
            torch.cuda.empty_cache()

    print(f"All batches processed and saved in directory {save_path}.")


def calculate_dnaberts_token_embedding(data_loader, tokenizer, model, device, model_max_length=400,
                                       embed_dir=None, gene=None, drug=None, is_single_gene=False,
                                       data_partition="train"):
    """
    Calculate token-level embeddings (all tokens preserved).

    Args:
        data_loader: DataLoader for input data.
        tokenizer: Tokenizer for sequences.
        model: DNABERT model instance.
        device (str): Device to run model on.
        model_max_length (int): Maximum sequence length.
        embed_dir (str): Directory to save embeddings.
        gene (str): Gene name for single-gene mode.
        drug (str): Drug name (for file naming).
        is_single_gene (bool): If True, process only specified gene.
        data_partition (str): Data partition name.
    """
    model.eval()
    os.makedirs(embed_dir, exist_ok=True)

    save_path = os.path.join(embed_dir, gene) if is_single_gene else embed_dir
    os.makedirs(save_path, exist_ok=True)

    for j, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            input_ids, attention_mask, res_phenotypes = prepare_multigene_input_fast(
                batch, tokenizer, model_max_length, is_single_gene=is_single_gene, gene=gene
            )

            batch_embeddings = []
            for i in range(len(input_ids)):
                with autocast(device_type="cuda"):
                    dnaberts_output = model(
                        input_ids=input_ids[i].to(device),
                        attention_mask=attention_mask[i].to(device)
                    )[0]

                dnaberts_output = dnaberts_output * attention_mask[i].unsqueeze(-1).to(device)
                batch_embeddings.append(dnaberts_output.cpu().numpy())

                del dnaberts_output
                torch.cuda.empty_cache()

            batch_embeddings = np.stack(batch_embeddings, axis=1)
            res_phenotypes = res_phenotypes.cpu().numpy()

            print("batch_embeddings shape:", batch_embeddings.shape)

            np.save(
                os.path.join(save_path, f"zs_{data_partition}_embeddings_batch_{j}.npy"),
                batch_embeddings
            )
            np.save(
                os.path.join(save_path, f"zs_{data_partition}_res_phenotypes_batch_{j}.npy"),
                res_phenotypes
            )

            del input_ids, attention_mask, res_phenotypes
            torch.cuda.empty_cache()

    print(f"All batches processed and saved in directory {save_path}.")


def calculate_dnaberts_mean_seq_embedding(data_loader, tokenizer, model, device, model_max_length=400,
                                         embed_dir=None, gene=None, drug=None, is_single_gene=False,
                                         data_partition="train"):
    """
    Calculate mean sequence embedding (average over sequence length).

    Args:
        data_loader: DataLoader for input data.
        tokenizer: Tokenizer for sequences.
        model: DNABERT model instance.
        device (str): Device to run model on.
        model_max_length (int): Maximum sequence length.
        embed_dir (str): Directory to save embeddings.
        gene (str): Gene name for single-gene mode.
        drug (str): Drug name (for file naming).
        is_single_gene (bool): If True, process only specified gene.
        data_partition (str): Data partition name.
    """
    model.eval()
    os.makedirs(embed_dir, exist_ok=True)

    save_path = os.path.join(embed_dir, gene) if is_single_gene else embed_dir
    os.makedirs(save_path, exist_ok=True)

    for j, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            input_ids, attention_mask, res_phenotypes = prepare_multigene_input_fast(
                batch, tokenizer, model_max_length, is_single_gene=is_single_gene, gene=gene
            )

            batch_embeddings = []
            for i in range(len(input_ids)):
                with autocast(device_type="cuda"):
                    dnaberts_output = model(
                        input_ids=input_ids[i].to(device),
                        attention_mask=attention_mask[i].to(device)
                    )[0]

                dnaberts_output = dnaberts_output * attention_mask[i].unsqueeze(-1).to(device)
                dnaberts_output = dnaberts_output.sum(dim=1) / attention_mask[i].sum(dim=1, keepdim=True).to(device)

                batch_embeddings.append(dnaberts_output.cpu().numpy())

                del dnaberts_output
                torch.cuda.empty_cache()

            batch_embeddings = np.stack(batch_embeddings, axis=1)
            res_phenotypes = res_phenotypes.cpu().numpy()

            np.save(
                os.path.join(save_path, f"zs_{data_partition}_embeddings_batch_{j}.npy"),
                batch_embeddings
            )
            np.save(
                os.path.join(save_path, f"zs_{data_partition}_res_phenotypes_batch_{j}.npy"),
                res_phenotypes
            )

            del input_ids, attention_mask, res_phenotypes
            torch.cuda.empty_cache()

    print(f"All batches processed and saved in directory {save_path}.")


def stack_final_embeddings(embed_dir, data_partition):
    """
    Stack batch embeddings into a single compressed file.

    Args:
        embed_dir (str): Directory containing batch embedding files.
        data_partition (str): Data partition name ('train', 'val', 'test').

    Returns:
        None. Saves stacked embeddings and phenotypes as .npz file.
    """
    embedding_files = sorted([
        os.path.join(embed_dir, f"zs_{data_partition}_embeddings_batch_{i}.npy")
        for i in range(len(glob.glob(os.path.join(embed_dir, f"zs_{data_partition}_embeddings_batch_*.npy"))))
    ])

    phenotype_files = sorted([
        os.path.join(embed_dir, f"zs_{data_partition}_res_phenotypes_batch_{i}.npy")
        for i in range(len(glob.glob(os.path.join(embed_dir, f"zs_{data_partition}_res_phenotypes_batch_*.npy"))))
    ])

    first_batch = np.load(embedding_files[0])
    first_phenotypes = np.load(phenotype_files[0])

    num_genes, hidden_dim = first_batch.shape[1], first_batch.shape[2]
    num_drugs = first_phenotypes.shape[1]
    total_samples = sum(np.load(f).shape[0] for f in embedding_files)

    print(f"Total samples: {total_samples}, Num genes: {num_genes}, Hidden dim: {hidden_dim}, Num drugs: {num_drugs}")

    final_embeddings = np.empty((total_samples, num_genes, hidden_dim), dtype=np.float32)
    final_phenotypes = np.empty((total_samples, num_drugs), dtype=np.int32)

    current_index = 0
    for emb_file, phen_file in zip(embedding_files, phenotype_files):
        batch_embeddings = np.load(emb_file)
        batch_phenotypes = np.load(phen_file)
        batch_size = batch_embeddings.shape[0]

        final_embeddings[current_index:current_index + batch_size] = batch_embeddings
        final_phenotypes[current_index:current_index + batch_size] = batch_phenotypes
        current_index += batch_size
        print(f"Stacked {emb_file} and {phen_file} → {current_index}/{total_samples} samples")

    save_path = os.path.join(embed_dir, f"zs_{data_partition}_embeddings_phenotypes.npz")
    np.savez_compressed(save_path, embeddings=final_embeddings, phenotypes=final_phenotypes)
    print(f"\nStacked embeddings and phenotypes saved at {save_path}, shape: {final_embeddings.shape}, {final_phenotypes.shape}")


def stack_final_phenotypes(embed_dir, data_partition, gene="inhA"):
    """
    Stack phenotype files for a specific gene.

    Args:
        embed_dir (str): Directory containing phenotype batch files.
        data_partition (str): Data partition name.
        gene (str): Gene name.

    Returns:
        np.ndarray: Stacked phenotypes array.
    """
    phenotype_files = sorted(glob.glob(
        os.path.join(embed_dir, gene, f"zs_{data_partition}_res_phenotypes_batch_*.npy")
    ))

    if not phenotype_files:
        print(f"No phenotype files found for gene {gene} in {embed_dir}")
        return None

    phenotypes = [np.load(f) for f in phenotype_files]
    stacked_phenotypes = np.vstack(phenotypes)

    return stacked_phenotypes
