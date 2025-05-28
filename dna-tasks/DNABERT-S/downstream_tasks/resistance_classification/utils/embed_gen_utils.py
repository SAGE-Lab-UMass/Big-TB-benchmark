import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
from torch.amp import autocast
import tqdm
import os
import glob
import ipdb


from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment

token = os.getenv("HF_AUTH_TOKEN")  # automatically loads the token from environment


def get_tokenizer_model(model_name_or_path, embed_type='zs', ft_model_path=None, model_max_length=400):
    """
    Load the model and tokenizer from the given path.
    
    Args:
        model_name_or_path (str): The path to the model.
        model_max_length (int): The maximum length of the input sequence to the model.
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
    """
    token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=model_max_length, 
                    return_tensors='pt', 
                    padding='max_length', 
                    truncation=True
                )
    
    return token_feat

def calculate_phenotypes(data_loader, tokenizer, model_max_length=400, embed_dir=None):
    """
    Calculate the phenotypes for the input batch of DNA sequences.

    Args:
        batch (list): A list of DNA sequences.
        tokenizer: The tokenizer to use for tokenization.
        model_max_length (int): The maximum length of the input sequence to the model.
    """

    
    # expected shapes
    # input_ids # Shape: num_genes, (batch_size, max sequence_length)
    # attention_mask # Shape: num_genes, (batch_size, max sequence_length)
    # res_phenotypes # Shape: (batch_size, num_drugs)

    res_phenotypes_all_drugs = None
    for j, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            _, _, res_phenotypes = prepare_multigene_input_fast(batch, tokenizer, model_max_length)
            
            # expected shapes
            # input_ids # Shape: num_genes, (batch_size, max sequence_length)
            # attention_mask # Shape: num_genes, (batch_size, max sequence_length)
            # res_phenotypes # Shape: (batch_size, num_drugs)

            # expected shape: (batch_size, num_genes, hidden_dim)

            
            if j==0:
                res_phenotypes_all_drugs = res_phenotypes
            else:
                res_phenotypes_all_drugs = torch.cat((res_phenotypes_all_drugs, res_phenotypes), dim=0)
                # expected shape: (num_samples, num_drugs)

            if j%20 == 0 and j!=0:
                np.save(os.path.join(embed_dir, "res_phenotypes_now.npy"), np.array(res_phenotypes_all_drugs.detach().cpu()))

    # expected shape: (num_samples, num_genes, hidden_dim)
    res_phenotypes_all_drugs = np.array(res_phenotypes_all_drugs.detach().cpu())
    # expected shape: (num_samples, num_drugs)
    
    return res_phenotypes_all_drugs


def calculate_dnaberts_embedding(data_loader, tokenizer, model, device, model_max_length=400, embed_dir=None, data_partition="train"):
    model.eval()

    # Ensure directory exists
    os.makedirs(embed_dir, exist_ok=True)

    for j, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            input_ids, attention_mask, res_phenotypes = prepare_multigene_input_fast(batch, tokenizer, model_max_length)

            batch_embeddings = []
            for i in range(len(input_ids)):
                with autocast(device_type="cuda"):
                    dnaberts_output = model(
                        input_ids=input_ids[i].to(device), 
                        attention_mask=attention_mask[i].to(device)
                    )[0]

                # Directly calculate mean on GPU without stacking
                dnaberts_output = dnaberts_output * attention_mask[i].unsqueeze(-1).to(device)
                dnaberts_output = dnaberts_output.sum(dim=1) / attention_mask[i].sum(dim=1, keepdim=True).to(device)
                
                # Move to CPU immediately
                batch_embeddings.append(dnaberts_output.cpu().numpy())

                del dnaberts_output
                torch.cuda.empty_cache()

            # Convert batch embeddings to a numpy array
            batch_embeddings = np.stack(batch_embeddings, axis=1)  # (batch_size, num_genes, hidden_dim)
            res_phenotypes = res_phenotypes.cpu().numpy()

            # Save each batch immediately
            np.save(os.path.join(embed_dir, f"zs_{data_partition}_embeddings_batch_{j}.npy"), batch_embeddings)
            np.save(os.path.join(embed_dir, f"zs_{data_partition}_res_phenotypes_batch_{j}.npy"), res_phenotypes)

            del input_ids, attention_mask, res_phenotypes
            torch.cuda.empty_cache()

    print("All batches processed and saved.")

    return None  # No longer return embeddings in memory


def calculate_dnaberts_embedding_part(data_loader, tokenizer, model, device, model_max_length=400, embed_dir=None):
    embeddings_list = []
    res_phenotypes_list = []
    model.eval()

    for j, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            input_ids, attention_mask, res_phenotypes = prepare_multigene_input_fast(batch, tokenizer, model_max_length)

            # Get DNABERT-S outputs and attention masks for the set of genes
            dnaberts_output_list = []
            for i in range(len(input_ids)):
                with autocast(device_type="cuda"):
                    dnaberts_output = model(
                        input_ids=input_ids[i].to(device), 
                        attention_mask=attention_mask[i].to(device)
                    )[0]
                
                dnaberts_output = dnaberts_output * attention_mask[i].unsqueeze(-1).to(device)
                dnaberts_output = dnaberts_output.sum(dim=1) / attention_mask[i].sum(dim=1, keepdim=True).to(device)
                dnaberts_output_list.append(dnaberts_output.cpu())  # Save directly to CPU

                del dnaberts_output  # Free GPU memory
                torch.cuda.empty_cache()

            # Stack and permute (batch_size, num_genes, hidden_dim)
            embedding = torch.stack(dnaberts_output_list, dim=0).permute(1, 0, 2)
            embeddings_list.append(embedding)
            res_phenotypes_list.append(res_phenotypes.cpu())  # Store directly on CPU
            
            del dnaberts_output_list, input_ids, attention_mask
            torch.cuda.empty_cache()

            # Periodically save intermediate results to avoid memory overflow
            if (j + 1) % 20 == 0:
                np.save(os.path.join(embed_dir, f"zs_train_embeddings_aligned.npy"), 
                        torch.cat(embeddings_list).numpy())
                np.save(os.path.join(embed_dir, f"zs_train_res_phenotypes_aligned.npy"), 
                        torch.cat(res_phenotypes_list).numpy())
                
                embeddings_list = []  # Clear saved list to free memory
                res_phenotypes_list = []

                torch.cuda.empty_cache()

    # Final concatenation and saving
    if len(embeddings_list) > 0:
        np.save(os.path.join(embed_dir, "zs_train_embeddings_final.npy"), 
                torch.cat(embeddings_list).numpy())
        np.save(os.path.join(embed_dir, "zs_train_res_phenotypes_final.npy"), 
                torch.cat(res_phenotypes_list).numpy())

    # Final output
    embeddings = torch.cat(embeddings_list).numpy()
    res_phenotypes = torch.cat(res_phenotypes_list).numpy()
    assert embeddings.shape[0] == res_phenotypes.shape[0], "Number of samples in embeddings and res_phenotypes do not match"

    return embeddings, res_phenotypes


def calculate_dnaberts_embedding_old(data_loader, tokenizer, model, device, model_max_length=400, embed_dir=None):
    embeddings = None
    res_phenotypes_all_drugs = None

    for j, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            input_ids, attention_mask, res_phenotypes = prepare_multigene_input_fast(batch, tokenizer, model_max_length)

            # get number of genes and drugs
            num_genes = len(input_ids)
            num_drugs = len(res_phenotypes)

            
            # expected shapes
            # input_ids # Shape: num_genes, (batch_size, max sequence_length)
            # attention_mask # Shape: num_genes, (batch_size, max sequence_length)
            # res_phenotypes # Shape: (batch_size, num_drugs)

            dnaberts_output_list = []
            attention_mask_list = []

            # Get DNABERT-S outputs and attention masks for the set of genes
            # Loop through each gene in the batch
            for i in range(num_genes):
                # Mixed precision for memory efficiency
                # with autocast(device_type="cuda"):
                dnaberts_output = model(input_ids=input_ids[i].to(device), attention_mask=attention_mask[i].to(device))[0].detach().cpu() # token-level embeddings
                
                dnaberts_output_list.append(dnaberts_output)                         # shape: (num_genes, batch_size, seq_len, hidden_dim)
                attention_mask_list.append(attention_mask[i].unsqueeze(-1))  # shape: (num_genes, batch_size, seq_len, 1)

                torch.cuda.empty_cache()

            # Apply attention mask to zero out padding
            masked_model_outputs = [emb * mask for emb, mask in zip(dnaberts_output_list, attention_mask_list)]
            # expected shape: (num_genes, batch_size, sequence_length, hidden_dim)

            del dnaberts_output_list
            torch.cuda.empty_cache()

            masked_model_outputs = torch.stack(masked_model_outputs, dim=0)
            attention_mask_tensor = torch.stack(attention_mask_list, dim=0)
            # expected shape: (num_genes, batch_size, sequence_length, hidden_dim)

            del attention_mask_list


            # Compute mean over sequence length (dim=2), avoiding padded tokens
            embedding = torch.sum(masked_model_outputs, dim=2) / torch.sum(attention_mask_tensor, dim=2)
            embedding = embedding.permute(1, 0, 2)
            # expected shape: (batch_size, num_genes, hidden_dim)

            del masked_model_outputs, attention_mask_tensor
            torch.cuda.empty_cache()
            
            if j==0:
                embeddings = embedding
                res_phenotypes_all_drugs = res_phenotypes
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)
                res_phenotypes_all_drugs = torch.cat((res_phenotypes_all_drugs, res_phenotypes), dim=0)

            if j%20 == 0 and j!=0:
                np.save(os.path.join(embed_dir, "zs_train_embeddings_aligned.npy"), np.array(embeddings.detach().cpu()))
                np.save(os.path.join(embed_dir, "zs_train_res_phenotypes_aligned.npy"), np.array(res_phenotypes_all_drugs.detach().cpu()))

                # np.savez(os.path.join(embed_dir, "train_data_aligned.npz"), 
                #                         embeddings=np.array(embeddings.detach().cpu()), 
                #                         res_phenotypes=np.array(res_phenotypes_all_drugs.detach().cpu())

            torch.cuda.empty_cache()


    embeddings = np.array(embeddings.detach().cpu())
    # expected shape: (num_samples, num_genes, hidden_dim)
    res_phenotypes = np.array(res_phenotypes.detach().cpu())
    # expected shape: (num_samples, num_drugs)
    assert embeddings.shape[0] == res_phenotypes.shape[0], "Number of samples in embeddings and res_phenotypes do not match"

    return embeddings, res_phenotypes


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


def prepare_multigene_input(batch, tokenizer, model_max_length):
    # Prepare lists to collect input IDs and attention masks for each sequence
    input_ids_list = []
    attention_mask_list = []
    res_phenotypes_list = []

    gene_order, _ = get_metadata(batch)

    # Loop through all keys in batch that start with 'seq'
    for key, text in batch.items():
        if key.startswith('gene_seq'):
            # Tokenize each sequence
            token_feat = get_tokens(text, tokenizer, model_max_length)
            
            # Append the unsqueezed tensors to the lists for concatenation later
            input_ids_list.append(token_feat['input_ids'].unsqueeze(1))
            attention_mask_list.append(token_feat['attention_mask'].unsqueeze(1))

        elif key.startswith('res_phenotype_drug'):
            res_phenotypes_list.append(text.unsqueeze(1))
            

    # Concatenate all sequences along the new dimension
    input_ids = torch.cat(input_ids_list, dim=1)  # Shape: (batch_size, num_genes, sequence_length)
    attention_mask = torch.cat(attention_mask_list, dim=1)  # Shape: (batch_size, num_genes, max allowed sequence_length)
    res_phenotypes = torch.cat(res_phenotypes_list, dim=1)  # Shape: (batch_size, num_drugs)

    # Unbind along the gene axis (dim=1)
    gene_specific_input_ids = torch.unbind(input_ids, dim=1)           # List of (batch_size, seq_len) for each gene
    gene_specific_attention_mask = torch.unbind(attention_mask, dim=1) # List of (batch_size, seq_len) for each gene

    # Move to GPU and return the tensors
    return gene_specific_input_ids, gene_specific_attention_mask, res_phenotypes.detach()


def get_gene_name(key, gene_order):
    # Extract the gene index from the key: e.g., 'gene_seq_3' -> 2 (0-based)
    gene_index = int(key.split('_')[-1]) - 1
    gene_name = gene_order[gene_index][0].replace('.fasta', '')

    return gene_name


def prepare_multigene_input_fast(batch, tokenizer, model_max_length):

    input_ids_list = []
    attention_mask_list = []
    res_phenotypes_list = []

    gene_order, _ = get_metadata(batch)

    # Collect gene keys and resistance keys
    gene_keys = [key for key in batch if key.startswith('gene_seq')]
    res_keys = [key for key in batch if key.startswith('res_phenotype_drug')]


    # Parallel tokenize sequences
    def tokenize_sequence(seq):
        token_feat = get_tokens(seq, tokenizer, model_max_length)
        return token_feat['input_ids'].unsqueeze(1), token_feat['attention_mask'].unsqueeze(1)

    with ThreadPoolExecutor() as executor:
        tokenized_results = list(executor.map(tokenize_sequence, [batch[k] for k in gene_keys]))


    # Split tokenization results into separate lists
    for input_id, attn_mask in tokenized_results:
        input_ids_list.append(input_id)
        attention_mask_list.append(attn_mask)

    # Concatenate tensors
    input_ids = torch.cat(input_ids_list, dim=1)          # Shape: (batch_size, num_genes, seq_len)
    attention_mask = torch.cat(attention_mask_list, dim=1)  # Shape: (batch_size, num_genes, seq_len)

    # Concatenate resistance phenotypes
    res_phenotypes = torch.cat([batch[k].unsqueeze(1) for k in res_keys], dim=1)  # (batch_size, num_drugs)

    # Unbind along gene axis (dim=1)
    gene_specific_input_ids = torch.unbind(input_ids, dim=1)
    gene_specific_attention_mask = torch.unbind(attention_mask, dim=1)

    return gene_specific_input_ids, gene_specific_attention_mask, res_phenotypes


def stack_final_embeddings(embed_dir, data_partition):
    # Collect all the batch files (in the natural saved order)
    embedding_files = [os.path.join(embed_dir, f"zs_{data_partition}_embeddings_batch_{i}.npy") 
                    for i in range(len(glob.glob(os.path.join(embed_dir, f"zs_{data_partition}_embeddings_batch_*.npy"))))]

    phenotype_files = [os.path.join(embed_dir, f"zs_{data_partition}_res_phenotypes_batch_{i}.npy") 
                    for i in range(len(glob.glob(os.path.join(embed_dir, f"zs_{data_partition}_res_phenotypes_batch_*.npy"))))]

    # Load the first file to get the embedding shape (num_genes, hidden_dim)
    first_batch = np.load(embedding_files[0])
    first_phenotypes = np.load(phenotype_files[0])

    num_genes, hidden_dim = first_batch.shape[1], first_batch.shape[2]
    num_drugs = first_phenotypes.shape[1]  # Phenotypes now have shape (num_samples, num_drugs)
    total_samples = sum(np.load(f).shape[0] for f in embedding_files)  # Total number of samples

    print(f"Total samples: {total_samples}, Num genes: {num_genes}, Hidden dim: {hidden_dim}, Num drugs: {num_drugs}")

    # Pre-allocate the final stacked arrays
    final_embeddings = np.empty((total_samples, num_genes, hidden_dim), dtype=np.float32)
    final_phenotypes = np.empty((total_samples, num_drugs), dtype=np.int32)  # Multi-drug phenotypes

    # Efficiently load and stack without holding all batches in RAM
    current_index = 0
    for emb_file, phen_file in zip(embedding_files, phenotype_files):
        batch_embeddings = np.load(emb_file)
        batch_phenotypes = np.load(phen_file)
        batch_size = batch_embeddings.shape[0]
        
        # Directly copy batch into the pre-allocated arrays
        final_embeddings[current_index:current_index + batch_size] = batch_embeddings
        final_phenotypes[current_index:current_index + batch_size] = batch_phenotypes
        current_index += batch_size
        print(f"Stacked {emb_file} and {phen_file} -> {current_index}/{total_samples} samples")

    # Save the final stacked embeddings and phenotypes together as a compressed .npz file
    save_path = os.path.join(embed_dir, f"zs_{data_partition}_embeddings_phenotypes.npz")
    np.savez_compressed(save_path, embeddings=final_embeddings, phenotypes=final_phenotypes)
    print(f"\nStacked embeddings and phenotypes saved at {save_path}, shape: {final_embeddings.shape}, {final_phenotypes.shape}")