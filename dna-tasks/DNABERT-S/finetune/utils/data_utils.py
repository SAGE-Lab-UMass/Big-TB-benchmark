import torch
from concurrent.futures import ThreadPoolExecutor

def get_metadata(batch):
    """
    Extracts metadata from the batch dictionary.
    Args:
        batch (dict): The batch dictionary containing metadata.
        Returns:
        tuple: A tuple containing gene_order and drug_order.
    """
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

def prepare_multigene_input_fast(batch, tokenizer, model_max_length):
    input_ids_list, attention_mask_list, res_phenotypes_list = [], [], []
    gene_order, _ = get_metadata(batch)

    gene_keys = [key for key in batch if key.startswith('gene_seq')]
    res_keys = [key for key in batch if key.startswith('res_phenotype_drug')]

    def tokenize_sequence(seq):
        token_feat = tokenizer.batch_encode_plus(
            seq, max_length=model_max_length, return_tensors='pt', padding='max_length', truncation=True)
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

    return gene_specific_input_ids, gene_specific_attention_mask, res_phenotypes