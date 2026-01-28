import os
import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers
from downstream_cnn_model import *

token = os.getenv("HF_AUTH_TOKEN")  # automatically loads the token from environment

def get_models(dnabert_model_name, dnabert_model_max_len, downstream_model_name, downstream_model_path, device):
    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            dnabert_model_name,
            cache_dir=None,
            model_max_length=dnabert_model_max_len,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    
    # Load the DNABERT model
    dnabert_model = transformers.AutoModel.from_pretrained(
            dnabert_model_name,
            trust_remote_code=True,
            use_auth_token=token
        )
    dnabert_model = nn.DataParallel(dnabert_model).to(device)
    
    # Load the downstream model
    downstream_model = MDCNN(dropout_rate=0)
    # downstream_model.load_state_dict(torch.load(downstream_model_path + "/" + downstream_model_name))
    # Safe and portable model loading
    state_dict = torch.load(
        downstream_model_path + "/" + downstream_model_name,
        map_location=device
    )
    downstream_model.load_state_dict(state_dict)

    downstream_model = nn.DataParallel(downstream_model).to(device)

    return tokenizer, dnabert_model, downstream_model

def get_model_class(model_name, in_dim=768, seq_len=5000, num_classes=11, device='cuda'):
    """
    Get the model class based on the model name.
    """
    if model_name == 'MDMLP':
        return MDMLP(dropout_rate=0).to(device)
    elif model_name == 'MDCNN':
        return MDCNN(num_classes=num_classes, dropout_rate=0).to(device)
    elif model_name == 'DNABERTCNN':
        return DNABERTCNN(seq_len=seq_len, in_dim=in_dim, stem_out=64).to(device)
    elif model_name == 'PooledDNABERTCNN':
        return PooledDNABERTCNN(in_dim=in_dim, stem_out=64).to(device)
    elif model_name == 'MeanDNABERTCNN':
        return MeanDNABERTCNN(seq_len=seq_len, in_dim=in_dim, num_classes=1, dropout_rate=0).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



