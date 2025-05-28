import os
import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers
from downstream_models import *

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


