import torch
import torch.nn as nn
import transformers
import os
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType


token = os.getenv("HF_AUTH_TOKEN")

# class DNABertEmbeddingWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, input_ids, attention_mask):
#         inputs_embeds = self.model.embeddings(input_ids)
#         return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)


def get_tokenizer_model(model_name_or_path, model_max_length=400):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    model = transformers.AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_auth_token=token
    )

    # wrapped_model = DNABertEmbeddingWrapper(model)

    return tokenizer, model

# def get_optimizer(model, args):
#     optimizer = torch.optim.AdamW([
#         {'params': model.module.base_model.parameters(), 'lr': args.lr},
#         {'params': model.module.fc.parameters(), 'lr': args.lr * args.lr_scale},
#     ])
#     return optimizer

def get_optimizer(model, args):
    backbone = model.module.base_model if hasattr(model, 'module') else model.base_model
    head = model.module.fc if hasattr(model, 'module') else model.fc

    base_model_params = filter(lambda p: p.requires_grad, backbone.parameters())
    fc_params = head.parameters()

    optimizer = torch.optim.AdamW([
        {'params': base_model_params, 'lr': args.lr},
        {'params': fc_params, 'lr': args.lr * args.lr_scale},
    ])
    return optimizer

def unfreeze_last_n_layers(base_model, n_last_layers=2):
    """
    Unfreeze the last N layers of the base model.

    Args:
        base_model: The base model to unfreeze.
        n_last_layers (int): The number of last layers to unfreeze.
    """
    # Freeze everything first
    for param in base_model.parameters():
        param.requires_grad = False

    # Collect all encoder layers
    encoder_layers = []
    for name, param in base_model.named_parameters():
        if "encoder.layer." in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            encoder_layers.append(layer_num)
    encoder_layers = sorted(set(encoder_layers))

    # Unfreeze last N layers
    last_layers = encoder_layers[-n_last_layers:]
    for name, param in base_model.named_parameters():
        for layer_idx in last_layers:
            if f"encoder.layer.{layer_idx}." in name:
                param.requires_grad = True

# after defining optimizer:
def get_scheduler(optimizer, args, train_loader):
    """
    Get the learning rate scheduler.
    
    Args:
        optimizer: The optimizer to use for training.
        args: The arguments containing the number of epochs.
        train_loader: The training data loader.
    """
    # Calculate total number of training steps
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return scheduler

def get_lora_config(args):
    """
    Get the LoRA configuration.
    
    Returns:
        dict: The LoRA configuration.
    """
    # Apply LoRA structure
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,   # true for inference
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["Wqkv"], 
    )

    return lora_config