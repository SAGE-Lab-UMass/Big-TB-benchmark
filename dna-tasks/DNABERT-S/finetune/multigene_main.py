# 4. train.py

import argparse
import os
import sys
import csv
import torch
import torch.nn as nn
import tqdm
import ipdb

from dataloader.dataloader import multi_gene_multi_drug_loader_csv
from multigene_model import DNABERTClassifier
from utils.model_utils import * 
from utils.classification_metric_utils import *
# from multigene_train import train_dnabert_classifier
from multigene_acc_grad_train import train_dnabert_classifier
from peft import get_peft_model

import warnings
warnings.filterwarnings('ignore')


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# can be commented if using A100 and triton
os.environ["FLASH_ATTENTION_DISABLE"] = "1"

csv.field_size_limit(sys.maxsize)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"\nUsing {n_gpu} GPUs!\n")

    torch.cuda.empty_cache()

    print("Loading train data...")
    train_loader = multi_gene_multi_drug_loader_csv(args, load_train=True, n_gpu=n_gpu)
    val_loader = multi_gene_multi_drug_loader_csv(args, load_train=False, n_gpu=n_gpu)
    print("done!\n")

    print("Loading tokenizer and model...")
    tokenizer, base_model = get_tokenizer_model(args.model_name, args.max_length)
    model = DNABERTClassifier(base_model, hidden_dim=768, num_drugs=args.num_drugs)

    # print("Apply LoRA!\n")
    # lora_config = get_lora_config(args)
    # model = get_peft_model(model, lora_config)
    # print(f"LoRA applied! Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Freeze and partially unfreeze
    unfreeze_last_n_layers(model.module.base_model if hasattr(model, 'module') else model.base_model, 
                           n_last_layers=args.n_last_layers_to_unfreeze)

    model = nn.DataParallel(model)
    model.to(device)
    print("done!\n")

    optimizer = get_optimizer(model, args)
    criterion = MaskedMultiWeightedBCE()
    scheduler = get_scheduler(optimizer, args, train_loader)

    print("Training...")
    train_dnabert_classifier(model, train_loader, val_loader, tokenizer, optimizer, criterion, scheduler, device, args, resume_checkpoint=None)
    print("done!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="zhihan1996/DNABERT-S")
    parser.add_argument('--max_length', type=int, default=2000, help="Max length of tokens") # works with 1 A100
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--datapath', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/eval', help="The dict of data")
    parser.add_argument('--pkl_file', type=str, default='geno_pheno_full_combined.csv', help="Pickle file containing geno-pheno mapping of isolate strains per drug")
    parser.add_argument('--model_checkpoint_path', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/finetune/saved_checkpoints/checkpoint_epoch_0.pth', help="Path to the model checkpoint")
    parser.add_argument('--output_dir', type=str, default="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/finetune")
    parser.add_argument('--num_drugs', type=int, required=False, default=11, help="Number of drugs to train on")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--train_dataname', type=str, default='geno_pheno_train_combined.csv', help="Name of the data used for training")
    parser.add_argument('--val_dataname', type=str, default='geno_pheno_val_combined.csv', help="Name of the data used for validating")
    parser.add_argument('--phenotype_file', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/eval/phenotype/master_table_resistance.csv', help="Contains the phenotype of the isolate strains per drug")
    parser.add_argument('--genotype_input_directory', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/eval/genotype/combined/', help="Contains the genotype of the isolate strains per drug")
    parser.add_argument('--lr_scale', type=float, default=5.0, help='Scaling factor for classification head learning rate')
    parser.add_argument('--test_split', type=str, default=0.2, help="Test split ratio")
    parser.add_argument('--n_last_layers_to_unfreeze', type=int, default=3, help="Number of last layers to unfreeze for training")
    parser.add_argument('--grad_accum_steps', type=int, default=12, help="Gradient accumulation steps")

    parser.add_argument('--lora_r', type=float, default=8, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=float, default=32, help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.1, help="LoRA dropout rate")
    


    args = parser.parse_args()

    train(args)
