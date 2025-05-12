def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import os
from sklearn.preprocessing import normalize
import csv
import sys
import numpy as np
import ipdb

import torch
import torch.nn as nn
from torch.amp import autocast
from dataloader.dataloader import multi_gene_multi_drug_loader_csv
from utils.embed_gen_utils import *

import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

csv.field_size_limit(sys.maxsize)
csv.field_size_limit(sys.maxsize)

def main(args):

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(n_gpu))

    torch.cuda.empty_cache()

    # get tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer, model = get_tokenizer_model(args.model_name, args.max_length)
    print("done!\n")

    # parallelize model
    model = nn.DataParallel(model)
    model.to(device)


    # DNA sequences wit corresponding R/S label saved in csv format
    # load data
    print("Loading train data...")
    train_loader = multi_gene_multi_drug_loader_csv(args, load_train=True, n_gpu=n_gpu)
    print("done!\n")
    

    # get dnabertS embeddings for our data
    print("Getting dnabertS embeddings for our training data...")
    calculate_dnaberts_embedding(train_loader, tokenizer, model, device, args.max_length, args.embed_dir, data="train")
    print("done!\n")

    del train_loader
    torch.cuda.empty_cache()
    print("Freeing up GPU memory...\n")


    print("Loading val data...")
    val_loader = multi_gene_multi_drug_loader_csv(args, load_train=False, n_gpu=n_gpu)
    print("done!\n")

    print("Getting dnabertS embeddings for our validation data...")
    calculate_dnaberts_embedding(val_loader, tokenizer, model, device, args.max_length, args.embed_dir, data="val")
    print("done!\n")

    del val_loader
    torch.cuda.empty_cache()
    print("Freeing up GPU memory...\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--model_name', type=str, default="zhihan1996/DNABERT-S", help='Model name')
    parser.add_argument('--max_length', type=int, default=3000, help="Max length of tokens")
    parser.add_argument('--train_batch_size', type=int, default=9, help="Batch size used for training dataset") # with 80GB VRAM - 9 batch size 
    parser.add_argument('--val_batch_size', type=int, default=9, help="Batch size used for validating dataset")
    parser.add_argument('--test_split', type=str, default=0.2, help="Test split ratio")
    parser.add_argument('--datapath', type=str, default='finetune_data/multidrug_classification/eval', help="The dict of data")
    parser.add_argument('--pkl_file', type=str, default='geno_pheno_full_combined.csv', help="Pickle file containing geno-pheno mapping of isolate strains per drug")
    parser.add_argument('--train_dataname', type=str, default='geno_pheno_train_combined.csv', help="Name of the data used for training")
    parser.add_argument('--val_dataname', type=str, default='geno_pheno_val_combined.csv', help="Name of the data used for validating")
    parser.add_argument('--embed_dir', type=str, default='training_output/transfer_learn/embeddings', help="Directory to save embeddings")
    parser.add_argument('--phenotype_file', type=str, default='finetune_data/multidrug_classification/training/phenotype/master_table_resistance.csv', help="Contains the phenotype of the isolate strains per drug")
    parser.add_argument('--genotype_input_directory', type=str, default='finetune_data/multidrug_classification/eval/genotype/combined/', help="Contains the genotype of the isolate strains per drug")
    
    args = parser.parse_args()
    main(args)