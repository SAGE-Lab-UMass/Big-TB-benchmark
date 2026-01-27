"""
DNABERT-2 Gene Embedding Generation Script

This script generates high-quality DNA sequence embeddings using the DNABERT-2 model
for antimicrobial resistance prediction. It processes genomic sequences from specific
genes and creates embeddings that capture sequence-level patterns relevant for 
resistance classification.

Key Features:
- Supports both zero-shot and fine-tuned DNABERT-2 models
- Generates multiple embedding types (mean_dim, mean_seq, token-level)
- Handles both single-gene and multi-gene drug resistance scenarios
- Memory-efficient batch processing for large datasets
- Automatic GPU memory management

Embedding Types:
- token: Per-token embeddings (768-dim per position)
- mean_seq: Sequence-wise averaged embeddings (768-dim per sequence)
- mean_dim: Dimension-wise averaged embeddings

Author: Saishradha Mohanty
"""

# Suppress sklearn warnings for cleaner output
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import sklearn.metrics

# Local imports
from dataloader.dataloader import multi_gene_multi_drug_loader_csv
from utils.embed_gen_utils import (
    get_tokenizer_model, 
    calculate_dnaberts_embedding,
    stack_final_phenotypes
)

# Increase CSV field size limit for large sequence data
csv.field_size_limit(sys.maxsize)

def main(args):
    """
    Main function for generating gene embeddings using DNABERT-2.
    
    This function orchestrates the entire embedding generation process:
    1. Sets up the GPU environment and loads the DNABERT-2 model
    2. Processes genomic sequences through the model to generate embeddings
    3. Handles both single-gene and multi-gene scenarios
    4. Saves embeddings in appropriate formats for downstream training
    
    Args:
        Command-line arguments containing model paths, data configuration,
        and embedding generation parameters
    """
    # Extract gene list from arguments
    genes = args.genes
    
    # -----------------------------------------------
    # GPU Setup and Environment Configuration
    # -----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"\t{n_gpu} GPUs available for embedding generation!")
    
    # Clear any existing GPU memory to start fresh
    torch.cuda.empty_cache()

    # -----------------------------------------------
    # Model and Tokenizer Loading
    # -----------------------------------------------
    print("Loading DNABERT-2 tokenizer and model...")
    print(f"Model: {args.model_name_or_path}")
    print(f"Embedding type: {args.embed_type}")
    print(f"Max sequence length: {args.max_length}")
    
    # Load the tokenizer and model (either zero-shot or fine-tuned)
    tokenizer, model = get_tokenizer_model(
        args.model_name_or_path, 
        args.embed_type, 
        args.ft_model_path, 
        args.max_length
    )
    
    # Enable multi-GPU processing if available
    model = nn.DataParallel(model)
    model.to(device)
    print("Model loaded and parallelized across GPUs\n")

    # -----------------------------------------------
    # Data Processing and Embedding Generation
    # -----------------------------------------------
    if args.is_single_gene_algo:
        # Single-gene scenario: Process all data at once
        print("=" * 60)
        print("SINGLE-GENE EMBEDDING GENERATION")
        print(f"Target gene: {genes}")
        print(f"Drug: {args.drug}")
        print("=" * 60)
        
        print("Loading full dataset...")
        full_dataloader = multi_gene_multi_drug_loader_csv(
            args, 
            args.is_single_gene_algo, 
            load_train=True, 
            n_gpu=n_gpu
        )
        print(f"Loaded dataset with {len(full_dataloader)} batches\n")

        print(f"Generating DNABERT-2 embeddings for gene {genes}...")
        print(f"Embedding type: {args.embed_type}")
        print(f"Output directory: {args.embed_dir}")
        
        # Generate embeddings for the full dataset
        calculate_dnaberts_embedding(
            full_dataloader, 
            tokenizer, 
            model, 
            device, 
            args.max_length, 
            args.embed_dir, 
            genes, 
            args.drug, 
            args.is_single_gene_algo, 
            data_partition="full", 
            embed_type=args.embed_type
        )
        print("Full dataset embedding generation complete\n")
        
    else:
        # WARNING: DO NOT RUN THE MULTI-GENE SECTION FOR ALL THE GENES TOGETHER AS IT TAKES TOO MUCH TIME AND RESOURCES.

        # Multi-gene scenario: Process train and validation separately
        print("=" * 60)
        print("MULTI-GENE EMBEDDING GENERATION")
        print(f"Target genes: {genes}")
        print(f"Drug: {args.drug}")
        print("=" * 60)
        
        # ---- Process Training Data ----
        print("Processing training data...")
        print(f"Batch size: {args.train_batch_size}")
        
        train_loader = multi_gene_multi_drug_loader_csv(
            args, 
            args.is_single_gene_algo, 
            load_train=True, 
            n_gpu=n_gpu
        )
        print(f"Loaded training data with {len(train_loader)} batches")

        print("Generating embeddings for training data...")
        calculate_dnaberts_embedding(
            train_loader, 
            tokenizer, 
            model, 
            device, 
            args.max_length, 
            args.embed_dir, 
            genes, 
            args.drug, 
            args.is_single_gene_algo, 
            data_partition="train", 
            embed_type=args.embed_type
        )
        print("Training data embedding generation complete")
        
        # Clean up memory before processing validation data
        del train_loader
        torch.cuda.empty_cache()
        print("GPU memory cleared\n")

        # ---- Process Validation Data ----
        print("Processing validation data...")
        print(f"  Batch size: {args.val_batch_size}")
        
        val_loader = multi_gene_multi_drug_loader_csv(
            args, 
            args.is_single_gene_algo, 
            load_train=False, 
            n_gpu=n_gpu
        )
        print(f"Loaded validation data with {len(val_loader)} batches")

        print("Generating embeddings for validation data...")
        calculate_dnaberts_embedding(
            val_loader, 
            tokenizer, 
            model, 
            device, 
            args.max_length, 
            args.embed_dir, 
            genes, 
            args.drug, 
            args.is_single_gene_algo,
            data_partition="val", 
            embed_type=args.embed_type
        )
        print("Validation data embedding generation complete")

    # -----------------------------------------------
    # Final Processing and Data Stacking
    # -----------------------------------------------
    # Clean up GPU memory after all embedding generation
    torch.cuda.empty_cache()
    print("Final GPU memory cleanup\n")

    print("=" * 60)
    print("POST-PROCESSING: Stacking and Compressing Embeddings")
    print("=" * 60)
    
    print("Stacking phenotype data and compressing to HDF5 format...")
    print(f"Output directory: {args.embed_dir}")
    print(f"Target gene: {genes}")
    
    # Stack final embeddings and phenotypes into efficient storage format
    stack_final_phenotypes(args.embed_dir, data_partition="full", gene=genes)

    # Uncomment below lines if stacking for train/val needed separatelyå
    # stack_final_embeddings(args.embed_dir, data_partition="train", drug=args.drug)
    # stack_final_embeddings(args.embed_dir, data_partition="val", drug=args.drug)
    å
    print("Phenotype stacking and compression complete\n")

    print("=" * 60)
    print("EMBEDDING GENERATION COMPLETED SUCCESSFULLY!")
    print(f"Generated embeddings for: {genes}")
    print(f"Embedding type: {args.embed_type}")
    print(f"Output location: {args.embed_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate DNABERT-2 gene embeddings for resistance prediction')
    
    # Model configuration
    parser.add_argument('--model_name_or_path', type=str, default="pretrained_models/DNABERT2", 
                        help='HuggingFace model name or path to DNABERT-2 model. We download the pretrained model and use it for zero-shot embedding generation.')
    parser.add_argument('--embed_method', type=str, default="zs", 
                        choices=["zs", "ft"],
                        help='Embedding method: "zs" for zero-shot, "ft" for fine-tuned')

    # Sequence processing parameters
    parser.add_argument('--max_length', type=int, default=3000, 
                        help='Maximum sequence length for tokenization (longer sequences will be truncated)')
    parser.add_argument('--embed_type', type=str, default='mean_seq', 
                        choices=['mean_dim', 'mean_seq', 'token'],
                        help='Type of embeddings to generate:\n'
                             '  - token: Per-position embeddings (768-dim per position)\n'
                             '  - mean_seq: Sequence-averaged embeddings (768-dim per sequence)\n' 
                             '  - mean_dim: Dimension-averaged embeddings')

    # Batch size configuration (adjust based on GPU memory)
    parser.add_argument('--train_batch_size', type=int, default=9, 
                        help='Batch size for training data processing (adjust for GPU memory)')
    parser.add_argument('--val_batch_size', type=int, default=9, 
                        help='Batch size for validation data processing')
    parser.add_argument('--full_batch_size', type=int, default=9, 
                        help='Batch size for full dataset processing')
    parser.add_argument('--test_split', type=str, default=0.2, 
                        help='Fraction of data to use for testing/validation')

    # Data paths and configuration
    parser.add_argument('--datapath', type=str, 
                        default='finetune_data/multidrug_classification/training', 
                        help='Base directory containing input data')
    parser.add_argument('--full_dataname', type=str, 
                        default='geno_pheno_full_combined.csv', 
                        help='CSV file with complete genotype-phenotype mappings')
    parser.add_argument('--train_dataname', type=str, 
                        default='geno_pheno_train_combined.csv', 
                        help='CSV file with training genotype-phenotype mappings')
    parser.add_argument('--val_dataname', type=str, 
                        default='geno_pheno_val_combined.csv', 
                        help='CSV file with validation genotype-phenotype mappings')
    
    # Output configuration
    parser.add_argument('--embed_dir', type=str, 
                        default='training_output/transfer_learn/embeddings', 
                        help='Directory to save generated embeddings')
    
    # Phenotype and genotype data
    parser.add_argument('--phenotype_file', type=str, 
                        default='finetune_data/multidrug_classification/training/phenotype/master_resistance_table.csv', 
                        help='CSV file containing resistance phenotypes for isolates')
    parser.add_argument('--genotype_input_directory', type=str, 
                        default='finetune_data/multidrug_classification/train/genotype/combined_aligned/', 
                        help='Directory containing aligned genomic sequences for each gene')
    
    # Target drug and genes
    parser.add_argument('--drug', type=str, default='RIFAMPICIN', 
                        help='Drug name for which to generate embeddings (must match phenotype file)')
    parser.add_argument('--genes', type=str, default='rpoB,rpoC', 
                        help='Comma-separated list of gene names to process (e.g., "katG,inhA" for ISONIAZID)')
    
    # Algorithm configuration
    parser.add_argument('--is_single_gene_algo', action='store_true', 
                        help='Enable this flag for single-gene resistance mechanisms (processes all data at once). Needed for SD-DNABERT-CNN cases. For MD-DNABERT-CNN, leave unset.')
    
    args = parser.parse_args()

    # Validate configuration
    print("=" * 80)
    print("DNABERT-2 Gene Embedding Generation")
    print("=" * 80)
    print(f"Model: {args.model_name_or_path}")
    print(f"Embedding method: {args.embed_method}")
    print(f"Embedding type: {args.embed_type}")
    print(f"Target drug: {args.drug}")
    print(f"Target genes: {args.genes}")
    print(f"Single-gene algorithm: {args.is_single_gene_algo}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Output directory: {args.embed_dir}")
    print("=" * 80)
    
    main(args)