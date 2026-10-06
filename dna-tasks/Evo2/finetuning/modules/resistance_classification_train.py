import subprocess
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from downstream_cnn_model import *
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from dataloader.dataloader import *
from dataloader.locus_order import DRUG_TO_LOCI, DRUGS, locus_order 
from utils.classification_metric_utils import *
# from utils.train_utils import *
from utils.token_train_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def stratified_split_dataset(full_dataset, label_dict, test_size=0.2, seed=42):
    """
    Stratified split for single- or multi-gene datasets without loading embeddings.
    Automatically uses label_dict keys for alignment and persists split indices.
    """

    print("\n[Split] Performing stratified train/test split...")
    print(f"Dataset size: {len(full_dataset)} | Label map: {len(label_dict)}")

    # ----------------------------------------------------------------
    # Extract ordered labels from the dataset's lookup or id list
    # ----------------------------------------------------------------
    if hasattr(full_dataset, "lookup"):  # e.g., TokenMemmapMap
        seq_ids = [full_dataset.blocks[bidx][0][ridx] for bidx, ridx in full_dataset.lookup]
    elif hasattr(full_dataset, "ids"):   # e.g., MultiGeneConcatDataset
        seq_ids = full_dataset.ids
    else:
        raise ValueError("Dataset type not recognized: missing lookup/ids attribute")

    # Filter only samples present in label_dict
    valid_ids = [sid for sid in seq_ids if sid in label_dict]
    labels = np.array([label_dict[sid] for sid in valid_ids], dtype=float)
    print(f"Valid samples found: {len(valid_ids)}")

    # ----------------------------------------------------------------
    # Convert IDs to dataset indices for Subset creation
    # ----------------------------------------------------------------
    id_to_idx = {sid: idx for idx, sid in enumerate(seq_ids)}
    valid_indices = np.array([id_to_idx[sid] for sid in valid_ids])

    # ----------------------------------------------------------------
    # Stratified split using sklearn
    # ----------------------------------------------------------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(valid_indices, labels))

    train_indices = valid_indices[train_idx]
    test_indices = valid_indices[test_idx]

    y_train, y_test = labels[train_idx], labels[test_idx]

    return train_indices, test_indices, y_train, y_test



def concatenate_gene_embeddings(embeddings, use_pca=False, pca_components=10):
    """
    Concatenate embeddings across genes along the sequence length axis.

    Input:
        embeddings: torch.Tensor of shape (num_samples, dim, seq_len, num_selected_genes)
    
    Output:
        torch.Tensor of shape (num_samples, dim, seq_len * num_selected_genes)
    """
    if use_pca:
        B, D, L, G = embeddings.shape  # (batch, dim, seq_len, num_genes)
    
        # Move embedding dim to last → shape: (batch, seq_len, num_genes, dim)
        embeddings = embeddings.permute(0, 2, 3, 1)
        
        # Flatten all time positions → shape: (batch, seq_len * num_genes, dim)
        flattened = embeddings.reshape(-1, D).cpu().numpy()
        
        # Fit PCA
        pca = PCA(n_components=pca_components)
        reduced = pca.fit_transform(flattened)  # (batch * seq_len * num_genes, n_components)

        # Reshape back: (batch, seq_len * num_genes, n_components)
        reduced = torch.from_numpy(reduced).float().reshape(B, L * G, pca_components)

        # Permute to (batch, n_components, seq_len * num_genes)
        concatenated_embeddings = reduced.permute(0, 2, 1)
    else:
        # Combine the seq_len and gene dimensions
        concatenated_embeddings = embeddings.reshape(embeddings.size(0), embeddings.size(1), -1)
    return concatenated_embeddings


DRUG_INDEX = {
    'AMIKACIN': 0, # r
    'CAPREOMYCIN': 1,
    'ETHAMBUTOL': 2, # r
    'ETHIONAMIDE': 3, # r
    'ISONIAZID': 4, # r
    'KANAMYCIN': 5, # r
    'LEVOFLOXACIN': 6, # r
    'MOXIFLOXACIN': 7, # r
    'PYRAZINAMIDE': 8, # r
    'RIFAMPICIN': 9, # r
    'STREPTOMYCIN': 10 # r
}

def build_label_map(label_file, drug, prefix="train"):
    print(f"Loading labels from: {label_file}")
    drug_index = DRUGS.index(drug)

    # TODO: Remove this hardcoded index
    drug_index = DRUG_INDEX[drug]

    label_np_file = np.load(label_file)
    labels = label_np_file["phenotypes"]  # shape = (num_samples, num_drugs)
    drug_labels = labels[:, drug_index]

    print(f"Building label map for drug: {drug} (index {drug_index})")

    print(f"Total samples for drug {drug}: {len(drug_labels)} (including missing labels)")

    # Keep only samples with valid (non -1/missing) labels
    valid_indices = np.where(drug_labels != -1)[0]
    drug_labels = drug_labels[valid_indices]
    print(f"Total samples for drug {drug}: {len(drug_labels)} (valid labels)")


    # label_map = {
    #     f"{prefix}_{i:06d}": float(label)
    #     for i, label in enumerate(drug_labels)
    # }

    # Build label map with sample IDs like "train_000123"
    label_map = {
        f"{prefix}_{i:06d}": float(drug_labels[j])
        for j, i in enumerate(valid_indices)
    }

    # print("Label map built for drug:", label_map)
    return label_map, drug_index


def _extract_single_gene(gene, gene_dir, data_partition):
    # TODO: Try with different paths for compressing
    zst_path = os.path.join(gene_dir, data_partition, f"{gene}.tar.zst")
    gene_folder = os.path.join(gene_dir, data_partition, gene)

    print("Gene folder:", gene_folder)

    if not os.path.exists(zst_path):
        print(f"[WARNING] File not found: {zst_path}")
        return None

    if os.path.exists(gene_folder):
        print(f"[INFO] Folder already exists: {gene_folder} — skipping extraction.")
    else:
        # Step 1: Create the gene folder
        os.makedirs(gene_folder, exist_ok=True, parents=True)

        # Step 2: Extract into that folder
        print(f"[INFO] Extracting: {zst_path} into {gene_folder}")
        try:
            subprocess.run([
                "tar",
                "--use-compress-program=unzstd",
                "-xvf",
                zst_path,
                "-C",
                gene_folder
            ], check=True)
            print(f"[INFO] Extraction complete: {gene_folder}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Extraction failed for {zst_path}: {e}")
            return None
        
    # TEMP FIX, REMOVE THIS LATER
    relative_gene_dir = gene_dir.lstrip("/")  # if using UNIX paths
    gene_folder = os.path.join(gene_folder, relative_gene_dir, gene)
    ###

    return gene_folder


def extract_genes(drug, gene_dir="/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/", data_partition="train", max_workers=4):
    gene_list = DRUG_TO_LOCI[drug]
    extracted_paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_single_gene, gene, gene_dir, data_partition): gene
            for gene in gene_list
        }

        for future in as_completed(futures):
            result = future.result()
            print("Result:", result)
            if result:
                extracted_paths.append(result)

    return extracted_paths

def train_test_split_data(full_dataset, train_batch_size, val_batch_size, val_frac=0.2, seed=42):
    val_len  = int(len(full_dataset)*val_frac)
    train_len= len(full_dataset)-val_len

    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len],generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,num_workers=4, pin_memory=True)

    return train_loader, val_loader


def main(args):
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(n_gpu))

    #-------------------

    memmap_dir = args.saved_embed_memmap_dir  # Folder containing memmaps
    prefix = "full"

    #-------------------
    print("Using per token embeddings for classification task.")

    if len(DRUG_TO_LOCI[args.drug]) == 1:
        print(f"Single gene drug {args.drug} selected, using per token embeddings.")
        # Load labels and build label map
        full_label_map, drug_index = build_label_map(args.phenotype_label_path, args.drug, prefix=prefix)

        gene = DRUG_TO_LOCI[args.drug][0]
        print(f"Using gene: {gene}")

        # Load meta file paths
        meta_paths = sorted(glob.glob(f"{memmap_dir}/{gene}/*_{args.embed_type}_meta.npz"))
        print(f"Found {len(meta_paths)} meta files")

        # Construct the Dataset
        if args.embed_type == 'token':
            print(f"Using {args.embed_type} embeddings")
            full_dataset = TokenMemmapMap(meta_paths, full_label_map)
        elif args.embed_type == 'pca':
            print(f"Using {args.embed_type} embeddings with (k={args.pca_components})")
            full_dataset = PcaMemmapMap(meta_paths, full_label_map, k=args.pca_components)
        else:
            print(f"Using {args.embed_type} embeddings")
            full_dataset = MeanMemmapMap(meta_paths, full_label_map, embed_type=args.embed_type)

        embeds, _ = full_dataset[0]
        model_dim = embeds.shape[0]
        model_seq_len = embeds.shape[1]
        # model_dim = embeds.shape[1]
        # model_seq_len = embeds.shape[0]

        print("model dim:", model_dim)
        print("model seq len:", model_seq_len)

        assert meta_paths, "No meta files found - check path or in_dim"
    else:
        # train_gene_dirs = extract_genes(args.drug, gene_base_path, data_partition="train")
        # val_gene_dirs = extract_genes(args.drug, gene_base_path, data_partition="val")

        loci = DRUG_TO_LOCI[args.drug]  # e.g., ['inhA', 'katG']

        gene_memmap_dirs = [
            f"{memmap_dir}/{gene}/" for gene in loci
        ]

        print(f"Multiple gene drug {args.drug} selected, concatenating gene embeddings.")
        full_label_map, _ = build_label_map(args.phenotype_label_path, args.drug, prefix=prefix)

        print("Concatenating multiple genes")
        if args.embed_type == 'token':
            print(f"Using {args.embed_type} embeddings")
            full_dataset = MultiGeneConcatDataset(gene_memmap_dirs, full_label_map)
        elif args.embed_type == 'pca':
            print(f"Using {args.embed_type} embeddings with (k={args.pca_components})")
            full_dataset = PcaMultiGeneConcatDataset(gene_memmap_dirs, full_label_map, k=args.pca_components)
        else:
            print(f"Using {args.embed_type} embeddings")
            full_dataset = MeanMultiGeneConcatDataset(gene_memmap_dirs, full_label_map, embed_type=args.embed_type)
        
        print("done!")
        embeds, _ = full_dataset[0]
        model_dim = embeds.shape[0]
        model_seq_len = embeds.shape[1]

        print(f"Concatenated embedding shape: {embeds.shape} (D, L)")

    train_loader, val_loader = train_test_split_data(full_dataset, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, val_frac=args.test_split)

    # -------------------
    train_labels = np.fromiter(full_label_map.values(), dtype=np.int32)
    num_resistant = (train_labels == 1).sum()        # resistant
    num_sensitive = (train_labels == 0).sum()        # susceptible

    pos_weight = torch.tensor(num_sensitive / (num_resistant + 1e-8), dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # train_on_token_embeddings(train_loader, val_loader, args.drug, num_sensitive, num_resistant, criterion, args.learning_rate, args.weight_decay, args.output_path, args.saved_model_path, model_name=args.model_name, model_dim=model_dim, model_seq_len=model_seq_len, k_folds=5, epochs=args.num_epochs, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, random_seed=args.random_seed, device=device)

    # -----------------------------------------------
    # Stratified train/test split (matching training procedure) - FAST VERSION
    # -----------------------------------------------

    train_idx, test_idx, train_labels, test_labels = stratified_split_dataset(
                                                        full_dataset=full_dataset,
                                                        label_dict=full_label_map,
                                                        test_size=args.test_split,
                                                        seed=args.random_seed
                                                    )
    
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    print(f"\nTraining samples: {len(train_dataset)},  Test samples: {len(test_dataset)}")
    # train_labels = dataset_labels[train_idx]
    # test_labels = dataset_labels[test_idx]

    # Get all labels in consistent order
    all_labels = np.array(list(full_label_map.values()))

    # Get labels for train and test indices
    train_labels = all_labels[train_idx]
    test_labels = all_labels[test_idx]

    print(f"Training set: {np.sum(train_labels == 0)} R, {np.sum(train_labels == 1)} S")
    print(f"Test set: {np.sum(test_labels == 0)} R, {np.sum(test_labels == 1)} S")

    # train_on_token_embeddings(train_loader,
    #             val_loader,
    #             args.drug,
    #             num_sensitive,
    #             num_resistant,
    #             criterion,
    #             args.learning_rate,
    #             args.weight_decay,
    #             args.output_path,
    #             args.saved_model_path,
    #             model_name=args.model_name,
    #             model_dim=model_dim,
    #             model_seq_len=model_seq_len,
    #             k_folds=5,
    #             epochs=args.num_epochs,
    #             train_batch_size=args.train_batch_size, 
    #             val_batch_size=args.val_batch_size,
    #             freeze_bias_frac=0.25,
    #             random_seed=42,
    #             device='cuda')

    cross_val_train_on_token_embeddings(
        train_dataset,
        args.drug,
        num_sensitive,
        num_resistant,
        criterion,
        args.learning_rate,
        args.weight_decay,
        args.output_path,
        args.saved_model_path,
        model_name=args.model_name,
        model_dim=model_dim,
        model_seq_len=model_seq_len,
        k_folds=5,
        epochs=args.num_epochs,
        train_batch_size=args.train_batch_size, 
        val_batch_size=args.val_batch_size,
        freeze_bias_frac=0.25,
        random_seed=args.random_seed,
        fold=args.fold,
        data_loader_workers=args.data_loader_workers,
        skip_completed=args.skip_completed,
        device='cuda',
        early_stopping_min_epochs=getattr(args, 'early_stopping_min_epochs', 5),
        early_stopping_patience=getattr(args, 'early_stopping_patience', 5),
        early_stopping_min_relative_improvement=getattr(args, 'early_stopping_min_relative_improvement', 1e-3),
        early_stopping_smoothing_window=getattr(args, 'early_stopping_smoothing_window', 3),
        use_training_loss_early_stopping=not (
            getattr(args, 'use_validation_early_stopping', False)
            or getattr(args, 'use_auc_early_stopping', False)
        ),
        use_auc_early_stopping=(
            getattr(args, 'use_validation_early_stopping', False)
            or getattr(args, 'use_auc_early_stopping', False)
        ),
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train downstream resistance task model')
    parser.add_argument('--model_name', type=str, default="DNABERTCNN", help='Model name')
    parser.add_argument('--saved_embed_memmap_dir', type=str, default='/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/full', help="Saved embeds directory which would be memmaps")
    parser.add_argument('--phenotype_label_path', type=str, default='training_output/zero_shot/token_embeddings_5000/dnabert2/zs_train_stacked_phenotypes.npz', help="Path to the phenotype labels npz file")
    
    
    parser.add_argument('--train_embed_name', type=str, default='zs_train_embeddings_phenotypes.npz', help="Embedding name")
    parser.add_argument('--val_embed_name', type=str, default='zs_val_embeddings_phenotypes.npz', help="Embedding name")
    parser.add_argument('--train_batch_size', type=int, default=128, help="Batch size used for training dataset")
    parser.add_argument('--val_batch_size', type=int, default=128, help="Batch size used for validating dataset")
    parser.add_argument('--test_split', type=str, default=0.2, help="Test split ratio")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train the model')
    parser.add_argument('--freeze_bias_frac', type=float, default=0.25, help='Fraction of bias to freeze in the model')
    parser.add_argument('--output_path', type=str, default='training_output/transfer_learn/classification_results', help="Directory to save the trained model")
    parser.add_argument('--saved_model_path', type=str, default='training_output/transfer_learn/saved_models', help="Directory to save the trained model")
    parser.add_argument('--random_seed', type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument('--embed_type', type=str, default='token', help="The type of embedding to use. Options: 'token', 'mean'")
    parser.add_argument('--drug', type=str, default='ISONIAZID', help="Drug to use for classification. Options: 'RIFAMPICIN', 'CIPROFLOXACIN', etc.")

    parser.add_argument('--use_pca', action='store_true', help="Whether to use PCA for dimensionality reduction on embedding dimensions")
    parser.add_argument('--pca_components', type=int, default=10, help="Number of PCA components to keep if use_pca is True")
    
    # Early stopping parameters (matching SD-CNN defaults)
    parser.add_argument('--early_stopping_min_epochs', type=int, default=5,
                        help="Minimum epochs before early stopping can trigger")
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help="Number of epochs to wait without improvement before stopping")
    parser.add_argument('--early_stopping_min_relative_improvement', type=float, default=1e-3,
                        help="Minimum relative improvement threshold (default: 0.001 = 0.1%%)")
    parser.add_argument('--early_stopping_smoothing_window', type=int, default=3,
                        help="Window size for smoothing training loss")
    parser.add_argument('--use_validation_early_stopping', action='store_true',
                        help="DEPRECATED alias: use validation-AUC early stopping instead of training-loss")
    parser.add_argument('--use_auc_early_stopping', action='store_true',
                        help="Use validation-AUC early stopping (validation split only) with resume support")
    
    args = parser.parse_args()
    main(args)
