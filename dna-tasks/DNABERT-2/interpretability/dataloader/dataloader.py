import os
import csv
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import torch.utils.data as util_data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from dataloader.utils import make_geno_pheno_pkl, load_alpha_matrix
from dataloader.locus_order import locus_order, DRUGS as drugs

import ipdb

class MultigeneMultidrugSamples(Dataset):
    def __init__(self, sequences, res_phenotypes, gene_names,
    drug_names, num_genes):
        """
        Initialize the dataset.

        Args:
            sequences (List[List[Any]]): A list of length `num_genes`, where each element is a list of sequences
                                         for one gene across all isolates.
            res_phenotypes (List[List[Any]]): A list of length `num_drugs`, where each element is a list of resistance phenotypes
                                         for one drug across all isolates.
            num_genes (int): Expected number of genes.
        """
        assert len(sequences) == num_genes
        
        self.sequences = sequences  # Store gene sequence data
        self.res_phenotypes = res_phenotypes  # Store drug resistance phenotype data
        self.gene_names = gene_names    
        self.drug_names = drug_names

    def __len__(self):
        """
        Returns:
            int: Number of isolates in the dataset.
        """
        return len(self.res_phenotypes[0])  

    def __getitem__(self, idx):
        """
        Retrieve an isolate at the specified index.

        Args:
            idx (int): Index of the isolate to retrieve.

        Returns:
            dict: A dictionary representing one isolate, containing:
                - Gene sequences:
                    Keys: 'gene_seq_1', 'gene_seq_2', ..., up to 'gene_seq_N', where N is the number of genes.
                    Values: dna sequence for a specific gene in the isolate.

                - Resistance phenotypes:
                    Keys: 'res_phenotype_drug_1', 'res_phenotype_drug_2', ..., up to 'res_phenotype_drug_M',
                    where M is the number of drugs.
                    Values: a label (1 for susceptible, 0 for resistant) indicating the resistance status
                    of the isolate to a specific drug.

                - Gene names: List of gene names.
                - Drug names: List of drug names.
        """
        # Create a dictionary entry for each sequence (seq1 through seq14)
        # 12 - num of genes/subsequences
        # 11 - num of drugs
        item = {
            **{f'gene_seq_{i+1}': self.sequences[i][idx] for i in range(len(self.sequences))},
            **{f'res_phenotype_drug_{i+1}': self.res_phenotypes[i][idx] for i in range(len(drugs))},
            'gene_order': self.gene_names,
            'drug_order': self.drug_names
        }

        return item
    
class IGSubsetDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]  # returns the isolate dict

'''
Assumed data format:
DNA sequence_1, DNA sequence_2, ..., DNA sequence_{# genes}, phenotype label per drug
'''

def multi_gene_multi_drug_loader_csv(args, load_train=True, n_gpu=1):
    delimiter = ","

    _ = create_multidrug_classification_data(args, delimiter)

    # Load the data from CSV
    csv_filename = args.train_dataname if load_train else args.val_dataname
    print(f"loading data from {csv_filename}...")
    
    with open(os.path.join(args.datapath, csv_filename)) as csvfile:
        # headers = list(csv.reader(csvfile, delimiter=delimiter))[0]
        # data = list(csv.reader(csvfile, delimiter=delimiter))[1:]

        reader = list(csv.reader(csvfile, delimiter=delimiter))
        headers = reader[0]  # First row as headers
        data = reader[1:]    # Remaining rows as data
    
    # Identify columns with `.fasta` suffix in the header, each column represents a gene
    fasta_columns = [i for i, header in enumerate(headers) if header.endswith(".fasta")]
    gene_names = [header for header in headers if header.endswith(".fasta")]
    print(f"Number of genes: {len(fasta_columns)}")
    print(f"Number of drugs: {len(drugs)}\n")

    # Extract sequences from columns with `.fasta` suffix
    sequences = [[row[i] for row in data] for i in fasta_columns]
    sequences_array = np.array(sequences)
    print(f"sequences shape: {sequences_array.shape}\n")    # (genes, isolates)

    
    # Identify columns with drug names in the header
    drug_columns = [i for i, header in enumerate(headers) if header in drugs]
    drug_names = [header for header in headers if header in drugs]


    resistance_categories = {'R': 0, 'S': 1, '-1.0': -1, '-1': -1}
    # Extract sequences from columns with drugs
    res_phenotypes = [[row[i] for row in data] for i in drug_columns]
    res_phenotypes_array = np.array(res_phenotypes)
    print(f"res_phenotypes shape: {res_phenotypes_array.shape}\n")  # (drugs, isolates)


    # convert labels to numeric values
    print("converting resistance labels to numeric values...")
    res_phenotypes_label = [
        [resistance_categories[res] for res in res_phenotype]
        for res_phenotype in res_phenotypes
    ]
    print(f"res_phenotypes_label shape: {np.array(res_phenotypes_label).shape}")    # (genes, isolates)
    print("done!\n")


    # Create dataset and loader
    print("\ncreating multigenemultidrug dataset and loader for dnabert-S...")
    dataset = MultigeneMultidrugSamples(sequences, res_phenotypes_label, gene_names=gene_names,
    drug_names=drug_names, num_genes=len(fasta_columns))

    # Initialize the data loaders
    batch_size = args.train_batch_size if load_train else args.val_batch_size
    loader = util_data.DataLoader(dataset, batch_size=batch_size*n_gpu, shuffle=False, num_workers=4*n_gpu)
    print("done!\n")

    return loader 


def create_multidrug_classification_data(args, delimiter):
    geno_pheno_df = create_genotype_phenotype_csv(args, delimiter)
    split_data_into_train_val_sets(args, geno_pheno_df)

    return geno_pheno_df
    

def split_data_into_train_val_sets(args, geno_pheno_df, split_type='other'):
    # Split the data into train and validation sets
    geno_pheno_df = geno_pheno_df.reset_index(drop=True)

    if split_type == 'custom':
        train_indices = geno_pheno_df.query("category=='set1_original_10202'").index
        train_data = geno_pheno_df.loc[train_indices]

        test_indices = geno_pheno_df.query("category!='set1_original_10202'").index
        val_data = geno_pheno_df.loc[test_indices]

        # total number of isolates in the training and validation set
        print(f"Number of isolates in the training set: {len(train_data)}")
        print(f"Number of isolates in the validation set: {len(val_data)}")
    else:
        print("splitting the data into train and validation sets by 80/20 ratio...")
        all_indices = geno_pheno_df.index
        train_indices, val_indices = train_test_split(all_indices, test_size=args.test_split, random_state=42)
        train_data = geno_pheno_df.loc[train_indices]
        val_data = geno_pheno_df.loc[val_indices]

        # train_data = geno_pheno_df.sample(frac=split_ratio, random_state=42)
        # val_data = geno_pheno_df.drop(train_data.index)

    del geno_pheno_df
    
    print("dropping the rows which have -1 as the resistance status for all drugs...")
    train_data = train_data[train_data[drugs].apply(lambda x: (x != -1).any(), axis=1)]
    val_data = val_data[val_data[drugs].apply(lambda x: (x != -1).any(), axis=1)]
    print("done!\n")

    # total number of isolates in the training and validation set
    print(f"Number of isolates in the training set after filtering : {len(train_data)}")
    print(f"Number of isolates in the validation set after filtering : {len(val_data)}")

    # Save the train and validation data to csv
    train_data.to_csv(os.path.join(args.datapath, args.train_dataname), index=False)
    val_data.to_csv(os.path.join(args.datapath, args.val_dataname), index=False)


def create_genotype_phenotype_csv(args, delimiter):
    data_path = os.path.join(args.datapath, args.pkl_file)

    # Determine whether pickle already exists
    if os.path.isfile(data_path):
        print("genotype-phenotype df main csv file already exists, proceeding with data loading")
    else:
        print("creating genotype-phenotype df csv file...")
        make_geno_pheno_pkl(args)

    # Get data from csv
    print("\nreading in the geno_pheno df csv...")
    geno_pheno_df = pd.read_csv(data_path, delimiter=delimiter)
    print("done!\n")

    return geno_pheno_df


def select_data_subset_for_ig(dataloader, drug_index, max_samples=200, ratio_sensitive=0.2):
    """
    Selects a subset of data points for IG, with most resistant and few sensitive samples.

    Args:
        test_dataloader: the test dataloader
        drug_index: index of the drug of interest
        max_samples: total number of samples to select
        ratio_sensitive: proportion of selected samples that are sensitive (label = 1)

    Returns:
        selected_data (list): list of isolate dictionaries
    """
    resistant_samples = []
    sensitive_samples = []

    drug_key = f"res_phenotype_drug_{drug_index + 1}"

    # Collect all eligible resistant/sensitive samples
    for batch in dataloader:
        for i in range(len(batch[drug_key])):
            label = batch[drug_key][i]
            item = {key: val[i] if isinstance(val, list) else val for key, val in batch.items()}
            
            if label == 0:
                resistant_samples.append(item)
            elif label == 1:
                sensitive_samples.append(item)
            # skip -1 or missing

    # Number of sensitive samples to include
    num_sensitive = int(max_samples * ratio_sensitive)
    num_resistant = max_samples - num_sensitive

    # Subset
    selected = resistant_samples[:num_resistant] + sensitive_samples[:num_sensitive]

    print(f"Selected {len(selected)} isolates: {num_resistant} resistant, {num_sensitive} sensitive. Dropped isolates with missing labels.")

    selected_dataloader = IGSubsetDataset(selected)
    return selected


# drugs_names= ['AMIKACIN', 'CAPREOMYCIN', 'ETHAMBUTOL', 'ETHIONAMIDE', 'ISONIAZID', 'KANAMYCIN', 'LEVOFLOXACIN', 'MOXIFLOXACIN', 'PYRAZINAMIDE', 'RIFAMPICIN', 'STREPTOMYCIN']

DRUG_INDEX = {
    'AMIKACIN': 0,
    'CAPREOMYCIN': 1,
    'ETHAMBUTOL': 2,
    'ETHIONAMIDE': 3,
    'ISONIAZID': 4,
    'KANAMYCIN': 5,
    'LEVOFLOXACIN': 6,
    'MOXIFLOXACIN': 7,
    'PYRAZINAMIDE': 8,
    'RIFAMPICIN': 9,
    'STREPTOMYCIN': 10
}

def build_label_map(label_file, drug, prefix="train"):
    print(f"Loading labels from: {label_file}")
    drug_index = drugs.index(drug)

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


class TokenMemmapMap(torch.utils.data.Dataset):
    def __init__(self, meta_paths, label_dict):
        self.blocks = []
        self.lookup = []
        self.label_dict = label_dict

        for bidx, meta_path in enumerate(meta_paths):
            meta = np.load(meta_path, allow_pickle=True)
            ids = meta["identifier"]
            shape = tuple(meta["shape"])
            mmap_path = meta_path.replace("_meta.npz", ".mmap")
            mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
            self.blocks.append((ids, mm))

            # Only keep rows where ID exists in label_dict
            for r in range(shape[0]):
                seq_id = ids[r]
                if seq_id in self.label_dict:
                    self.lookup.append((bidx, r))

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        bidx, ridx = self.lookup[idx]
        ids, mm = self.blocks[bidx]
        seq_id = ids[ridx]
        x = torch.from_numpy(mm[ridx].astype("float32")).t()  # shape: (D, L)
        y = torch.tensor(self.label_dict[seq_id], dtype=torch.float32)
        return x, y
    

class PcaMemmapMap(torch.utils.data.Dataset):          # PCA‑K
    """
    Same idea but for PCA-compressed matrices.
        • in_dim = k (e.g. 10)
        • underlying shape each row: (L , k)  → we transpose to (k , L)
    """
    def __init__(self, meta_paths, label_dict, k):
        self.blocks = [] # [(ids, memmap_k), …]
        self.lookup = []
        self.k = k
        self.label_dict = label_dict

        for bidx, meta_path in enumerate(meta_paths):
            meta = np.load(meta_path, allow_pickle=True)
            ids = meta["identifier"]
            shape = tuple(meta["shape"])
            mmap_path = meta_path.replace(f"_pc{self.k}_meta.npz", f"_pc{self.k}.mmap")
            mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
            self.blocks.append((ids, mm))

            # Only keep rows where ID exists in label_dict
            for r in range(shape[0]):
                seq_id = ids[r]
                if seq_id in self.label_dict:
                    self.lookup.append((bidx, r))
        
    def __len__(self):  
        return len(self.lookup)
    
    def __getitem__(self, idx):
        bidx, ridx = self.lookup[idx]
        ids, mm = self.blocks[bidx]
        seq_id = ids[ridx]
        x = torch.from_numpy(mm[ridx].astype("float32")).t()  # shape: (K, L)
        y = torch.tensor(self.label_dict[seq_id], dtype=torch.float32)
        return x, y
    

class MeanMemmapMap(torch.utils.data.Dataset):
    """
    Loads the *_pcmean.mmap chunks produced above.
    Each item → (tensor [1,L] float32 ,  label)
    """
    def __init__(self, meta_paths, label_dict, embed_type='mean_seq'):
        self.blocks = [] # [(ids, memmap_k), …]
        self.lookup = []
        self.embed_type = embed_type
        self.label_dict = label_dict

        for bidx, meta_path in enumerate(meta_paths):
            meta = np.load(meta_path, allow_pickle=True)
            
            ids = meta["identifier"]
            shape = tuple(meta["shape"])
            mmap_path = meta_path.replace(f"_{embed_type}_meta.npz", f"_{embed_type}.mmap")

            # Check the file size
            actual_size = os.path.getsize(mmap_path)
            print(f"Actual file size of {mmap_path}: {actual_size} bytes")

            expected_size = np.prod(shape) * np.dtype("float16").itemsize
            print(f"Expected file size: {expected_size} bytes")


            mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
            self.blocks.append((ids, mm))

            print(f"Loading memmap from {mmap_path}")
            print(f"Expected shape: {shape}, total elements: {np.prod(shape)}, total bytes: {np.prod(shape) * np.dtype('float16').itemsize}")

            # Only keep rows where ID exists in label_dict
            for r in range(shape[0]):
                seq_id = ids[r]
                if seq_id in self.label_dict:
                    self.lookup.append((bidx, r))

    def __len__(self):  
        return len(self.lookup)

    def __getitem__(self, idx):
        bidx, ridx = self.lookup[idx]
        ids, mm = self.blocks[bidx]
        seq_id = ids[ridx]
        # x = torch.from_numpy(mm[ridx].astype("float32")).t() if self.embed_type == 'mean_dim' else torch.from_numpy(mm[ridx].astype("float32")) # shape: (1, L) 

        x = torch.from_numpy(mm[ridx].astype("float32")) if self.embed_type == 'mean_dim' else torch.from_numpy(mm[ridx].astype("float32")).t() # shape: (L, 1) 
        y = torch.tensor(self.label_dict[seq_id], dtype=torch.float32)
        return x, y
    

class MultiGeneConcatDataset(torch.utils.data.Dataset):
    """
    Concatenates per-gene token tensors on demand:
        genes = ["katG","inhA"]  →  x shape (768 , L_katG+L_inhA)
    """
    def __init__(self, gene_dirs, label_map):
        self.gene_dirs = gene_dirs  # list of memmap base directories (e.g., mymaps/katG, mymaps/inhA)
        self.label_map = label_map
        self.blocks = {}

        for gene_dir in gene_dirs:
            metas = sorted(Path(gene_dir).glob("*_meta.npz"))
            gene = Path(gene_dir).name
            blks = []
            for meta_path in metas:
                meta = np.load(meta_path, allow_pickle=True)
                ids = meta["identifier"].astype(str)
                shape = tuple(meta["shape"])
                mmap_path = str(meta_path).replace("_meta.npz", ".mmap")
                mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
                blks.append((ids, mm))
            self.blocks[gene] = blks

        # Extract common identifiers across all genes
        # id_sets = [set(id_ for blk in self.blocks[Path(g).name] for id_ in blk[0]) for g in gene_dirs]
        # self.ids = sorted(set.intersection(*id_sets))

        # Extract common identifiers across all genes
        id_sets = [set(id_ for blk in self.blocks[Path(g).name] for id_ in blk[0]) for g in gene_dirs]
        common_ids = set.intersection(*id_sets)

        # Keep only IDs that also exist in the filtered label map (to remove the missing -1 label samples for the drug)
        valid_ids = set(self.label_map.keys())
        self.ids = sorted(common_ids & valid_ids)


    def __len__(self):
        return len(self.ids)

    def _row(self, gene, ident):
        gene = Path(gene).name
        for ids, mm in self.blocks[gene]:
            w = np.where(ids == ident)[0]
            if w.size:
                return torch.from_numpy(mm[w[0]].astype("float32")).t()  # (320, L)
        raise KeyError(f"{ident} missing in {gene}")

    def __getitem__(self, idx):
        ident = self.ids[idx]
        parts = [self._row(gene, ident) for gene in self.gene_dirs]
        x = torch.cat(parts, dim=1)  # length-concat
        y = torch.tensor(self.label_map[ident], dtype=torch.float32)
        return x, y


class PcaMultiGeneConcatDataset(torch.utils.data.Dataset):
    """
    Returns PCA-compressed token tensors.
        genes = ["rpsL","gid"], k = 10  →
            x  shape (k , L_rpsL+L_gid)
    """
    def __init__(self, gene_dirs, label_map, k):
        self.gene_dirs = gene_dirs  # list of memmap base directories (e.g., mymaps/katG, mymaps/inhA)
        self.k = k
        self.label_map = label_map
        self.blocks = {}

        for gene_dir in gene_dirs:
            metas = sorted(Path(gene_dir).glob(f"*_pc{k}_meta.npz"))
            gene = Path(gene_dir).name
            if not metas:
                raise FileNotFoundError(f"No *_pc{k}_meta.npz for {gene}")
            blks = []
            for meta_path in metas:
                meta = np.load(meta_path, allow_pickle=True)
                ids = meta["identifier"].astype(str)
                shape = tuple(meta["shape"])
                mmap_path = str(meta_path).replace("_meta.npz", ".mmap")
                mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)

                # print(f"Loading memmap for gene {gene} from {mmap_path}")
                # print(f"Expected shape: {shape}, total elements: {np.prod(shape)}, total bytes: {np.prod(shape) * np.dtype('float16').itemsize}")


                blks.append((ids, mm))
            self.blocks[gene] = blks

        # Extract common identifiers across all genes
        id_sets = [set(id_ for blk in self.blocks[Path(g).name] for id_ in blk[0]) for g in gene_dirs]
        common_ids = set.intersection(*id_sets)

        # Keep only IDs that also exist in the filtered label map (to remove the missing -1 label samples for the drug)
        valid_ids = set(self.label_map.keys())
        self.ids = sorted(common_ids & valid_ids)


    def __len__(self): 
        return len(self.ids)

    
    def _row(self, gene, ident):
        gene = Path(gene).name
        for ids, mm in self.blocks[gene]:
            w = np.where(ids == ident)[0]
            if w.size:
                return torch.from_numpy(mm[w[0]].astype("float32")).t()  # (k, L)
        raise KeyError(f"{ident} missing in {gene}")

    def __getitem__(self, idx):
        ident = self.ids[idx]
        parts = [self._row(gene, ident) for gene in self.gene_dirs]
        x = torch.cat(parts, dim=1)  # length-concat
        y = torch.tensor(self.label_map[ident], dtype=torch.float32)
        return x, y



class MeanMultiGeneConcatDataset(torch.utils.data.Dataset):
    """
    Concatenate (k=1) mean-compressed tokens for several genes.
    Each sample → tensor shape (1 , ΣL_gene)   and a label.
    """
    def __init__(self, gene_dirs, label_map, embed_type='mean_seq'):
        self.gene_dirs = gene_dirs  # list of memmap base directories (e.g., mymaps/katG, mymaps/inhA)
        self.label_map = label_map
        self.embed_type = embed_type
        self.blocks = {}

        for gene_dir in gene_dirs:
            metas = sorted(Path(gene_dir).glob(f"*_{embed_type}_meta.npz"))
            gene = Path(gene_dir).name
            blks = []
            
            if not metas:
                raise FileNotFoundError(f"No *_{embed_type}_meta.npz for {gene}")
            
            for meta_path in metas:
                meta = np.load(meta_path, allow_pickle=True)
                ids = meta["identifier"].astype(str)
                shape = tuple(meta["shape"])
                mmap_path = Path(str(meta_path).replace(f"_{embed_type}_meta.npz", f"_{embed_type}.mmap"))

                mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)
                blks.append((ids, mm))
            self.blocks[gene] = blks

        # Extract common identifiers across all genes
        # id_sets = [set(id_ for blk in self.blocks[Path(g).name] for id_ in blk[0]) for g in gene_dirs]
        # self.ids = sorted(set.intersection(*id_sets))

        # Extract common identifiers across all genes
        id_sets = [set(id_ for blk in self.blocks[Path(g).name] for id_ in blk[0]) for g in gene_dirs]
        common_ids = set.intersection(*id_sets)

        # Keep only IDs that also exist in the filtered label map (to remove the missing -1 label samples for the drug)
        valid_ids = set(self.label_map.keys())
        self.ids = sorted(common_ids & valid_ids)

    def __len__(self): 
        return len(self.ids)
    
    def _row(self, gene, ident):
        gene = Path(gene).name
        for ids, mm in self.blocks[gene]:
            w = np.where(ids == ident)[0]
            if w.size:
                # (L,1) → (1,L)
                # return torch.from_numpy(mm[w[0]].astype("float32")).t() if self.embed_type == 'mean_dim' else torch.from_numpy(mm[w[0]].astype("float32"))
                return torch.from_numpy(mm[w[0]].astype("float32")) if self.embed_type == 'mean_dim' else torch.from_numpy(mm[w[0]].astype("float32")).t() # (1, L) -> (L, 1)
            
        raise KeyError(f"{ident} missing in {gene}")
    
    def __getitem__(self, idx):
        ident = self.ids[idx]
        parts = [self._row(gene, ident) for gene in self.gene_dirs]
        x = torch.cat(parts, dim=1)  # (1 , ΣL)
        y = torch.tensor(self.label_map[ident], dtype=torch.float32)
        return x, y
