import os
import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data as util_data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from dataloader.utils import make_geno_pheno_pkl, load_alpha_matrix
from dataloader.locus_order import locus_order, drugs

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
        # 13 - num of genes/subsequences
        # 12 - num of drugs
        item = {
            **{f'gene_seq_{i+1}': self.sequences[i][idx] for i in range(len(self.sequences))},
            **{f'res_phenotype_drug_{i+1}': self.res_phenotypes[i][idx] for i in range(len(drugs))},
            'gene_order': self.gene_names,
            'drug_order': self.drug_names
        }

        return item
    

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
