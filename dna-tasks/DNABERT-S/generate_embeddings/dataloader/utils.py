"""
Functions for running CNN on MTB data to predict ABR phenotypes
Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang Ho Yoon
"""


import sys
import glob
import pickle
import os
import yaml
import ipdb

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold, StratifiedKFold
from Bio import SeqIO

from dataloader.locus_order import locus_order, BASE_TO_COLUMN

# Mapping to use for one-hot encoding
# BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}


# Get one hot vector
# def get_one_hot(sequence):
#     """
# 	Creates a one-hot encoding of a sequence
# 	Parameters
# 	----------
# 	sequence: iterable of str
# 		Sequence containing only ACTG- characters

# 	Returns
# 	-------
# 	np.ndarray of int
# 		L (seq len) x len(one-hot encoded sequence)
# 	"""

#     seq_len = len(sequence)
#     seq_in_index = [BASE_TO_COLUMN.get(b, b) for b in sequence]
#     one_hot = np.zeros((seq_len, len(BASE_TO_COLUMN)), dtype=int)

#     # Check if any element in seq_in_index is not an integer
#     for i, val in enumerate(seq_in_index):
#         if not isinstance(val, int):
#             raise ValueError(f"Non-integer value found at position {i}: {val}")

#     # Assign the found positions to 1
#     one_hot[np.arange(seq_len), np.array(seq_in_index)] = 1

#     return one_hot

def get_one_hot(sequence):
    """
    Creates a one-hot encoding of a sequence
    Parameters
    ----------
    sequence: iterable of str
        Sequence containing characters like ACTG- and potentially others

    Returns
    -------
    np.ndarray of int
        L (seq len) x len(one-hot encoded sequence)
    """

    # Define sequence length and initialize the one-hot matrix
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, len(BASE_TO_COLUMN)), dtype=int)
    
    # Use a list comprehension to get indices, and filter out invalid characters
    valid_indices = [(i, BASE_TO_COLUMN[b]) for i, b in enumerate(sequence) if b in BASE_TO_COLUMN]
    
    # Assign 1s to the appropriate positions
    if valid_indices:
        rows, cols = zip(*valid_indices)
        one_hot[rows, cols] = 1

    return one_hot


def sequence_dictionary(filename):
    """
	Creates a dataframe that contains the sequence of each locus for each isolate
	Note that this function splits the identifier names in the fasta file on '/'
	and takes the last entry

	Parameters
	----------
	filename: str
		path to directory containing genotype data (one fasta file containing
		sequences for all isolates at a particular locus)

	Returns
	-------
	pd.DataFrame with one column, indexed by strain name
		column name will be the beginning string of the file name
	"""
    seq_dict = SeqIO.to_dict(
        SeqIO.parse(filename, "fasta"),
        key_function=lambda x: x.id.split("/")[-1].split(".cut")[0])

    # create a dictionary of identifier: sequence
    for identifier, sequence in seq_dict.items():
        seq_dict[str(identifier)] = str(sequence.seq)

    df = pd.DataFrame.from_dict(seq_dict, orient='index')
    gene_name = filename.split("/")[-1].split("_")[0]
    df.columns = [gene_name if gene_name.endswith(".fasta") else gene_name + ".fasta"]

    return df


def make_genotype_df(genotype_input_directory):
    """
    Create a dataframe with the genotypes for each isolate at each locus
    Hard-codes the ordering of the loci so that they are in same order upon re-runs

	Parameters
	----------
	genotype_input_directory: str
		path to directory containing fasta files of genotype inputs

	Returns
	-------
	pd.DataFrame:
		indexed by isolate name, one column per locus
	"""
    # Make a df that combines all genotype data
    dfs_list = []

    for l in locus_order:
        print("reading locus", l)
        print("looking for fasta files", f"{genotype_input_directory}/{l}*.fasta")
        df_files = glob.glob(f"{genotype_input_directory}/{l}*.fasta")
        if len(df_files)==1:
            df_file = df_files[0]
        else:
            raise ValueError
        print("reading fasta file", df_file)
        _df = sequence_dictionary(df_file)
        print("found this many seqs", len(_df))
        dfs_list.append(_df)
    df_genos = dfs_list[0].join(dfs_list[1:], how='outer')
    print(f"size of df geno {len(df_genos)}\n")
    return df_genos
    

def alpha_mat(res_phenotypes_label, weight=1.0):
    """
    Creates an alpha matrix (reflects proportion of strains resistant
    (-ve)/sensitive (+ve)) that is compatible with PyTorch.

    Parameters
    ----------
    res_phenotypes_label: torch.Tensor
        Tensor of resistance values (0 for resistance, 1 for sensitivity, -1 for unknown)
        for each drug.

    weight: float
        Weight by which to multiply the sensitive class (to up or downweight
        sensitive relative to resistant strains)

    Returns
    -------
    torch.Tensor
        Weighted resistance/sensitivity values proportionate to the number of strains
        with phenotypic data.
    """
    # Get the number of drugs and strains
    # Ensure weight is a PyTorch tensor
    weight = torch.tensor(weight, dtype=torch.float32)
    res_phenotypes_label = torch.tensor(res_phenotypes_label, dtype=torch.float32)
    
    num_drugs, num_strains = res_phenotypes_label.shape 

    # Initialize alpha matrix
    alphas = torch.zeros(num_drugs, dtype=torch.float32)
    alpha_matrix = torch.zeros_like(res_phenotypes_label, dtype=torch.float32)

    for drug_index in range(num_drugs):
        # Identify resistant (0) and sensitive (1) strains, ignoring unknowns (-1)
        resistant_mask = res_phenotypes_label[drug_index, :] == 0
        sensitive_mask = res_phenotypes_label[drug_index, :] == 1
        
        # Count the number of resistant and sensitive strains
        resistant_num = torch.sum(resistant_mask).item()
        sensitive_num = torch.sum(sensitive_mask).item()
        
        print(f"Drug index {drug_index} has {resistant_num} resistant and {sensitive_num} sensitive strains")

        # Calculate alpha value for the drug, handling cases where both counts are zero
        if resistant_num + sensitive_num > 0:
            alphas[drug_index] = resistant_num / (resistant_num + sensitive_num)
        else:
            alphas[drug_index] = 0

        # Populate the alpha matrix with weighted values
        alpha_matrix[drug_index, sensitive_mask] = weight * alphas[drug_index]
        alpha_matrix[drug_index, resistant_mask] = -alphas[drug_index]
        
        print("Type of alpha matrix", type(alpha_matrix))

    return alpha_matrix



def make_geno_pheno_pkl(args):
    """
    Creates and saves a pd.DataFrame as a pkl that contains the numeric encoded
    phenotype information and the one-hot encoded sequence information for each isolate

    Required kwargs:
        phenotype_file: path to input phenotype file with columns "Isolate" and drug names
        genotype_input_directory: path to directory with input fasta files
        pkl_file: path to save the complete genotype/phenotype file
	"""

    # get table for phenotypes
    df_phenos = pd.read_csv(args.phenotype_file, index_col="Isolate", sep=",", dtype=str).fillna(-1)

    # make table of all genotypes
    df_genos = make_genotype_df(args.genotype_input_directory)

    # to save on RAM, only save genotypes for isolates for which we have phenotypes
    isolate_ids = list(df_phenos.index)
    n_strains = len(isolate_ids)
    df_genos.index = df_genos.index.astype(str)
    df_genos = df_genos.loc[df_genos.index.intersection(isolate_ids)]

    # Drop rows where we're missing the sequence for a locus
    df_genos = df_genos.dropna(axis="index")

    print(f"the shape of df geno is {df_genos.shape}\n")

    print("\ncombinining genotypes and phenotypes into a dataframe...")
    # combined dataframe of all genotypes and phenotypes
    df_geno_pheno_full = df_genos.join(df_phenos, how='inner')

    print("dumping the geno_pheno df to csv...")
    data_path = os.path.join(args.datapath, args.pkl_file)
    df_geno_pheno_full.to_csv(data_path)
    print("done!")


def create_X(df_geno_pheno):
    """
	Create an input X matrix, with output dimensions:
		n_strains x len(one-hot)=5 here x longest locus length x no. of loci

    Parameters
    ----------
    df_geno_pheno: pd.DataFrame
        generated by make_geno_pheno_pkl, contains the numeric encoded
        phenotype information and the one-hot encoded sequence information for each isolate

    Returns
    -------
    np.ndarray
        with shape N_strains, len(one_hot), L_longest_locus, N_loci
        contains the one-hot encoded sequence information for each locus for each strain
	"""

    def _get_shapes(df_geno_pheno):
        """
		Finds the length of each gene in the input dataframe
		Parameters
		----------
		df_geno_pheno: pd.Dataframe

		Returns
		-------
		dict of str: int
			length of coordinates in each column
		"""
        shapes = {}
        for column in df_geno_pheno.columns:
            if "one_hot" in column:
                shapes[column] = df_geno_pheno.loc[df_geno_pheno.index[0], column].shape[0]

        return shapes

    shapes = _get_shapes(df_geno_pheno)

    # Length of longest gene locus
    n_genes = len(shapes)
    L_longest = max(list(shapes.values()))
    L_one_hot_encoding = len(BASE_TO_COLUMN)
    print("\tfound n genes", n_genes, "and longest gene", L_longest)

    # Number of strains in model
    n_strains = df_geno_pheno.shape[0]

    # define shape of matrix - fill with zeros (effectively accomplishes padding)
    X = np.zeros((n_strains, L_one_hot_encoding, L_longest, n_genes))

    # for each strain and gene locus
    for idx, strain in enumerate(df_geno_pheno.index):
        for gene_index, gene in enumerate(shapes.keys()):
            one_hot_gene = df_geno_pheno.loc[strain, gene]
            X[idx, :, range(0, one_hot_gene.shape[0]), gene_index] = one_hot_gene

    return X


def masked_multi_weighted_bce(alpha, y_pred):
    """
	Calculates the masked weighted binary cross-entropy in multi-classification

	Parameters
	----------
	alpha: an element from the alpha matrix, a matrix of target y values weighted
		by proportion of strains with resistance data for any given drug
	y_pred: model-predicted y values
    weights: list of float (optional, default=[1., 1.])
        A list of two weights to be applied to the sensitive and resistant n_strains,
        respectively

	Returns
	-------
	scalar value of the masked weighted BCE.
	"""
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    num_not_missing = K.sum(mask, axis=-1)
    alpha = K.abs(alpha)
    bce = - alpha * y_true_ * K.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * K.log(1.0 - y_pred)
    masked_bce = bce * mask
    return K.sum(masked_bce, axis=-1) / num_not_missing


def masked_weighted_accuracy(alpha, y_pred):
    """
	Calculates the mased weighted accuracy of a model's predictions
	Parameters
	----------
	alpha: an element from the alpha matrix, a matrix of target y values weighted
		by proportion of strains with resistance data for any given drug
	y_pred: model-predicted y values

	Returns
	-------
	scalar value of the masked weighted accuracy.
	"""

    total = K.sum(K.cast(K.not_equal(alpha, 0.), K.floatx()))
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    correct = K.sum(K.cast(K.equal(y_true_, K.round(y_pred)), K.floatx()) * mask)
    return correct / total

def load_alpha_matrix(alpha_matrix_path, y_array, df_geno_pheno, args):
    """
    Loads in the alpha matrix, if file exists, otherwise creates alpha matrix

    Parameters
    ----------
    alpha_matrix_path: str
        path to alpha matrix. Will be used to save matrix if matrix does not exist

    Returns
    -------
    np.ndarray
        The alpha matrix
    """

    if os.path.isfile(alpha_matrix_path):
        print("alpha matrix already exists, loading alpha matrix...")
        alpha_matrix = alpha_matrix = torch.load("alpha_matrix.pt") # np.loadtxt(alpha_matrix_path, delimiter=',')

    else:
        if "weight_of_sensitive_class" in args:
            # print('creating alpha matrix with weight', kwargs["weight_of_sensitive_class"])
            # alpha_matrix = alpha_mat(y_array, df_geno_pheno, kwargs["weight_of_sensitive_class"])
            pass
        else:
            print("creating alpha matrix with equal weights to sensitive and resistant classes...")
            alpha_matrix = alpha_mat(y_array)
        #  np.savetxt(alpha_matrix_path, alpha_matrix, delimiter=',')
        torch.save(alpha_matrix, "alpha_matrix.pt")

    return alpha_matrix

def split_into_traintest(X_sparse, df_geno_pheno, category):
    """
    Splits the X dataframe into training and test set based on annotation in df_geno_pheno

    Parameters
    ----------
    X_sparse: sparse.COO
        a sparse-encoded np.ndarray containing genetic information for all isolates
    df_geno_pheno: pd.DataFrame
        Dataframe of genetic and phenotypic information. Contains the exact isolates in the exact order used to create X_sparse.
        Contains a column called "category" that will be used to split isolates into training and test
    category: str
        Name of the training set category

    Returns:
    -------
    sparse.COO
        a sparse-encoded np.ndarray containing genetic information for all TRAINING SET isolates
    sparse.COO
        a sparse-encoded np.ndarray containing genetic information for all TEST SET isolates
    """
    X_all = X_sparse.todense()

    df_geno_pheno = df_geno_pheno.reset_index(drop=True)

    train_indices = df_geno_pheno.query("category==@category").index
    test_indices = df_geno_pheno.query("category!=@category").index

    print("splitting X pkl")
    X_train = X_all[train_indices, :]
    X_test = X_all[test_indices, :]
    del X_all

    X_sparse_train = sparse.COO(X_train)
    sparse.save_npz(pkl_file_sparse_train, X_sparse_train, compressed=False)

    X_sparse_test = sparse.COO(X_test)
    sparse.save_npz(pkl_file_sparse_test, X_sparse_test, compressed=False)

    return X_sparse_train, X_sparse_test


def get_threshold_val(y_true, y_pred):
    """
    Compute the optimal threshold for prediction  based on the max sum of specificity and Sensitivity

    NB that we encoded R as 0, S as 1, so smaller predictions indicate higher chance of resistance

    We count falsely predicted resistance as a false positive, falsely predicted sensitivity as a false negative

    Parameters
    ----------
    y_true: np.array
        Actual labels for isolates
    y_pred: np.array
        Predicted labels for isolates

    Returns
    -------
    dict of str -> float with entries:
        sens: sensitivity at chosen threshold
        spec: specificity at chosen threshold
        threshold: chosen threshold value
    """

    num_samples = y_pred.shape[0]
    fpr_ = []
    tpr_ = []
    thresholds = np.linspace(0, 1, 101)
    num_sensitive = np.sum(y_true==1)
    num_resistant = np.sum(y_true==0)
    for threshold in thresholds:

        fp_ = 0 # number of false positives
        tp_ = 0 # number of true positives

        for i in range(num_samples):
            # If y is predicted resistant
            if (y_pred[i] < threshold):
                if (y_true[i] == 1): fp_ += 1
                if (y_true[i] == 0): tp_ += 1

        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    # valid_inds = np.where(fpr_ <= 0.1)
    valid_inds = np.arange(101)
    sens_spec_sum = (1 - fpr_) + tpr_
    best_sens_spec_sum = np.max(sens_spec_sum[valid_inds])
    best_inds = np.where(best_sens_spec_sum == sens_spec_sum[valid_inds])

    if best_inds[0].shape[0] == 1:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))
    else:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))[-1]

    return {'threshold': np.squeeze(thresholds[valid_inds][best_sens_spec_ind]),
            'spec': 1 - fpr_[valid_inds][best_sens_spec_ind],
            'sens': tpr_[valid_inds][best_sens_spec_ind]}
