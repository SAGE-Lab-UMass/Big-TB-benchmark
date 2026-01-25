"""
Code for running CNN on MTB data to predict ABR phenotypes
Updated for parquet + HDF5 unified data format
Authors:
    Michael Chen (original version)
    Anna G. Green
    Chang-ho Yoon
    Updated by: Saishradha Mohanty (2025)
"""

import os, glob, yaml, h5py, sparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from Bio import SeqIO
from sklearn.model_selection import KFold, StratifiedKFold
import ipdb

# ------------------------------------------------------------
# 1️⃣ Constants and mappings
# ------------------------------------------------------------
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}

# LOCUS_ORDER = [
#     "acpM-kasA", "gid", "rpsA", "clpC", "embCAB", "aftB-ubiA", "rrs-rrl",
#     "ethAR", "oxyR-ahpC", "tlyA", "KatG", "rpsL", "rpoBC",
#     "FabG1-inhA", "eis", "gyrBA", "panD", "pncA"
# ]

LOCUS_ORDER = [
    "gyrB", 
    "gyrA", 
    "rpoB", # done1 - dim - seq - mdim - ip
    "rpoC", # done1 - dim - seq - mdim
    "rpsL", # done1 - dim - seq - mdim
    "fabG1",
    "inhA", #done1 - dim - seq - mdim - ip
    "rrs", # done1 - dim - seq - mdim
    "rrl", # done1 - dim - seq - mdim
    "tlyA", # done1 - dim - seq - mdim
    "katG", #done1 - dim - seq - mdim - ip
    "pncA", #done1 - dim - seq - mdim
    "eis", # done1 - dim - seq - mdim
    "embC", #done1 - dim - seq - mdim
    "embA", # done1 - dim - seq - mdim
    "embB", #done1 - dim - seq - mdim
    "ethA", # done1 - dim - seq - mdim
    "ethR", # done1 - dim - seq - mdim
    "gid" # done1 - dim - seq - mdim
]

DRUG_TO_LOCI = {
    'ISONIAZID': ['inhA', 'katG'], # resource error
    'RIFAMPICIN': ['rpoB', 'rpoC'], # ran
    'ETHAMBUTOL': ['embC', 'embA', 'embB'], # ran
    'PYRAZINAMIDE': ['pncA'], # ran
    'STREPTOMYCIN': ['rpsL', 'rrs', 'gid'], # ran
    'KANAMYCIN': ['rrs'], # ran
    'AMIKACIN': ['rrs', 'eis'], # ran
    'CAPREOMYCIN': ['rrs', 'rrl', 'tlyA'], # ran
    'LEVOFLOXACIN': ['gyrB', 'gyrA'], # only single class issue for AUC - fix it
    'MOXIFLOXACIN': ['gyrB', 'gyrA'], # ran
    'ETHIONAMIDE': ['inhA', 'ethA', 'ethR'], # ran
    # Add more drugs and their corresponding loci as needed
} 

# ------------------------------------------------------------
# One-hot encoding and genotype assembly
# ------------------------------------------------------------
def get_one_hot(sequence):
    """Return one-hot encoding of a DNA sequence.
    Shape: (L_seq, len(one hot encoding letters))
    """
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, len(BASE_TO_COLUMN)), dtype=int)
    for i, base in enumerate(sequence):
        if base in BASE_TO_COLUMN:
            one_hot[i, BASE_TO_COLUMN[base]] = 1
    return one_hot


def sequence_dictionary(filename):
    """Load one FASTA file and return a DataFrame with isolate IDs and sequences."""
    seq_dict = SeqIO.to_dict(
        SeqIO.parse(filename, "fasta"),
        key_function=lambda x: x.id.split("/")[-1].split(".cut")[0]
    )

    # create a dictionary of identifier: sequence
    for identifier, sequence in seq_dict.items():
        seq_dict[str(identifier)] = str(sequence.seq)

    df = pd.DataFrame.from_dict(seq_dict, orient='index')
    gene_name = filename.split("/")[-1].split("_")[0]
    df.columns = [gene_name]

    return df

def make_genotype_df(genotype_input_directory, drug, drug_to_loci=DRUG_TO_LOCI):
    """Join all loci into one DataFrame (rows = isolates, columns = loci)."""
    drug_locus_order = drug_to_loci.get(drug, [])

    if not drug_locus_order:
        raise ValueError(f"No loci found for drug: {drug}")
    
    dfs = []
    for locus in drug_locus_order:
        print("Looking for fasta files", f"{genotype_input_directory}/{locus}*.fasta")
        fasta_files = glob.glob(os.path.join(genotype_input_directory, f"{locus}*.fasta"))
        if not fasta_files:
            print(f"Missing FASTA for {locus}")
            continue
        df = sequence_dictionary(fasta_files[0])
        dfs.append(df)
    df_genos = dfs[0].join(dfs[1:], how='outer')
    print(f"Created genotype DataFrame with {df_genos.shape[0]} isolates and {len(dfs)} loci.")
    return df_genos

# ------------------------------------------------------------
# Phenotype encoding
# ------------------------------------------------------------
def rs_encoding_to_numeric(df_geno_pheno, drugs_list):
    """Convert R/S/-1 to numeric {R:0, S:1, missing:-1} encoding."""
    if isinstance(drugs_list, str):
        drugs_list = [drugs_list]

    y_all_rs = df_geno_pheno[drugs_list].fillna('-1').astype(str)
    mapping = {'R': 0, 'S': 1, '-1.0': -1, '-1': -1}
    y_all = y_all_rs.replace(mapping)
    y_all.index = list(range(0, y_all.shape[0]))
    return y_all, y_all.values


# ------------------------------------------------------------
# 4️⃣ Alpha matrix (class weighting)
# ------------------------------------------------------------
# def alpha_mat(subset_y, df_geno_pheno, weight=1.):
#     """Compute per-drug alpha matrix (sensitivity/resistance weights)."""
#     num_drugs = subset_y.shape[1]
#     y_cnn = subset_y
#     alphas = np.zeros(num_drugs, dtype=float)
#     alpha_matrix = np.zeros_like(y_cnn, dtype=float)

#     for drug in range(num_drugs):
#         resistant = np.sum(y_cnn[:, drug] == 0)
#         sensitive = np.sum(y_cnn[:, drug] == 1)
#         total = resistant + sensitive
#         if total == 0:
#             continue
#         alphas[drug] = resistant / float(total)
#         alpha_matrix[:, drug][y_cnn[:, drug] == 1] = weight * alphas[drug]
#         alpha_matrix[:, drug][y_cnn[:, drug] == 0] = -alphas[drug]

#     return alpha_matrix


# def alpha_mat(subset_y, df_geno_pheno, weight=1.0, drug_name=None):
#     """
#     Create alpha matrix for a single drug (reflects proportion of resistant/sensitive strains).

#     Parameters
#     ----------
#     subset_y : np.ndarray
#         2D array (N × 1) containing numeric resistance values for the given drug:
#         0 = Resistant, 1 = Sensitive, -1 = Missing
#     df_geno_pheno : pd.DataFrame
#         Dataframe containing genotype/phenotype info (used only for consistency/logging)
#     weight : float
#         Weight multiplier for sensitive strains.
#     drug_name : str, optional
#         The drug name for logging (not required, but useful for debugging).

#     Returns
#     -------
#     np.ndarray
#         Alpha matrix of shape (N, 1)
#     """
#     if subset_y.ndim == 1:
#         subset_y = subset_y.reshape(-1, 1)

#     # Mask missing data
#     valid_idx = np.where(subset_y != -1)[0]
#     y_valid = subset_y[valid_idx]

#     resistant_num = np.sum(y_valid == 0)
#     sensitive_num = np.sum(y_valid == 1)

#     if resistant_num + sensitive_num == 0:
#         raise ValueError("No valid phenotype data found for this drug!")

#     # Compute class ratio (proportion of resistant)
#     alpha_val = resistant_num / float(resistant_num + sensitive_num)
#     if drug_name:
#         print(f"📊 {drug_name}: {resistant_num} resistant, {sensitive_num} sensitive (α={alpha_val:.3f})")

#     # Initialize full alpha matrix (zeros where missing)
#     alpha_matrix = np.zeros_like(subset_y, dtype=np.float32)

#     # Assign weights
#     alpha_matrix[np.where(subset_y == 1.0)] = weight * alpha_val   # Sensitive → positive
#     alpha_matrix[np.where(subset_y == 0.0)] = -alpha_val           # Resistant → negative

#     return alpha_matrix


def alpha_mat(subset_y, df_geno_pheno, weight=1.0, drug_name=None):
    """
    Create alpha matrix for a single drug (reflects proportion of resistant/sensitive strains). NaN fixed
    """ 
    if subset_y.ndim == 1:
        subset_y = subset_y.reshape(-1, 1)

    # Mask missing (-1)
    valid_idx = np.where(subset_y != -1)[0]
    y_valid = subset_y[valid_idx]

    resistant_num = np.sum(y_valid == 0)
    sensitive_num = np.sum(y_valid == 1)

    if resistant_num + sensitive_num == 0:
        raise ValueError("No valid phenotype data found for this drug!")

    alpha_val = resistant_num / float(resistant_num + sensitive_num)
    if drug_name:
        print(f"{drug_name}: {resistant_num} resistant, {sensitive_num} sensitive (α={alpha_val:.3f})")

    # Initialize alpha matrix (same shape as y)
    alpha_matrix = np.zeros_like(subset_y, dtype=np.float32)
    alpha_matrix[np.where(subset_y == 1.0)] = weight * alpha_val   # sensitive → +α
    alpha_matrix[np.where(subset_y == 0.0)] = -alpha_val           # resistant → −α
    alpha_matrix[np.where(subset_y == -1.0)] = 0.0                 # missing → 0

    # Safety: replace NaNs
    alpha_matrix = np.nan_to_num(alpha_matrix, nan=0.0)

    return alpha_matrix



def load_alpha_matrix(alpha_matrix_path, y_array, df_geno_pheno, **kwargs):
    """Load or create alpha matrix."""
    if os.path.isfile(alpha_matrix_path):
        print("alpha matrix already exists, loading alpha matrix")
        alpha_matrix = np.loadtxt(alpha_matrix_path, delimiter=',')
    else:
        print("creating alpha matrix for single-drug setup")
        weight = kwargs.get("weight_of_sensitive_class", 1.0)
        alpha_matrix = alpha_mat(y_array, df_geno_pheno, weight, drug_name=kwargs["drug"])
        np.savetxt(alpha_matrix_path, alpha_matrix, delimiter=',')

    return alpha_matrix


# ------------------------------------------------------------
# 5️⃣ Parquet + HDF5 unified data loading
# ------------------------------------------------------------
# def make_geno_pheno_dataset(**kwargs):
#     """
#     Create and save phenotype (Parquet) and one-hot genotype (HDF5) datasets.
#     """
#     metadata_path = kwargs["metadata_path"]
#     h5_path = kwargs["h5_path"]

#     df_phenos = pd.read_csv(kwargs["phenotype_file"], index_col="Isolate", sep=",", dtype=str).fillna("-1")
#     df_genos = make_genotype_df(kwargs["genotype_input_directory"])
#     df_genos.index = df_genos.index.astype(str)

#     isolates = df_phenos.index.intersection(df_genos.index)
#     df_genos = df_genos.loc[isolates].dropna()
#     print(f"📊 Overlap isolates: {len(df_genos)}")

#     df_raw = df_genos.join(df_phenos, how="inner")
#     df_raw.to_parquet(metadata_path, index=True)
#     print(f"💾 Saved metadata to {metadata_path}")

#     with h5py.File(h5_path, 'w') as h5f:
#         for isolate_id, row in df_genos.iterrows():
#             for locus, seq in row.items():
#                 try:
#                     onehot = get_one_hot(seq)
#                     h5f.create_dataset(f"{isolate_id}/{locus}", data=onehot, compression='gzip')
#                 except Exception as e:
#                     print(f"⚠️ Skipping {isolate_id}/{locus}: {e}")
#     print(f"💾 Saved one-hot encodings to {h5_path}")

def make_geno_pheno_dataset(**kwargs):
    """
    Create and save phenotype (Parquet) and one-hot genotype (HDF5) datasets.
    Each gene's one-hot encoding is stored as (L_seq, 5) to match SD-CNN expectation.
    """
    metadata_path = kwargs["metadata_path"]
    h5_path = kwargs["h5_path"]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    df_phenos = pd.read_csv(kwargs["phenotype_file"], index_col="New_ID", sep=",", dtype=str).fillna("-1")
    df_genos = make_genotype_df(kwargs["genotype_input_directory"], kwargs["drug"])
    df_genos.index = df_genos.index.astype(str)

    isolates = df_phenos.index.intersection(df_genos.index)
    df_genos = df_genos.loc[isolates].dropna()
    print(f"Overlap isolates: {len(df_genos)}")

    # Save metadata (phenotypes + raw sequences)
    df_raw = df_genos.join(df_phenos, how="inner")
    df_raw.to_parquet(metadata_path, index=True)
    print(f"Saved metadata to {metadata_path}")

    # Write one-hot encodings
    with h5py.File(h5_path, 'w') as h5f:
        for isolate_id, row in df_genos.iterrows():
            for locus, seq in row.items():
                try:
                    onehot = get_one_hot(seq)
                    h5f.create_dataset(f"{isolate_id}/{locus}", data=onehot, compression='gzip')
                except Exception as e:
                    print(f"Skipping {isolate_id}/{locus}: {e}")
    print(f"Saved one-hot encodings to {h5_path}")


def load_combined_geno_pheno(**kwargs):
    """
    Load combined phenotype + genotype data from Parquet + HDF5.
    Returns DataFrame with per-locus one-hot features attached.
    """
    metadata_path = kwargs["metadata_path"]
    h5_path = kwargs["h5_path"]
    

    df_meta = pd.read_parquet(metadata_path)
    with h5py.File(h5_path, 'r') as h5f:
        first_isolate = next(iter(h5f.keys()))
        loci = list(h5f[first_isolate].keys())

    df_onehot = pd.DataFrame(index=df_meta.index, columns=[f"{l}_one_hot" for l in loci], dtype="object")
    with h5py.File(h5_path, 'r') as h5f:
        for isolate in df_meta.index:
            if isolate not in h5f:
                continue
            for locus in loci:
                try:
                    df_onehot.at[isolate, f"{locus}_one_hot"] = h5f[f"{isolate}/{locus}"][:]
                except KeyError:
                    df_onehot.at[isolate, f"{locus}_one_hot"] = None

    df_combined = pd.concat([df_meta, df_onehot], axis=1)
    print(f"Combined DataFrame shape: {df_combined.shape}")
    return df_combined


# ------------------------------------------------------------
# 6️⃣ Input matrix creation
# ------------------------------------------------------------
# def create_X(df_geno_pheno):
#     """Create model input matrix: N × 5 × L_longest × N_loci."""
#     shapes = {
#         col: df_geno_pheno.iloc[0][col].shape[0]
#         for col in df_geno_pheno.columns if "one_hot" in col
#     }
#     n_genes = len(shapes)
#     L_longest = max(shapes.values())
#     n_strains = df_geno_pheno.shape[0]

#     X = np.zeros((n_strains, len(BASE_TO_COLUMN), L_longest, n_genes))
#     for i, strain in enumerate(df_geno_pheno.index):
#         for j, (col, L) in enumerate(shapes.items()):
#             gene = df_geno_pheno.at[strain, col]
#             X[i, :, :gene.shape[0], j] = gene
#     return X

# def create_X(df_geno_pheno):
#     """
#     Create input X matrix with shape:
#         (n_strains, 5, L_longest, n_genes)
#     Each gene's one-hot matrix has shape (L_seq, 5),
#     so we transpose before insertion to match SD-CNN convention.
#     """
#     shapes = {
#         col: df_geno_pheno.iloc[0][col].shape[0]
#         for col in df_geno_pheno.columns if "one_hot" in col
#     }

#     n_genes = len(shapes)
#     L_longest = max(shapes.values())
#     n_strains = df_geno_pheno.shape[0]

#     X = np.zeros((n_strains, len(BASE_TO_COLUMN), L_longest, n_genes))
#     print(f"🧬 Building X: {n_strains} isolates × {n_genes} loci × {L_longest} bp (max length)")

#     for i, strain in enumerate(df_geno_pheno.index):
#         for j, (col, L) in enumerate(shapes.items()):
#             gene = df_geno_pheno.at[strain, col]
#             # Transpose to match SD-CNN expected layout
#             X[i, :, :gene.shape[0], j] = gene.T

#     return X


def create_X(df_geno_pheno, drug_name):
    """
    Create numpy input array X for the given drug.
    Shape: (n_strains, 5, L_longest, n_genes_selected)
    """
    # Get loci for this drug
    loci_for_drug = DRUG_TO_LOCI.get(drug_name.upper(), [])
    if not loci_for_drug:
        raise ValueError(f"No loci defined for drug '{drug_name}' in DRUG_TO_LOCI")

    # Find matching one-hot columns
    available_cols = [c for c in df_geno_pheno.columns if c.endswith("_one_hot")]
    selected_cols = []
    for locus in loci_for_drug:
        # case-insensitive match
        candidates = [c for c in available_cols if locus.lower() in c.lower()]
        if not candidates:
            print(f"Locus '{locus}' not found in DataFrame — skipping.")
            continue
        selected_cols.extend(candidates)

    if not selected_cols:
        raise ValueError(f"No matching one-hot loci found for '{drug_name}' in DataFrame")

    # Determine dimensions
    shapes = {col: df_geno_pheno.iloc[0][col].shape[0] for col in selected_cols}
    n_genes = len(shapes)
    L_longest = max(shapes.values())
    n_strains = df_geno_pheno.shape[0]

    # check order of the loci
    print(f"Loci : {shapes.keys()}")


    print(f"Building X for {drug_name}: {n_strains} isolates × {n_genes} loci × {L_longest} bp")

    # Initialize tensor
    X = np.zeros((n_strains, 5, L_longest, n_genes), dtype=np.float32)

    # Fill tensor
    for i, strain in enumerate(df_geno_pheno.index):
        for j, col in enumerate(shapes.keys()):
            one_hot_gene = df_geno_pheno.loc[strain, col].astype(np.float32)
            L = one_hot_gene.shape[0]
            # transpose to (5, L)
            X[i, :, :L, j] = one_hot_gene.T

    print(f"Created X with shape: {X.shape}")
    return X


# ------------------------------------------------------------
# 7️⃣ Loss and metrics
# ------------------------------------------------------------
def masked_multi_weighted_bce(alpha, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    num_not_missing = K.sum(mask, axis=-1)
    alpha = K.abs(alpha)
    bce = - alpha * y_true_ * K.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * K.log(1.0 - y_pred)
    masked_bce = bce * mask

    # return K.sum(masked_bce, axis=-1) / num_not_missing
    num_not_missing = K.maximum(K.sum(mask, axis=-1), K.epsilon())
    return K.sum(masked_bce, axis=-1) / num_not_missing



def masked_weighted_accuracy(alpha, y_pred):
    total = K.sum(K.cast(K.not_equal(alpha, 0.), K.floatx()))
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    correct = K.sum(K.cast(K.equal(y_true_, K.round(y_pred)), K.floatx()) * mask)
    return correct / total


# ------------------------------------------------------------
# 8️⃣ Threshold computation
# ------------------------------------------------------------
def get_threshold_val(y_true, y_pred):
    """Compute optimal threshold maximizing (sensitivity + specificity)."""
    thresholds = np.linspace(0, 1, 101)
    num_sensitive = np.sum(y_true == 1)
    num_resistant = np.sum(y_true == 0)
    fpr_, tpr_ = [], []

    for t in thresholds:
        fp = np.sum((y_pred < t) & (y_true == 1))
        tp = np.sum((y_pred < t) & (y_true == 0))
        fpr_.append(fp / float(num_sensitive))
        tpr_.append(tp / float(num_resistant))

    fpr_, tpr_ = np.array(fpr_), np.array(tpr_)
    sens_spec_sum = (1 - fpr_) + tpr_
    best_idx = np.argmax(sens_spec_sum)
    return {
        'threshold': thresholds[best_idx],
        'spec': 1 - fpr_[best_idx],
        'sens': tpr_[best_idx]
    }
