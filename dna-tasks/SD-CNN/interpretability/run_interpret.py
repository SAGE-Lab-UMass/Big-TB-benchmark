import os
import sys
import yaml
import joblib
import shap
import sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from tb_cnn_codebase import *
from parameters.locus_order import locus_order, drugs, BASE_TO_COLUMN
from map_mar import get_confident_mutation_hits
from utils import *

shap.initjs()


def load_geno_pheno_dataset(config):
    parquet_file = config["metadata_path"]
    h5_file = config["h5_path"]
    if os.path.isfile(parquet_file) and os.path.isfile(h5_file):
        print("Found existing parquet/HDF5 files.")
    else:
        print("creating genotype-phenotype df dataset")
        make_geno_pheno_dataset(**config)
        print("done!\n")

    print("loading combined genotype-phenotype data")
    df_geno_pheno = load_combined_geno_pheno(**config)
    print(f"Loaded {len(df_geno_pheno)} isolates.\n")

    return df_geno_pheno


def parse_arguments():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python script.py <parameter_file.yaml>")
    return sys.argv[1]

def load_config(input_file):
    with open(input_file, "r") as f:
        return yaml.safe_load(f)

def load_and_filter_data(config):
    print("\nloading and filtering data...")
    print("loading pkl geno-pheno file")
    # df_geno_pheno = pd.read_pickle(config["pkl_file"]).reset_index(drop=True)

    df_geno_pheno = load_geno_pheno_dataset(config)
    print("Columns in df_geno_pheno: ", df_geno_pheno.columns)

    # Assuming df_geno_pheno is your DataFrame
    # example_array = df_geno_pheno.loc[df_geno_pheno.first_valid_index(), "gid.fasta_one_hot"]

    all_indices = df_geno_pheno.index
    train_indices, test_indices = train_test_split(
        all_indices, test_size=config["test_size"], random_state=config["random_seed"]
    )

    print("\nloading sparse X matrix")
    # X_sparse = sparse.load_npz(config["pkl_file_sparse_train"])
    # X_sparse = joblib.load("X_sparse.joblib")
    X_sparse_path = config["X_sparse_path"]
    # ---------------------------------------------------
    # Check if file exists
    # ---------------------------------------------------
    if os.path.exists(X_sparse_path):
        print(f"Loading existing X_sparse from: {X_sparse_path}")
        X_sparse = sparse.load_npz(X_sparse_path)
        print(f"Loaded X_sparse with shape {X_sparse.shape} and {X_sparse.nnz} non-zero entries")

    # ---------------------------------------------------
    # Else create and save it
    # ---------------------------------------------------
    else:
        print("Creating X_sparse ...")
        X_all = create_X(df_geno_pheno, config["drug"])
        X_sparse = sparse.COO(X_all)
        del X_all  # free memory

        os.makedirs(os.path.dirname(X_sparse_path), exist_ok=True)
        sparse.save_npz(X_sparse_path, X_sparse)
        print(f"Saved new X_sparse at: {X_sparse_path}")

    print("\nencoding data")
    y_df, y_array = rs_encoding_to_numeric(df_geno_pheno, config["drug"])
    y_matrix = y_df.values.astype(int)
    y_matrix = y_matrix.reshape(-1, 1)
    print("Done encoding data")

    # Filter only the isolates with any resistance annotation
    mask_valid = (y_matrix != -1).ravel()
    # y_matrix = y_matrix[mask_valid]
    # idx = np.where(y_matrix.sum(axis=1) != -len(drugs))

    # index of the required drug from drugs list
    # drug_index = drugs.index(config["drug"])

    # select that column from y_matrix
    # y_matrix = y_matrix[:, drug_index]

    # Deduplicate data if path provided
    X_valid = X_sparse[mask_valid].todense()
    y_valid = y_matrix[mask_valid]
    df_geno_pheno_valid = df_geno_pheno[mask_valid].reset_index(drop=True)
    print(f"After filtering for valid labels, X shape: {X_valid.shape}, y shape: {y_valid.shape}")

    full_indices = dedup_and_save_indices(X_valid, y_valid, data_name="full", out_dir=config["deduplicated_data_output_path"])
    X_dedup = X_valid[full_indices]
    y_dedup = y_valid[full_indices]
    df_geno_pheno_dedup = df_geno_pheno_valid.iloc[full_indices].reset_index(drop=True)

    # return X_sparse[mask_valid].todense(), y_matrix[mask_valid], df_geno_pheno
    return X_dedup, y_dedup, df_geno_pheno_dedup

def get_cnn_model(input_shape, filter_size):
    model = models.Sequential()
    model.add(layers.Conv2D(
        64, (5, filter_size),
        activation='relu',
        data_format='channels_last',
        input_shape=input_shape.shape[1:]
    ))
    model.add(layers.Conv2D(64, (1, 12), activation='relu'))
    model.add(layers.MaxPooling2D((1, 3)))
    model.add(layers.Conv2D(32, (1, 3), activation='relu'))
    model.add(layers.Conv2D(32, (1, 3), activation='relu'))
    model.add(layers.MaxPooling2D((1, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=np.exp(-9.0))
    model.compile(optimizer=opt,
                    loss=masked_multi_weighted_bce,
                    metrics=[masked_weighted_accuracy])
    return model

def get_locus_name(col_name, locus_order):
    for locus in locus_order:
        if col_name.startswith(locus + ".fasta"):
            return locus
    raise ValueError(f"Locus not found in locus_order for column: {col_name}")

def build_feature_names(df_geno_pheno, drug):
    print("\nBuilding padded feature names...")
    feature_names = []

    # Step 1: Identify all one-hot columns and the longest gene length
    # one_hot_columns = [col for col in df_geno_pheno.columns if col.endswith("_one_hot")]
    # Step 1: Identify one-hot columns only for loci relevant to this drug
    one_hot_columns = [
        col for col in df_geno_pheno.columns
        if col.endswith("_one_hot")
        and any(col.startswith(f"{locus}") for locus in DRUG_TO_LOCI[drug])
    ]
    print(f"One-hot columns for drug {drug}: {one_hot_columns}")

    # order the columns based on the locus order as the genes were passed to the model in this order for training (IMP!)
    one_hot_columns = sorted(
        one_hot_columns,
        key=lambda x: locus_order.index(get_locus_name(x, DRUG_TO_LOCI[drug]))
    )
    print(f"Ordered one-hot columns: {one_hot_columns}")

    longest_gene_len = max(df_geno_pheno.iloc[0][col].shape[0] for col in one_hot_columns)
    print(f"Longest gene length determined: {longest_gene_len}")

    # Step 2: Build padded feature names
    for col in one_hot_columns:
        gene = col.replace("_one_hot", "").replace(".fasta", "")

        print(f"Processing gene: {gene}")
        for pos in range(longest_gene_len):
            feature_names.append(f"{gene}_{pos}")

    print(f"Longest gene length: {longest_gene_len}")
    print(f"Total features: {len(feature_names)}")
    return feature_names


def run_cnn_with_shap(config):
    drug = config["drug"]

    # Load full data
    X, y, df_geno_pheno = load_and_filter_data(config)
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    feature_names = build_feature_names(df_geno_pheno, drug)

    print("Done building feature names.")

    # Load the model
    saved_model_path = config['saved_model_path']
    if os.path.isfile(saved_model_path):
        print("\nLoading model...")
        try:
            model = models.load_model(saved_model_path, custom_objects={
                'masked_weighted_accuracy': masked_weighted_accuracy,
                'masked_multi_weighted_bce': masked_multi_weighted_bce
            })
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {saved_model_path}: {e}")
    else:
        raise FileNotFoundError(f"Model directory not found: {saved_model_path}")
    print("Model loaded successfully.")


    # Optionally predict on entire data (debugging / inspection)
    # y_pred = model.predict(X_train)
    # print("Prediction complete. Now computing SHAP values...")


    # Compute SHAP values

    # Ensure the "shap_values" subfolder exists inside the configured output path
    shap_output_dir = os.path.join(config["output_path"], "shap_values")
    os.makedirs(shap_output_dir, exist_ok=True)

    # if shap values already computed, load them
    shap_values_file = os.path.join(shap_output_dir, f"shap_values_{drug}.npy")
    if os.path.isfile(shap_values_file):
        print(f"Loading existing SHAP values from {shap_values_file}...")
        shap_values = np.load(shap_values_file)
        print("SHAP values loaded successfully.")
    else:
        print(f"Computing SHAP values for {drug}...")
        shap_values = compute_shap_for_drug(model, X, feature_names, shap_output_dir, is_deep_model=True, y=y)
        # shap_values = compute_shap_values(model, X_train, feature_names, dedup_data_path, is_deep_model=True, y_train=y_train)
        print("SHAP values shape:", shap_values.shape)
        
        # save shap values for future use
        shap_values_file = os.path.join(shap_output_dir, f"shap_values_{drug}.npy")
        np.save(shap_values_file, shap_values)
        print(f"SHAP values saved to {shap_values_file}")

    # Analyze SHAP for each drug
    summary_rows = []

    imp_feats, below_thresh_feats = get_imp_features_by_summary(
        shap_values, feature_names, drug_index=0)

    plot_shap_summary(shap_values, feature_names, config["output_path"], drug, drug_index=0)

    print(f"\nImportant features for {drug}: {imp_feats}")
    print(f"Top features below threshold for {drug}: {below_thresh_feats}")

    summary_rows.append({
        "Drug": drug,
        "Important_features": ", ".join(imp_feats),
        "Below_threshold_features": ", ".join(below_thresh_feats)
    })

    # Save and return summary
    imp_feature_summary_df = pd.DataFrame(summary_rows)
    return imp_feature_summary_df


def main():
    input_file = parse_arguments()
    config = load_config(input_file)
    output_dir = config['output_path']
    has_neg_strand = bool(config["has_neg_strand"])
    vcf_who_map_dir = config["WHO_VCF_mapped_dir"]
    model_type = config["model_type"]

    imp_feature_summary_df = run_cnn_with_shap(config)
    # ipdb.set_trace()

    drug = config["drug"]
    # top_important_features = imp_feature_summary_df["Important_features"].split(", ")
    top_important_features = imp_feature_summary_df["Important_features"].str.split(", ").sum()
    # two_most_important_below_threshold = imp_feature_summary_df["Below_threshold_features"].split(", ")
    ten_most_important_below_threshold = imp_feature_summary_df["Below_threshold_features"].str.split(", ").sum()

    important_features = top_important_features + ten_most_important_below_threshold
    print(f"Important features for {drug}: {important_features}")

    # get_confident_mutation_hits(
    #     vcf_who_map_dir, 
    #     important_features, 
    #     drug,
    #     output_path=output_dir,
    #     output_file_name=f"causal_variant_map_mar_{drug}.csv",
    #     embed_type="one_hot",
    #     has_neg_strand=has_neg_strand,
    # )

    
    # important_features, ten_most_important_below_threshold = get_important_features(drug, genotype_sites_file, input_data_file, saved_model_path, plot_dir)

    get_confident_mutation_hits(
        vcf_who_map_dir, 
        important_features, 
        ten_most_important_below_threshold, 
        drug,
        model_type=model_type,
        has_neg_strand=has_neg_strand,
        output_csv=f"{output_dir}/causal_variant_map_mar_{drug}.csv"
    )

    # print("Most important features for {drug}: ", important_features)
    # print("Most important below threshold features for {drug}: ", ten_most_important_below_threshold)


if __name__ == "__main__":
    main()

