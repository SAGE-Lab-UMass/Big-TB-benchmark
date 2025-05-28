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
        print("genotype-phenotype df files already exist, proceeding with modeling")
    else:
        print("creating genotype-phenotype df dataset")
        make_geno_pheno_dataset(config)
        print("done!\n")

    print("loading combined genotype-phenotype data")
    df_geno_pheno = load_combined_geno_pheno(config)

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
    train_indices, _ = train_test_split(all_indices, test_size=0.2, random_state=42)
    train_df = df_geno_pheno.loc[train_indices]

    print("\nloading sparse train")
    # X_sparse = sparse.load_npz(config["pkl_file_sparse_train"])

    # joblib.dump(X_sparse, "X_sparse.joblib")
    # ipdb.set_trace()
    X_sparse = joblib.load("X_sparse.joblib")

    print("\nencoding data")
    y_df, y_array = rs_encoding_to_numeric(train_df, drugs)
    y_matrix = y_df[drugs].values.astype(int)
    print("Done encoding data")

    # print("\nloading alpha file")
    # alpha_matrix = load_alpha_matrix(config["alpha_file"], y_array, df_geno_pheno)
    # print("Done loading alpha file")

    # Filter isolates with any resistance annotation
    idx = np.where(y_matrix.sum(axis=1) != -len(drugs))
    
    return X_sparse[idx].todense(), y_matrix[idx], df_geno_pheno

def get_cnn_model(input_shape, filter_size):
    kernel_size = len(BASE_TO_COLUMN)
    model = models.Sequential([
        layers.Conv2D(64, (kernel_size, filter_size), activation='relu', input_shape=input_shape),
        layers.Lambda(lambda x: tf.squeeze(x, 1)),
        layers.Conv1D(64, 12, activation='relu'),
        layers.MaxPooling1D(3),
        layers.Conv1D(32, 3, activation='relu'),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(drugs), activation='sigmoid')
    ])
    model.compile(optimizer=Adam(np.exp(-9)), loss=masked_multi_weighted_bce, metrics=[masked_weighted_accuracy])
    return model

def get_locus_name(col_name, locus_order):
    for locus in locus_order:
        if col_name.startswith(locus + ".fasta"):
            return locus
    raise ValueError(f"Locus not found in locus_order for column: {col_name}")

def build_feature_names(df_geno_pheno):
    print("\nBuilding padded feature names...")
    feature_names = []

    # Step 1: Identify all one-hot columns and the longest gene length
    one_hot_columns = [col for col in df_geno_pheno.columns if col.endswith("_one_hot")]

    # order the columns based on the locus order as the genes were passed to the model in this order (IMP!)
    one_hot_columns = sorted(
        one_hot_columns,
        key=lambda x: locus_order.index(get_locus_name(x, locus_order))
    )

    longest_gene_len = max(df_geno_pheno.iloc[0][col].shape[0] for col in one_hot_columns)

    # Step 2: Build padded feature names
    for col in one_hot_columns:
        gene = col.replace("_one_hot", "").replace(".fasta", "")
        # if gene == "ethA":
        #     gene = "ethAR"
        # if gene == "ethR":
        #     continue

        print(f"Processing gene: {gene}")
        for pos in range(longest_gene_len):
            feature_names.append(f"{gene}_{pos}")

    print(f"Longest gene length: {longest_gene_len}")
    print(f"Total features: {len(feature_names)}")
    return feature_names


def run_cnn_with_shap(config):
    # Load full data
    X_train, y_train, df_geno_pheno = load_and_filter_data(config)
    feature_names = build_feature_names(df_geno_pheno)

    print("Done building feature names.")

    # Load the model
    saved_model_path = config['saved_model_path']
    if os.path.isdir(saved_model_path):
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
    shap_values = compute_shap_values(model, X_train, feature_names, is_deep_model=True, y_train=y_train)
    print("SHAP values shape:", shap_values.shape)

    # Analyze SHAP for each drug
    summary_rows = []
    for i, drug in enumerate(drugs):
        imp_feats, below_thresh_feats = get_imp_features_by_summary(
            shap_values, feature_names, drug_index=i)

        plot_shap_summary(shap_values, feature_names, config["output_path"], drug, drug_index=i)

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

    for _, row in imp_feature_summary_df.iterrows():
        drug = row["Drug"]
        top_important_features = row["Important_features"].split(", ")
        two_most_important_below_threshold = row["Below_threshold_features"].split(", ")

        important_features = top_important_features + two_most_important_below_threshold

        get_confident_mutation_hits(
            vcf_who_map_dir, 
            important_features, 
            drug,
            output_path=output_dir,
            output_file_name=f"causal_variant_map_mar_{drug}.csv",
            embed_type="one_hot",
            has_neg_strand=has_neg_strand,
        )

if __name__ == "__main__":
    main()


# def compute_shap_values(model, X_train):
#     print("\nComputing SHAP values for CNN...")
#     # explainer = shap.Explainer(model, X_train)
#     # shap_values = explainer(X_train)
#     # return shap_values

#     # Make sure input matches model input shape
#     if len(X_train.shape) != 4:
#         raise ValueError(f"X_train must be 4D (batch, height, width, channels), but got {X_train.shape}")

#     # Select a small representative background for SHAP
#     background = X_train[np.random.choice(X_train.shape[0], size=min(100, X_train.shape[0]), replace=False)]

#     # Use DeepExplainer for CNN (Keras) model
#     # explainer = shap.DeepExplainer(model, background)
#     explainer = shap.GradientExplainer(model, background)

#     # Compute SHAP values on the same or subset of training data
#     shap_values = explainer.shap_values(X_train)

#     ipdb.set_trace()

#     return shap_values

# def get_imp_features_by_summary(shap_values, feature_names, drug_index, importance_pec_threshold=0.2, features_below_threshold=4):
#     # Collapse SHAP values across base channel (axis=1)
#     shap_array = np.abs(shap_values.values).sum(axis=1)[:, :, drug_index]  # shape: (samples, features)
#     importance = pd.Series(shap_array.mean(axis=0), index=feature_names)
#     importance_df = importance.sort_values(ascending=False).to_frame("mean_abs_shap")

#     total_importance = importance.sum()
#     mask = (importance / total_importance) > importance_pec_threshold
#     important_features = importance[mask].index.tolist()
#     two_most_important_below_threshold = importance[~mask].index[:features_below_threshold].tolist()

#     return important_features, two_most_important_below_threshold, importance_df

# def plot_shap_summary(shap_values, feature_names, output_dir, drug_name, drug_index):
#     print(f"Plotting SHAP summary for {drug_name}...")
#     shap_vals_for_drug = shap.Explanation(
#         values=shap_values.values[:, :, drug_index],
#         base_values=shap_values.base_values[:, drug_index],
#         data=shap_values.data,
#         feature_names=feature_names
#     )

#     plt.figure()
#     shap.plots.bar(shap_vals_for_drug, show=False)
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f"shap_summary_{drug_name}.png"), bbox_inches='tight')
#     plt.close()