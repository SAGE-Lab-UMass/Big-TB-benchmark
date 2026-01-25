import os
import sys
import yaml
import joblib
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import random
import ipdb

from sklearn.model_selection import train_test_split
from utils import create_output_dir
# from mutation_level_interpret_log_reg import get_confident_mutation_hits
from map_mar import get_confident_mutation_hits
from parameters.locus_order import DRUG_TO_LOCI

shap.initjs()

def parse_arguments():
    """
    Parse command-line arguments.
    Expects a single argument: the path to the YAML parameter file.
    """
    if len(sys.argv) < 2:
        raise ValueError("Usage: python script.py <parameter_file.yaml>")
    return sys.argv[1]

def load_config(input_file):
    """
    Load and parse the YAML configuration file.
    Returns a dictionary with all required parameters.
    """
    print("\nReading parameter file:")
    print("Input file:", input_file)
    with open(input_file, "r") as f:
        config = yaml.safe_load(f)
    print(config)
    print("\n")
    return config

def prepare_output_dirs(config):
    """
    Prepare and save output locations using the create_output_dir utility.
    Returns the drug-specific output directory and the model path.
    """
    drug = config["drug"]
    model_dir = config["model_dir"]
    drug_output_dir, saved_model_path = create_output_dir(model_dir, drug)
    return drug_output_dir, saved_model_path

def read_genotypes(genotype_sites_file, drug):
    """
    Read the genotypes of interest from CSV and construct genotype column names.
    """
    print("Reading in genotypes of interest...")
    genotypes_df = pd.read_csv(genotype_sites_file, index_col=0)

    selected_loci = [f"/{gene}" for gene in DRUG_TO_LOCI[drug]]
    drug_genotypes = genotypes_df[genotypes_df["locus"].isin(selected_loci)]

    genotype_columns = [f"{locus}_{site}" for locus, site in zip(drug_genotypes.locus, drug_genotypes.sites)]
    print("Done!\n")

    print(f"Length of genotype columns for drug {drug}: {len(genotype_columns)}")

    return genotype_columns

def load_input_data(input_data_file):
    """
    Load the main input data from CSV.
    """
    input_data_df = pd.read_csv(input_data_file, index_col=0, low_memory=False)
    return input_data_df


def dedup_and_save_indices(X, y, data_name="full", out_dir="dedup_geno_data"):
    """
    Deduplicate (X, y) pairs by hashing their bytes + labels.

    Supports both pandas DataFrames and numpy arrays.
    Saves the indices of unique (X, y) pairs to a .npy file for reuse.
    """

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{data_name}_dedup_indices.npy")

    # Reuse cached indices if available
    if os.path.exists(out_path):
        uniq_indices = np.load(out_path)
        print(f"[{data_name}] Loaded cached dedup indices ({len(X)} → {len(uniq_indices)})")
        return uniq_indices

    print(f"[{data_name}] Deduplicating {len(X)} samples ...")

    uniq_indices = []
    seen = set()

    # Convert to numpy arrays if DataFrame/Series
    if hasattr(X, "iloc"):
        X = X.to_numpy()
    if hasattr(y, "iloc"):
        y = y.to_numpy()

    for i in tqdm(range(len(X)), desc=f"[{data_name}]"):
        # Convert sample to bytes (for hashing)
        x_bytes = X[i].tobytes()

        # Handle y as scalar or vector safely
        y_val = np.ravel(y[i])[0] if np.ndim(y[i]) > 0 else y[i]
        y_val = int(y_val) if isinstance(y_val, (np.integer, np.bool_)) or str(y_val).isdigit() else str(y_val)

        key = (x_bytes, y_val)
        if key not in seen:
            seen.add(key)
            uniq_indices.append(i)

    uniq_indices = np.array(uniq_indices)
    np.save(out_path, uniq_indices)

    # Logging
    reduction = len(X) - len(uniq_indices)
    frac = 100.0 * reduction / len(X)
    log_msg = (
        f"[{data_name}] data deduplicated {len(X)} → {len(uniq_indices)} "
        f"({reduction} removed, {frac:.1f}% reduction)"
    )
    print(log_msg)

    with open(os.path.join(out_dir, "dedup_log.txt"), "a") as f:
        f.write(log_msg + "\n")

    return uniq_indices


def split_data(X, y, drug, test_size=0.2, random_state=42):
    """
    Perform a train-test split on the given DataFrame indices.
    Returns the train and test subsets of the original DataFrame.
    """
    all_indices = X.index
    train_indices, test_indices = train_test_split(
        all_indices, test_size=test_size, random_state=random_state, stratify=y)
    X_train_df = X.loc[train_indices]
    y_train_df = y.loc[train_indices]

    X_test_df = X.loc[test_indices]
    y_test_df = y.loc[test_indices]
    
    print("Total train samples:", X_train_df.shape)
    print("Total test samples:", X_test_df.shape)
    return X_train_df, y_train_df, X_test_df, y_test_df

def drop_missing_for_drug(df, drug, genotype_columns):
    """
    Drop missing values for the specified drug and extract the corresponding
    X (features) and Y (labels) DataFrames/Series for model fitting.
    """
    print(f"\nDropping missing values from training set for drug {drug}...")
    for_fitting = df.dropna(subset=[drug])

    X = for_fitting[genotype_columns]
    Y = for_fitting[drug]
    print("X shape after dropping NaN:", X.shape)
    print("y shape after dropping NaN:", Y.shape)
    print(f"\nNumber of samples for drug {drug}:")
    print(for_fitting.groupby(drug).size())
    return X, Y, for_fitting

def load_model(saved_model_path, drug):
    """
    Load the pre-trained Logistic Regression model for the specified drug.
    """
    print("Loading the model...")
    model_path = f"{saved_model_path}/LogisticRegression_bestC.model"
    classifier = joblib.load(model_path)
    return classifier

def compute_shap_values(classifier, X):
    """
    Compute SHAP values using the provided model and feature matrix X.
    Returns the SHAP values object.
    """
    print("Computing SHAP values...")
    print("X shape for SHAP computation:", X.shape)
    explainer = shap.LinearExplainer(classifier, X)
    shap_values = explainer(X)
    return shap_values

def compute_shap_for_drug(model, X, shap_output_path, y=None, background_frac=0.1):
    # ---- new method of importance ----
    # select 10% of training data as background
    # and explain on full - background set 
    bg_size_all = min(160, int(len(X) * background_frac))
    ex_size_all = len(X) - bg_size_all

    print("Computing SHAP values...")
    print("X background shape for SHAP computation:", X.shape)

    shap_values = compute_shap_values_strat(
        model,
        X,
        shap_output_path,
        y=y,
        seed=42,
    )

    return shap_values


import numpy as np
import pandas as pd
import shap
import random
import os
from sklearn.model_selection import train_test_split

def compute_shap_values_strat(
    model,
    X,
    shap_output_path,
    y=None,
    seed=42,
    bg_frac=0.2,
    max_bg=160,
):
    """
    Compute SHAP values using stratified selection of background and explainer samples.

    Parameters
    ----------
    model : trained model (e.g., Keras/PyTorch/Sklearn)
        The fitted model for which SHAP values will be computed.
    X : np.ndarray or pd.DataFrame
        Input feature matrix.
    shap_output_path : str
        Path to save SHAP output if needed.
    y : array-like
        Labels for stratified sampling.
    seed : int, default=42
        Random seed for reproducibility.
    bg_frac : float, default=0.2
        Fraction of samples to use as SHAP background.
    max_bg : int, default=160
        Maximum number of background samples.
    """

    print("\n=== Computing SHAP values ===")

    if y is None:
        raise ValueError("Stratification requires labels (y).")

    # Ensure numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    # y = np.array(y).reshape(-1)
    y = np.array(y).astype(int).reshape(-1)

    N = len(X)
    assert N == len(y), "X and y must have the same length"

    print(f"Total samples: {N}")
    print(f"Label distribution: {np.bincount(y)}")

    # =====================================================
    # Step 1 — Stratified background/explainer selection
    # =====================================================
    idx = np.arange(N)
    rng = np.random.default_rng(seed)

    if np.unique(y).size > 1:
        bg_idx, ex_idx = train_test_split(
            idx, train_size=bg_frac, stratify=y, random_state=seed
        )
    else:
        perm = rng.permutation(idx)
        cut = max(1, int(round(bg_frac * N)))
        bg_idx, ex_idx = perm[:cut], perm[cut:]

    # Cap background if too large
    if len(bg_idx) > max_bg:
        if np.unique(y[bg_idx]).size > 1:
            bg_idx, _ = train_test_split(
                bg_idx, train_size=max_bg, stratify=y[bg_idx], random_state=seed
            )
        else:
            bg_idx = rng.choice(bg_idx, size=max_bg, replace=False)

        mask = np.ones(N, dtype=bool)
        mask[bg_idx] = False
        ex_idx = np.where(mask)[0]

    # Ensure explainer subset is not empty
    if len(ex_idx) == 0:
        ex_idx = np.array([bg_idx[-1]])
        bg_idx = bg_idx[:-1]

    print(f"Background samples: {len(bg_idx)}")
    print(f"Explainer samples: {len(ex_idx)}")
    print(f"BG label counts: {np.bincount(y[bg_idx])}")
    print(f"EX label counts: {np.bincount(y[ex_idx])}")

    X_background = X[bg_idx]
    X_explain = X[ex_idx]
    y_explain = y[ex_idx]

    # =====================================================
    # Step 2 — Sanity check for SHAP input
    # =====================================================
    # Allow both tabular and image-style tensors
    if X_background.ndim not in (2, 4):
        raise ValueError(
            f"Unsupported X shape {X_background.shape}. "
            "Expected 2D (tabular) or 4D (image-style) input."
        )

    # =====================================================
    # Step 3 — Build SHAP Explainer
    # =====================================================
    print("Creating SHAP Explainer...")
    explainer = shap.Explainer(model, X_background)
    print("Explainer created successfully.")

    # =====================================================
    # Step 4 — Compute SHAP values
    # =====================================================
    E = min(len(X_explain), N - len(X_background))
    print(f"Explaining {E} samples from {len(X_explain)} explainer pool")
    explain_indices = random.sample(range(len(X_explain)), E)

    X_explain_subset = X_explain[explain_indices]
    y_explain_subset = y_explain[explain_indices]

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_explain_subset)
    print("SHAP values computed successfully.")

    # Optionally save
    os.makedirs(shap_output_path, exist_ok=True)
    np.save(os.path.join(shap_output_path, "shap_values.npy"), shap_values)
    np.save(os.path.join(shap_output_path, "explained_labels.npy"), y_explain_subset)

    print(f"Saved SHAP values to {shap_output_path}")

    return shap_values

    

def plot_shap_summary(shap_values, drug, plot_dir, filename="shap_summary_plot.png"):
    """
    Generate and display the SHAP summary bar plot.
    """
    print("Generating SHAP summary bar plot...")
    shap_fig = shap.plots.bar(shap_values)

    # Create the directory if it doesn't exist
    output_plot_dir = os.path.join(plot_dir, drug)
    os.makedirs(output_plot_dir, exist_ok=True)
    output_path = os.path.join(output_plot_dir, filename)

    # Create the plot
    plt.figure()
    shap.plots.bar(shap_values, show=False)

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    print(f"SHAP summary plot saved to: {output_path}")

    plt.close()


def get_feature_contributions(importance_df, importance_pec_threshold=0.2, features_below_threshold=10):
    shap_vals = importance_df["mean_abs_shap"].values
    feature_index = importance_df.index.values

    # Compute total importance once
    total_importance = shap_vals.sum()

    # Create a boolean mask for features exceeding the threshold
    mask = (shap_vals / total_importance) > importance_pec_threshold

    # 1) All features above threshold
    important_features = feature_index[mask].tolist()

    # 2) The top 10 from below threshold
    below_threshold_features = feature_index[~mask]
    ten_most_important_below_threshold = below_threshold_features[:features_below_threshold].tolist()

    return important_features, ten_most_important_below_threshold

def get_imp_features_by_summary(shap_values, genotype_columns, importance_pec_threshold=0.2):
    """
    Get the most important features by mean absolute SHAP value.
    Returns a DataFrame with the feature importance values.
    """
    shap_values_df = pd.DataFrame(shap_values, columns=genotype_columns)
    importance = np.abs(shap_values_df).mean(axis=0)
    ordered_columns = importance.sort_values(ascending=False).index.tolist()
    shap_values_df = shap_values_df[ordered_columns]

    importance_df = (
        pd.DataFrame(importance, columns=["mean_abs_shap"])
        .sort_values("mean_abs_shap", ascending=False)
    )
    important_features, ten_most_important_below_threshold = get_feature_contributions(importance_df, importance_pec_threshold)

    important_features = list(map(lambda x: x.strip("/"), important_features))
    ten_most_important_below_threshold = list(map(lambda x: x.strip("/"), ten_most_important_below_threshold))

    return important_features, ten_most_important_below_threshold


def get_important_features(drug, genotype_sites_file, input_data_file, saved_model_path, plot_dir, shap_values_dir, dedup_dir=None):
    # Read genotype columns
    genotype_columns = read_genotypes(genotype_sites_file, drug)
    
    # Load the main input data
    input_data_df = load_input_data(input_data_file)

    X = input_data_df[genotype_columns]
    y = input_data_df[drug]

    full_indices = dedup_and_save_indices(X, y, data_name="full", out_dir=dedup_dir)
    X_dedup = X.iloc[full_indices]
    y_dedup = y.iloc[full_indices]
    df_geno_pheno_dedup = input_data_df.iloc[full_indices].reset_index(drop=True)

    # Drop missing values for the target drug and extract X, Y
    X_valid, y_valid, df_geno_pheno_valid = drop_missing_for_drug(df_geno_pheno_dedup, drug, genotype_columns)
    
    # Load the pre-trained model
    classifier = load_model(saved_model_path, drug)
    
    # Compute SHAP values on the background data
    shap_values = compute_shap_for_drug(classifier, X_valid, shap_output_path=shap_values_dir, y=y_valid, background_frac=0.1)
    
    # Get important mutations by SHAP summary
    important_features, ten_most_important_below_threshold = get_imp_features_by_summary(shap_values, genotype_columns)

    # plot SHAP figures
    # plot_shap_summary(shap_values, drug, plot_dir)

    print(f"\nImportant features for drug {drug}:")
    print(important_features)

    print(f"\nTen most important features below threshold for drug {drug}:")
    print(ten_most_important_below_threshold)

    return important_features, ten_most_important_below_threshold

def main():
    # 1) Parse command-line arguments
    input_file = parse_arguments()
    
    # 2) Load configuration
    config = load_config(input_file)

    # 3) Prepare output directories
    drug_output_dir, saved_model_path = prepare_output_dirs(config)
    
    drug = config["drug"]
    genotype_sites_file = config["genotype_sites_file"]
    input_data_file = config["input_data_file"]
    output_dir = config["output_dir"]
    shap_values_dir = config["shap_values_dir"]
    WHO_VCF_mapped_dir = config["WHO_VCF_mapped_dir"]
    has_neg_strand = bool(config["has_neg_strand"])
    model_type = config["model_type"]
    plot_dir = config["plot_dir"]
    dedup_dir = config["dedup_dir"]
    output_file_name = f"confident_mutation_hits_{drug}.csv"

    important_features, ten_most_important_below_threshold = get_important_features(drug, genotype_sites_file, input_data_file, saved_model_path, plot_dir, shap_values_dir, dedup_dir=dedup_dir)

    get_confident_mutation_hits(
        WHO_VCF_mapped_dir, 
        important_features, 
        ten_most_important_below_threshold, 
        drug,
        model_type=model_type,
        has_neg_strand=has_neg_strand,
        output_csv=f"{output_dir}/{output_file_name}"
    )

    print("Most important features for {drug}: ", important_features)
    print("Most important below threshold features for {drug}: ", ten_most_important_below_threshold)



if __name__ == "__main__":
    main()
