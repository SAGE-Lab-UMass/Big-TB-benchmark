import os
import ipdb
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from parameters.locus_order import drugs


def create_output_dir(output_dir, drug):
    """
    Create output directory if it does not exist

    Parameters
    ----------
    output_dir: str
        Path to output directory

    drug: str
        Name of the drug to create a subdirectory for

    Returns
    -------
    str
        Path to the created output directory
    """
    # Create the base output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the drug-specific subdirectory
    # if drug not upper case, convert it to uppercase letters
    output_path = os.path.join(output_dir, drug.upper())
    saved_models_path = os.path.join(output_path, "saved_models")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(saved_models_path, exist_ok=True)
    
    return output_path, saved_models_path


def correct_shap_value_shape(shap_values, feature_names):
    # Collapse over base channels (axis=0 → A, T, G, C, -)
    shap_abs = np.abs(shap_values)        # shape: (samples, bases, positions, loci, drugs)
    # shap_mean = shap_abs.mean(axis=0)     # mean over samples → shape: (bases, positions, loci, drugs)
    shap_collapsed = shap_abs.sum(axis=1)  # sum over bases → shape: (samples, positions, loci, drugs)
    # shape: (10, 10241, 12, 13)

    # Get SHAP values for one drug
    # drug_index = 0  # or whichever drug you're analyzing
    # shap_for_drug = shap_collapsed[:, :, drug_index]  # shape: (10241, 12)

    # # Flatten to 1D
    # shap_1d = shap_for_drug.T.flatten()  # shape: (11, 10241) → (10241 * 12,)
    # print("Feature names shape:", len(feature_names))

    # assert len(feature_names) == shap_1d.shape[0]

    # Transpose to shape: (samples, loci, positions, drugs)
    shap_transposed = np.transpose(shap_collapsed, (0, 2, 1, 3))  # shape: (10, 12, 10241, 13)

    # Reshape to (samples, positions * loci, drugs)
    n_samples, n_loci, n_pos, n_drugs = shap_transposed.shape
    shap_all_drugs = shap_transposed.reshape(n_samples, n_loci * n_pos, n_drugs)  # shape: (10, 12 * 10241, 13)
    print("SHAP all drugs shape:", shap_all_drugs.shape)

    assert len(feature_names) == shap_all_drugs.shape[1]

    # Create a SHAP Explanation object
    expl = shap.Explanation(
        values=shap_all_drugs.mean(axis=0)[:, 0],  # mean over samples
        base_values=None,
        data=None,
        feature_names=feature_names
    )

    # Plot bar chart
    # Plot and save
    plt.figure()
    shap.plots.bar(expl, show=False)
    plt.savefig("shap_summary_drug0.png", bbox_inches="tight")
    plt.close()

    return shap_all_drugs

def select_representative_samples(X, n_samples=30):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    X_flat = X.reshape(X.shape[0], -1)
    X_pca = PCA(n_components=min(50, X_flat.shape[1])).fit_transform(X_flat)
    kmeans = KMeans(n_clusters=n_samples, random_state=42).fit(X_pca)
    _, indices = np.unique(kmeans.labels_, return_index=True)
    return X[indices]


def select_R_representative_samples(X_train, y_train, n_samples=100, min_per_drug=7, random_state=42):
    """
    Selects representative samples for SHAP:
    - At least `min_per_drug` samples per drug where the label is 1.
    - Each sample must have at least one positive label.
    - Total number of samples = n_samples

    Args:
        X_train (np.ndarray): Input features, shape (N, ...)
        y_train (np.ndarray): Binary labels, shape (N, D)
        n_samples (int): Total number of samples to return
        min_per_drug (int): Minimum number of samples with 1 for each drug
        random_state (int): Random seed

    Returns:
        np.ndarray: X_subset of shape (n_samples, ...)
    """
    np.random.seed(random_state)

    selected_indices = set()
    num_drugs = y_train.shape[1]

    for drug_idx in range(num_drugs):
        positive_indices = np.where(y_train[:, drug_idx] == 1)[0]
        if len(positive_indices) < min_per_drug:
            raise ValueError(f"Not enough positive samples for drug {drug_idx}: only {len(positive_indices)} found.")
        chosen = np.random.choice(positive_indices, size=min_per_drug, replace=False)
        selected_indices.update(chosen)

    selected_indices = list(selected_indices)

    # Filter to only samples with at least one label = 1
    valid_indices = np.where(y_train.sum(axis=1) > 0)[0]
    remaining_pool = list(set(valid_indices) - set(selected_indices))

    # Fill up to n_samples
    remaining_needed = n_samples - len(selected_indices)
    if remaining_needed > 0:
        extra_indices = np.random.choice(remaining_pool, size=remaining_needed, replace=False)
        selected_indices.extend(extra_indices)

    # Final selection
    selected_indices = np.array(selected_indices[:n_samples])
    return X_train[selected_indices]



def compute_shap_values(model, X_train, feature_names, is_deep_model=False, y_train=None):
    """
    Compute SHAP values using the provided model and training data.
    Uses DeepExplainer/GradientExplainer for deep models, and Tree/Linear for others.
    """
    print("\nComputing SHAP values...")

    print("y_train shape:", y_train.shape)


    if is_deep_model:
        if len(X_train.shape) != 4:
            raise ValueError(f"X_train must be 4D (batch, height, width, channels), got {X_train.shape}")

        # TODO: Choose this as a representative set too
        X_explain = X_train[np.random.choice(X_train.shape[0], size=min(70, X_train.shape[0]), replace=False)]
        
        # X_background = select_representative_samples(X_train, n_samples=100)
        X_background = select_R_representative_samples(X_train, y_train, n_samples=100)
        print(f"Subset of data created with shape: {X_background.shape}")
        print(f"Background data created")

        explainer = shap.DeepExplainer(model, X_background)
        print(f"Explainer created")

        shap_values_all_dims = explainer.shap_values(X_explain)
        print(f"SHAP values computed")

        shap_values = correct_shap_value_shape(shap_values_all_dims, feature_names)      

        # view_shap(shap_values, df_geno_pheno)
        # shap.image_plot(shap_values=shap_values, 
        #             show=True)   

        # return shap.Explanation(
        #     values=np.stack(shap_values, axis=-1),
        #     base_values=explainer.expected_value,
        #     data=X_train
        # )
    else:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

    return shap_values
    


def get_imp_features_by_summary(shap_values, feature_names, drug_index=None, importance_pec_threshold=0.2, features_below_threshold=10):
    """
    Extract important features from SHAP values.

    Args:
        shap_values: SHAP values (can be Explanation or list from DeepExplainer)
        feature_names: list of feature names
        drug_index: index of drug (for multi-output models)
        importance_pec_threshold: threshold (%) for considering features important
        features_below_threshold: number of "almost important" features to return

    Returns:
        important_features, below_threshold_features, importance_df, ordered_columns
    """
    if drug_index is not None:
        # CNN: shap_values is a list (samples x features x classes)
        # Select SHAP for this drug only: shape: (samples, 11 * 10241, drugs)
        shap_array = np.abs(shap_values[:, :, drug_index])
    else:
        # Logistic regression / linear: shap_values is a SHAP Explanation object
        shap_array = np.abs(shap_values.values)  # shape: (samples, features)

    importance = pd.Series(shap_array.mean(axis=0), index=feature_names)

    importance_df = importance.to_frame("mean_abs_shap").sort_values("mean_abs_shap", ascending=False)

    important_features, ten_most_important_below_threshold = get_feature_contributions(importance_df)

    return important_features, ten_most_important_below_threshold


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


def plot_shap_summary(shap_values, feature_names, output_dir, drug_name, drug_index=None):
    """
    Plot SHAP summary bar chart.

    Args:
        shap_values: SHAP values (Explanation or list)
        feature_names: List of feature names
        output_dir: Directory to save plot
        drug_name: Name of the drug
        drug_index: If provided, selects output for that drug
    """
    print(f"Plotting SHAP summary for {drug_name}...")

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"shap_summary_{drug_name}_less.png")

    print("Shape value shape passed:", shap_values.shape)

    if drug_index is not None:
        values = shap_values.mean(axis=0)[:, drug_index]
        base_values = None
        data = None
    else:
        values = shap_values.values
        base_values = shap_values.base_values
        data = shap_values.data

    shap_exp = shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=feature_names
    )

    plt.figure()
    shap.plots.bar(shap_exp, show=False)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"Saved SHAP plot to {plot_path}")



# def view_shap(shap_values, df_geno_pheno):
#     sample_index = 1
#     drug_index = 0
#     # SHAP values shape: (samples, bases, seq_len, loci, drugs)
#     shap_sample = shap_values[sample_index, :, :, :, drug_index]  # shape: (5, 10241, 11)

#     # Reshape to long format: (base, seq_pos, locus)
#     base_names = ["A", "T", "G", "C", "-"]
#     loci_names = [col.replace("_one_hot", "").replace(".fasta", "") 
#                 for col in df_geno_pheno.columns if col.endswith("_one_hot")]

#     # Convert to long-format DataFrame
#     records = []
#     for b in range(shap_sample.shape[0]):  # 5 bases
#         for l in range(shap_sample.shape[2]):  # 11 loci
#             gene = loci_names[l]
#             for p in range(shap_sample.shape[1]):  # 10241 positions
#                 records.append((gene, p, base_names[b], shap_sample[b, p, l]))

#     shap_long_df = pd.DataFrame(records, columns=["Gene", "Position", "Base", "SHAP_value"])

#     # export to csv
#     shap_long_df.to_csv(f"shap_long_df_{sample_index}.csv", index=False)

