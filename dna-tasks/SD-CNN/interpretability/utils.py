import os, random
import ipdb
from tqdm import tqdm
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
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

def dedup_and_save_indices(X, y, data_name="full", out_dir="dedup_geno_data"):
    """
    Deduplicate (X, y) pairs by hashing their bytes + labels.
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

    for i in tqdm(range(len(X)), desc=f"[{data_name}]"):
        print("y shape:", y.shape)

        # DO THE DRUG SELECTION y shape: (13011, 13)

        # Convert X[i] to bytes (immutable hashable form)
        x_bytes = X[i].tobytes()
        # y_val = int(y[i])
        y_val = int(np.ravel(y[i])[0])
 
        key = (x_bytes, y_val)
        if key not in seen:
            seen.add(key)
            uniq_indices.append(i)

    uniq_indices = np.array(uniq_indices)
    np.save(out_path, uniq_indices)

    # Logging
    reduction = len(X) - len(uniq_indices)
    frac = 100.0 * reduction / len(X)
    log_msg = (f"[{data_name}] data deduplicated {len(X)} → {len(uniq_indices)} "
               f"({reduction} removed, {frac:.1f}% reduction)")
    print(log_msg)

    with open(os.path.join(out_dir, "dedup_log.txt"), "a") as f:
        f.write(log_msg + "\n")

    return uniq_indices


def correct_shap_value_shape(shap_values, feature_names, shap_output_path=None):
    # Collapse over base channels (axis=0 → A, T, G, C, -)
    shap_abs = np.abs(shap_values)        # shape: (samples, bases, positions, loci, drugs)
    # shap_mean = shap_abs.mean(axis=0)     # mean over samples → shape: (bases, positions, loci, drugs)
    shap_collapsed = shap_abs.sum(axis=1)  # sum over bases → shape: (samples, positions, loci, drugs)
    # shape: (10, 10241, 12, 13)

    # Transpose to shape: (samples, loci, positions, drugs)
    shap_transposed = np.transpose(shap_collapsed, (0, 2, 1, 3))  # shape: (10, 2, 10241, 1)

    # Reshape to (samples, positions * loci, drugs)
    n_samples, n_loci, n_pos, n_drugs = shap_transposed.shape
    shap_all_drugs = shap_transposed.reshape(n_samples, n_loci * n_pos, n_drugs)  # shape: (10, 2 * 10241, 1)
    print("SHAP all drugs shape:", shap_all_drugs.shape)

    assert len(feature_names) == shap_all_drugs.shape[1]

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

def compute_shap_for_drug(model, X, feature_names, shap_output_path, is_deep_model=False, y=None, background_frac=0.1):
    # print(f"After deduplication, X shape: {X_dedup.shape}, y shape: {y_dedup.shape}")

    # if y_dedup is not None:
    #     valid_indices = np.where(y_dedup != -1)[0]
    #     X_valid = X_dedup[valid_indices]
    #     y_valid = y_dedup[valid_indices]
    #     print(f"After filtering for valid labels, X shape: {X_valid.shape}, y shape: {y_valid.shape}")

    # ---- new method of importance ----
    # select 10% of training data as background
    # and explain on full - background set 
    bg_size_all = min(160, int(len(X) * background_frac))
    ex_size_all = len(X) - bg_size_all

    # shap_values = compute_shap_values(
    #     model,
    #     X,
    #     feature_names,
    #     shap_output_path,
    #     background_size=bg_size_all,
    #     explainer_size=ex_size_all,
    #     is_deep_model=is_deep_model,
    #     y=y,
    # )

    shap_values = compute_shap_values_strat(
        model,
        X,
        feature_names,
        shap_output_path,
        is_deep_model=is_deep_model,
        y=y,
        seed=42,
    )

    return shap_values


def compute_shap_values(model, X, feature_names, shap_output_path, background_size, explainer_size, is_deep_model=False, y=None):
    """
    Compute SHAP values using the provided model and training data.
    Uses DeepExplainer/GradientExplainer for deep models, and Tree/Linear for others.
    """
    print("\nComputing SHAP values...")

    print("y shape:", y.shape)

    # ---------------------------------------------------
    # Select background from training data
    # ---------------------------------------------------
    B = min(background_size, len(X))
    print(f"Using background size {B} from {len(X)} training samples")

    # Randomly sample indices
    bg_indices = random.sample(range(len(X)), B)

    # Select background subset
    X_background = X[bg_indices]

    # Convert to TensorFlow tensor (and move to GPU if needed)
    # X_background_tf = tf.convert_to_tensor(X_background, dtype=tf.float32)
    # print(f"Background tensor shape: {X_background.shape}")

    # validation = full_ds minus background
    all_indices = set(range(len(X)))
    val_indices = list(all_indices - set(bg_indices))
    X_val = X[val_indices]
    y_val = y[val_indices]


    if len(X.shape) != 4:
        raise ValueError(f"X_train must be 4D (batch, height, width, channels), got {X.shape}")
    
    explainer = shap.DeepExplainer(model, X_background)
    # explainer = shap.GradientExplainer(model, X_background_tf)

    print(f"Explainer created")

    # explanation from validation set
    E = min(explainer_size, len(X_val))
    print(f"Explaining {E} samples from {len(X_val)} validation samples")
    explain_indices = random.sample(range(len(X_val)), E)

    X_explain = X_val[explain_indices]
    y_explain = y_val[explain_indices]
    
    shap_values_all_dims = explainer.shap_values(X_explain)
    print(f"SHAP values computed")

    shap_values = correct_shap_value_shape(shap_values_all_dims, feature_names, shap_output_path)      

    return shap_values


from sklearn.model_selection import train_test_split
import numpy as np
import random
import shap

def compute_shap_values_strat(
    model,
    X,
    feature_names,
    shap_output_path,
    is_deep_model=False,
    y=None,
    seed=42,
):
    """
    Compute SHAP values using stratified selection of background and explainer samples.
    """

    print("\nComputing SHAP values...")

    if y is None:
        raise ValueError("Stratification requires labels (y).")

    N = len(X)
    assert N == len(y), "X and y must have the same length"

    print(f"Total samples: {N}")
    print(f"y shape: {y.shape}")

    y = np.array(y).reshape(-1)
    print(f"y shape after flatten: {y.shape}")

    # =====================================================
    # Step 1 — Stratified background/explainer selection
    # =====================================================
    bg_frac = 0.2
    max_bg = 160
    y = np.array(y).astype(int)

    idx = np.arange(N)

    if np.unique(y).size > 1:
        bg_idx, ex_idx = train_test_split(
            idx, train_size=bg_frac, stratify=y, random_state=seed
        )
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        cut = max(1, int(round(bg_frac * N)))
        bg_idx, ex_idx = perm[:cut], perm[cut:]

    # cap background if needed
    if len(bg_idx) > max_bg:
        if np.unique(y[bg_idx]).size > 1:
            bg_idx, _ = train_test_split(
                bg_idx, train_size=max_bg, stratify=y[bg_idx], random_state=seed
            )
        else:
            rng = np.random.default_rng(seed)
            bg_idx = rng.choice(bg_idx, size=max_bg, replace=False)
        mask = np.ones(N, dtype=bool)
        mask[bg_idx] = False
        ex_idx = np.where(mask)[0]

    # ensure explainer subset not empty
    if len(ex_idx) == 0:
        ex_idx = np.array([bg_idx[-1]])
        bg_idx = bg_idx[:-1]

    print(f"Background samples: {len(bg_idx)}, Explainer samples: {len(ex_idx)}")
    print(f"BG label counts: {np.bincount(y[bg_idx])}")
    print(f"EX label counts: {np.bincount(y[ex_idx])}")

    X_background = X[bg_idx]
    X_explain = X[ex_idx]
    y_explain = y[ex_idx]

    explainer_size = len(X) - len(X_background)

    # =====================================================
    # Step 2 — Sanity check for deep models
    # =====================================================
    if len(X.shape) != 4:
        raise ValueError(f"X must be 4D (batch, height, width, channels); got {X.shape}")

    # =====================================================
    # Step 3 — Build SHAP Explainer
    # =====================================================
    if is_deep_model:
        explainer = shap.DeepExplainer(model, X_background)
        # explainer = shap.GradientExplainer(model, X_background)  # alternative
    else:
        raise ValueError("Non-deep model mode not yet implemented in this snippet.")

    print("Explainer created successfully.")

    # =====================================================
    # Step 4 — Compute SHAP values
    # =====================================================
    E = min(explainer_size, len(X_explain))
    print(f"Explaining {E} samples from {len(X_explain)} explainer pool")
    explain_indices = random.sample(range(len(X_explain)), E)

    X_explain_subset = X_explain[explain_indices]
    y_explain_subset = y_explain[explain_indices]

    shap_values_all_dims = explainer.shap_values(X_explain_subset)
    print("SHAP values computed.")

    shap_values = correct_shap_value_shape(
        shap_values_all_dims, feature_names, shap_output_path
    )

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
    plot_path = os.path.join(output_dir, f"shap_summary_{drug_name}.png")

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



# import numpy as np
# from sklearn.model_selection import train_test_split
# import numpy as np

# def stratified_bg_explain_indices_from_ds(ds, bg_frac=0.10, seed=42, max_bg=None):
#     """
#     Disjoint BG/EX split for a torch-style dataset ds (getitem -> (x, y)).

#     Guarantees:
#       • If BOTH classes (R=1, S=0) exist in the pool, BG is stratified and contains both.
#       • If only one class exists, BG is a random subset.
#       • BG and EX are disjoint; EX is non-empty.
#       • If max_bg is provided, |BG| ≤ max_bg (label balance preserved when possible).

#     Returns: (bg_idx, ex_idx, y_pool)
#       bg_idx, ex_idx: np.int64 arrays of indices into ds
#       y_pool:         np.int64 array of 0/1 labels for ds
#     """
#     N = len(ds)
#     assert N >= 2, "Need at least 2 samples to split."
#     y_pool = np.array([int(ds[i][1]) for i in range(N)], dtype=int)
#     idx = np.arange(N)

#     # 1) initial split (stratified if both labels present)
#     if np.unique(y_pool).size > 1: # both classes present
#         bg_idx, ex_idx = train_test_split(
#             idx, train_size=bg_frac, stratify=y_pool, random_state=seed
#         ) #stratified split
#     else:  # single-class pool → random split
#         rng = np.random.default_rng(seed)
#         cut = max(1, int(round(bg_frac * N)))
#         perm = rng.permutation(idx)
#         bg_idx, ex_idx = perm[:cut], perm[cut:] #random split

#      # 2) optional cap on BG size, preserving class presence if possible
#     if max_bg is not None and len(bg_idx) > max_bg: 
#         if np.unique(y_pool[bg_idx]).size > 1:
#             bg_idx, _ = train_test_split(
#                 bg_idx, train_size=max_bg, stratify=y_pool[bg_idx], random_state=seed
#             )
#         else:
#             rng = np.random.default_rng(seed)
#             bg_idx = rng.choice(bg_idx, size=max_bg, replace=False)

#         mask = np.ones(N, dtype=bool); mask[bg_idx] = False
#         ex_idx = np.where(mask)[0] # recompute ex_idx as complement

#     # ensure at least one isolate exists to explain
#     if len(ex_idx) == 0:
#         ex_idx = np.array([bg_idx[-1]]) #move one from bg to ex
#         bg_idx = bg_idx[:-1] #remove last from bg

#     return bg_idx.astype(np.int64), ex_idx.astype(np.int64), y_pool


# # NEW: stratified, disjoint BG/EX (on unique pool labels)
# bg_idx, explain_idx, y_pool = stratified_bg_explain_indices_from_ds(
#     pool_ds, bg_frac=bg_frac, seed=seed, max_bg=max_bg
# )

# meta = {
#     "run": run_name,
#     "N": int(N),
#     "bg_frac": float(bg_frac),
#     "max_bg": None if max_bg is None else int(max_bg),
#     "actual_bg": int(len(bg_idx)),
#     "actual_explain": int(len(explain_idx)),
#     "seed": int(seed),
#     # split sanity:
#     "bg_pos": int((y_pool[bg_idx] == 1).sum()),
#     "bg_neg": int((y_pool[bg_idx] == 0).sum()),
#     "ex_pos": int((y_pool[explain_idx] == 1).sum()),
#     "ex_neg": int((y_pool[explain_idx] == 0).sum()),
# }
# print(f"[SHAP] {run_name} | BG={meta['actual_bg']} (R={meta['bg_pos']}, S={meta['bg_neg']}) "
#         f"| EX={meta['actual_explain']} (R={meta['ex_pos']}, S={meta['ex_neg']})")