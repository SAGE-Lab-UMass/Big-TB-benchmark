import os
import pandas as pd
import ipdb
from collections import defaultdict
from parameters.locus_order import GENE_OPERONIC_PAIR_MAPS

def find_drug_column(df, drug):
    for col in df.columns:
        if col.lower() == drug.lower():
            return col
    raise ValueError(f"Drug column '{drug}' not found in DataFrame.")

def export_relevant_set(relevant_set, output_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pd.DataFrame(sorted(relevant_set), columns=["WHO_R_features"]).to_csv(output_path, index=False)
    print(f"Exported relevant WHO-R features to {output_path}")

def create_mutations_column(mapped_df, has_neg_strand):
    """
    Efficiently create the 'WHO_mutation_feature' column:
    - Use neg strand if available, else pos strand
    - Subtract 1 from non-null, non-negative values
    - Keep negative values as-is
    - Skip (set NaN) if position is null
    """
    # Copy to avoid SettingWithCopyWarning
    mapped_df = mapped_df.copy()

    if has_neg_strand:
        position = mapped_df["rel_gapped_mutation_position_neg_strand"].where(
            mapped_df["rel_gapped_mutation_position_neg_strand"].notna(),
            mapped_df["rel_gapped_mutation_position_pos_strand"]
        )
    else:
        position = mapped_df["rel_gapped_mutation_position_pos_strand"]

    # Subtract 1 only for non-null and non-negative positions, as -ve positions are upstream positions already 0-indexed
    adjusted_position = position.where(position < 0, position - 1)

    # Create a temporary mapped gene name (don't overwrite original)
    gene_operon = mapped_df["gene"].map(GENE_OPERONIC_PAIR_MAPS).fillna(mapped_df["gene"])

    # Generate WHO_mutation_feature only for non-null positions
    mask = position.notna()
    mapped_df.loc[mask, "WHO_mutation_feature"] = (
        gene_operon[mask] + "_" + adjusted_position.loc[mask].astype(int).astype(str)
    )

    return mapped_df


def read_csv_file(file_path, has_neg_strand=False):
    """
    Reads CSV and attaches WHO_mutation_feature column.
    """
    try:
        df = pd.read_csv(file_path)
        return create_mutations_column(df, has_neg_strand)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

def find_hits_at_k(predicted, relevant, k):
    hits = sum(1 for feature in predicted[:k] if feature in relevant)
    return hits

def compute_ap_at_k(predicted, relevant, k):
    hits = 0
    sum_precisions = 0
    print("Relevant features:", relevant)
    print("Predicted features:", predicted[:k])
    for i, feature in enumerate(predicted[:k], start=1):
        print("Feature:", feature)
        
        if feature in relevant:
            hits += 1
            sum_precisions += hits / i
            print("Hit:", hits, "Precision sum:", sum_precisions)
    # return sum_precisions / min(k, len(relevant)) if relevant else 0.0
    return sum_precisions / len(relevant) if relevant else 0.0


def compute_ar_at_k(predicted, relevant, k):
    hits = find_hits_at_k(predicted, relevant, k)
    # return hits / len(relevant) if relevant else 0.0
    return hits / min(k, len(relevant)) if relevant else 0.0

def compute_p_at_k(predicted, relevant, k):
    hits = find_hits_at_k(predicted, relevant, k)
    # return hits / len(relevant) if relevant else 0.0
    return hits / k if relevant else 0.0

def compute_r_at_k(predicted, relevant, k):
    hits = find_hits_at_k(predicted, relevant, k)
    # return hits / len(relevant) if relevant else 0.0
    return hits / len(relevant) if relevant else 0.0

def compute_map_mar_for_directory(directory, important_features_ranked, drug, has_neg_strand=False):
    # Step 1: Build relevant set (all features marked "R" across all files)
    relevant_set = set()

    for file in os.listdir(directory):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(directory, file)
        df = read_csv_file(file_path, has_neg_strand)
        print(f"Processing file: {file}")

        if df is None or "WHO_mutation_feature" not in df.columns:
            continue

        try:
            drug_col = find_drug_column(df, drug)
            relevant_set.update(
                df.loc[df[drug_col] == "R", "WHO_mutation_feature"].dropna()
            )
        except Exception as e:
            print(f"Error in file {file}: {e}")
            continue

    export_relevant_set(relevant_set, f"/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/SD-CNN/interpretability/output/relevant_features_{drug}_new.csv")

    # Step 2: Compute MAP@k and MAR@k using the final relevant_set
    p_k_scores = {}
    r_k_scores = {}
    map_k_scores = {}
    mar_k_scores = {}
    hits_k_num = {}

    for k in [1, 5, 10]:
        p_k_scores[k] = compute_p_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k)
        r_k_scores[k] = compute_r_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k)
        map_k_scores[k] = compute_ap_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k)
        # mar_k_scores[k] = compute_ar_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k)
        hits_k_num[k] = find_hits_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k)

    return p_k_scores, r_k_scores, map_k_scores, hits_k_num


def save_map_mar_results(p, r, map, hits, output_csv):
    rows = [{"k": k, "P@k": p[k], "R@k": r[k], "MAP@k": map[k], "Hits@k": hits[k]} for k in sorted(map)]
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved MAP/MAR results to {output_csv}")


def get_confident_mutation_hits(
    vcf_who_map_directory, 
    important_features, 
    two_most_important_below_threshold, 
    drug,
    model_type="sd-cnn",
    has_neg_strand=False,
    output_csv="confident_mutation_hits.csv"
    ):
    # important_model_features = list(set(important_features).union(two_most_important_below_threshold))
    important_model_features = important_features + two_most_important_below_threshold

    output_csv = f"/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/SD-CNN/interpretability/output/map_mar_{drug}_new.csv"

    p, r, map, hits = compute_map_mar_for_directory(
        vcf_who_map_directory,
        important_model_features,
        drug,
        has_neg_strand
    )
    
    save_map_mar_results(p, r, map, hits, output_csv)