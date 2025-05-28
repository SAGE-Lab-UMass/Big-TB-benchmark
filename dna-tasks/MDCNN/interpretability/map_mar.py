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
        #mapped_df.loc[mask, "gene"] + "_" + adjusted_position.loc[mask].astype(int).astype(str)
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
    return hits / len(relevant) if relevant else 0.0

def compute_map_mar(true_causal_variants_df, directory, important_features_ranked, drug, has_neg_strand=False, embed_type='one_hot'):
    # Build relevant set (all features marked "R" across all files)
    relevant_set = set(true_causal_variants_df["gapped_mutation_feature"].dropna())
    print(f"Relevant set size for drug {drug}: {len(relevant_set)}")

    # Compute MAP@k and MAR@k using the final relevant_set
    map_k_scores = {}
    mar_k_scores = {}
    hits_k_num = {}

    for k in [1, 5, 10]:
        map_k_scores[k] = compute_ap_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k)
        mar_k_scores[k] = compute_ar_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k)
        hits_k_num[k] = find_hits_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k)

    return map_k_scores, mar_k_scores, hits_k_num


def create_combined_variants_csv(vcf_who_map_directory, combined_variants_csv="combined_variants.csv"):
    # Initialize an empty list to store DataFrames
    all_dataframes = []

    # Loop through all CSV files in the directory
    for filename in os.listdir(vcf_who_map_directory):
        if filename.endswith(".csv"):
            print(f"Processing file: {filename}")
            file_path = os.path.join(combined_variants_csv, filename)
            df = pd.read_csv(file_path, na_values=[''], keep_default_na=False)
            all_dataframes.append(df)

    # Concatenate all DataFrames into one big DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(combined_variants_csv, index=False)
    print(f"All genotype variants combined into: {combined_variants_csv}")

    return combined_df


def compute_true_causal_variant_set(vcf_who_map_directory, drug, output_path, has_neg_strand=False, combined_variants_csv="combined_variants.csv"):
    if os.path.exists(combined_variants_csv):
        print(f"Combined variants CSV already exists: {combined_variants_csv}, loading it.")
        combined_variants_df = pd.read_csv(combined_variants_csv)
    else:
        # Create the combined variants CSV
        print(f"Creating combined variants CSV: {combined_variants_csv}")
        combined_variants_df = create_combined_variants_csv(vcf_who_map_directory, combined_variants_csv)

    # Create the relevant set CSV
    updated_combined_variants_df = create_mutations_column(combined_variants_df, has_neg_strand)
    drug_col = find_drug_column(updated_combined_variants_df, drug)
    confidence_column = f"{drug_col}_confidence"

    # Filtering the dataframe to include only variants with confidence values indicating resistance (R)
    causal_variants_df = updated_combined_variants_df[updated_combined_variants_df[confidence_column].str.startswith(("1)", "2)"))][['variant', 'WHO_mutation_feature']].drop_duplicates()

    causal_variants_df.columns = ["variants", "gapped_mutation_feature"]

    # Save the relevant set to a CSV file
    causal_variants_path = os.path.join(output_path, f"true_causal_variants_{drug}.csv")

    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(causal_variants_path), exist_ok=True)
    causal_variants_df.to_csv(causal_variants_path, index=False)
    print(f"Resistant variants saved to: {causal_variants_path}")

    return causal_variants_df


def save_map_mar_results(map, mar, hits, output_csv):
    rows = [{"k": k, "MAP@k": map[k], "MAR@k": mar[k], "Hits@k": hits[k]} for k in sorted(map)]
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved MAP/MAR results to {output_csv}")


def get_confident_mutation_hits(
    vcf_who_map_directory, 
    important_model_features, 
    drug,
    output_path,
    output_file_name,
    embed_type='one_hot',
    has_neg_strand=False,
    ):

    causal_variants_path = os.path.join(output_path, f"true_causal_variants_{drug}.csv")
    if os.path.exists(causal_variants_path):
        print(f"True causal variants CSV already exists: {causal_variants_path}, loading it.")
        true_causal_variants_df = pd.read_csv(causal_variants_path)
    else:
        # Create the true causal variants CSV
        print(f"Creating true causal variants CSV: {causal_variants_path}")
        true_causal_variants_df = compute_true_causal_variant_set(
            vcf_who_map_directory,
            drug,
            output_path,
            has_neg_strand
        )

    map, mar, hits = compute_map_mar(
        true_causal_variants_df,
        vcf_who_map_directory,
        important_model_features,
        drug,
        has_neg_strand, 
        embed_type
    )
    
    save_map_mar_results(map, mar, hits, output_file_name)