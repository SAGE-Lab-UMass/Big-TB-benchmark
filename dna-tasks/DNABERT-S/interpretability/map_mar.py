import os
import pandas as pd
import ipdb
from collections import defaultdict
from dataloader.locus_order import GENE_OPERONIC_PAIR_MAPS

k_list = [1, 5, 10]

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

    # export mapped df to csv
    # mapped_df.to_csv("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/Regression_l2/interpretability/interpretability_output/debug.csv", index=False)

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

def find_hits_at_k(predicted, relevant, k, model_embed_type):
    """
    Computes the number of hits among the top-k predicted features.

    Args:
        predicted: list of predicted features
            - If model_embed_type == "tokens": [(gene, start_pos, end_pos), ...]
            - Else: [f"{gene}_{pos}", ...]
        relevant: list of ground truth positions
            - Always: [f"{gene}_{pos}", ...]
        k: number of top features to evaluate
        model_embed_type: "tokens", "one-hot", "maj-min"

    Returns:
        hits (int): number of predicted features that match ground truth
    """
    hits = 0
    if model_embed_type == "tokens":
        relevant_dict = {}
        for rel in relevant:
            gene_k, pos_k = rel.split("_")
            pos_k = int(pos_k)
            relevant_dict.setdefault(gene_k, set()).add(pos_k)

        for gene_i, start_i, end_i in predicted[:k]:
            start_i, end_i = int(start_i), int(end_i)

            # Fast lookup only in relevant positions for this gene
            for pos_k in relevant_dict.get(gene_i, []):
                if start_i <= pos_k <= end_i:
                    hits += 1
                    break  # stop after the first match for this prediction
    else:
        hits = sum(1 for feature in predicted[:k] if feature in relevant)
    return hits

def compute_ap_at_k(predicted, relevant, k, model_embed_type):
    hits = 0
    sum_precisions = 0
    matched = set()

    print("Relevant features:", relevant)
    print("Predicted features:", predicted[:k])
    if model_embed_type == "tokens":
        relevant_dict = defaultdict(set)
        for rel in relevant:
            gene_k, pos_k = rel.split("_")
            relevant_dict[gene_k].add(int(pos_k))

        for i, (gene_i, start_i, end_i) in enumerate(predicted[:k], start=1):
            start_i, end_i = int(start_i), int(end_i)

            for pos_k in relevant_dict.get(gene_i, []):
                rel_str = f"{gene_i}_{pos_k}"
                if start_i <= pos_k <= end_i and rel_str not in matched:
                    hits += 1
                    matched.add(rel_str)
                    sum_precisions += hits / i
                    print(f"Hit at rank {i}: {rel_str}, Precision: {hits / i}")
                    break  # avoid double-counting this prediction
    else:
        for i, feature in enumerate(predicted[:k], start=1):
            if feature in relevant and feature not in matched:
                hits += 1
                matched.add(feature)
                sum_precisions += hits / i
                print(f"Hit at rank {i}: {feature}, Precision: {hits / i}")

        # for i, feature in enumerate(predicted[:k], start=1):
        #     print("Feature:", feature)
            
            # if feature in relevant:
            #     hits += 1
            #     sum_precisions += hits / i
            #     print(f"Hit at rank {i}: {rel}, Precision: {hits / i}")
    # return sum_precisions / min(k, len(relevant)) if relevant else 0.0
    return sum_precisions / len(relevant) if relevant else 0.0


def compute_ar_at_k(predicted, relevant, k, model_embed_type):
    hits = find_hits_at_k(predicted, relevant, k, model_embed_type)
    # return hits / len(relevant) if relevant else 0.0
    return hits / min(k, len(relevant)) if relevant else 0.0

def compute_p_at_k(predicted, relevant, k, model_embed_type):
    hits = find_hits_at_k(predicted, relevant, k, model_embed_type)
    # return hits / len(relevant) if relevant else 0.0
    return hits / k if relevant else 0.0

def compute_r_at_k(predicted, relevant, k, model_embed_type):
    hits = find_hits_at_k(predicted, relevant, k, model_embed_type)
    # return hits / len(relevant) if relevant else 0.0
    return hits / len(relevant) if relevant else 0.0

def compute_relevant_set(directory, drug, has_neg_strand=False):
    output_path = f"relevant_features_{drug}.csv"
    if os.path.exists(output_path):
        print(f"Relevant set already exists at {output_path}. Loading...")
        relevant_set = pd.read_csv(output_path)["WHO_R_features"].tolist()
        return set(relevant_set)
    else:
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

        export_relevant_set(relevant_set, f"relevant_features_{drug}.csv")
    return relevant_set

def compute_map_mar_for_directory(directory, important_features_ranked, drug, model_embed_type, has_neg_strand=False):
    # Step 1: Build relevant set (all features marked "R" across all files)
    relevant_set = compute_relevant_set(directory, drug, has_neg_strand)

    # Step 2: Compute MAP@k and MAR@k using the final relevant_set
    map_k_scores = {}
    mar_k_scores = {}
    hits_k_num = {}

    for k in k_list:
        map_k_scores[k] = compute_ap_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k, model_embed_type=model_embed_type)
        mar_k_scores[k] = compute_ar_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k, model_embed_type=model_embed_type)
        hits_k_num[k] = find_hits_at_k(predicted=important_features_ranked, relevant=relevant_set, k=k, model_embed_type=model_embed_type)

    return map_k_scores, mar_k_scores, hits_k_num


def save_map_mar_results(map, mar, hits, output_csv):
    rows = [{"k": k, "MAP@k": map[k], "MAR@k": mar[k], "Hits@k": hits[k]} for k in sorted(map)]
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved MAP/MAR results to {output_csv}")


def get_confident_mutation_hits(
    vcf_who_map_directory, 
    important_model_features, 
    drug,
    model_embed_type='one-hot',
    has_neg_strand=False,
    output_csv="confident_mutation_hits.csv"
    ):

    output_csv = f"map_mar_{drug}.csv"

    map, mar, hits = compute_map_mar_for_directory(
        vcf_who_map_directory,
        important_model_features,
        drug,
        model_embed_type,
        has_neg_strand
    )
    
    save_map_mar_results(map, mar, hits, output_csv)