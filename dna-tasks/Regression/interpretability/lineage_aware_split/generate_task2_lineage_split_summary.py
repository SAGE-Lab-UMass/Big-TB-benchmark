"""
Generate Task 2 Lineage Split Summary Statistics for DNA Regression

Produces a summary table analogous to Table B in task2_random_vs_lineage_split.xlsx.
For each drug × heldout-lineage combination, only the test-set isolates (the held-out
lineage) are used to count WHO-catalogue carriers, mirroring the protein-level Table B.

Dataset phenotype convention: 0 = resistant (R), 1 = susceptible (S).

Output columns (per drug × lineage row):
- Drug, Held-out lineage
- Residues discovered  : cohort-wide (same as Table A; loaded from random-split summary)
- Test carriers        : isolate-position pairs among heldout-lineage test isolates
- Test R / Test S      : split by phenotype (R = 0.0, S = 1.0)
- Positions w/ 0 test carriers : WHO positions with no carriers in the test set
- Min / Median / Max test carriers per position
"""

import os
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WHO_CATALOGUE_PATH = (
    "/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB"
    "/dna-tasks/data/WHO_catalogue_2023/WHO_resistance_variants_all_2023.csv"
)
GENOTYPE_PHENOTYPE_PATH = (
    "/project/pi_annagreen_umass_edu/saishradha/project_data_curation"
    "/benchmarking/Regression_l2/input_data/reduced_prepared_data"
    "/combined_geno_pheno_df.csv"
)
VCF_WHO_MAPPED_DIR = (
    "/project/pi_annagreen_umass_edu/saishradha/project_data_curation"
    "/vcf_who_mapped_data"
)
TRAINING_OUTPUT_BASE = (
    "/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB"
    "/dna-tasks/Regression/training_output/lineage_aware_results_logreg_1000"
)
RANDOM_SPLIT_SUMMARY_PATH = (
    "/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB"
    "/dna-tasks/Regression/interpretability/data/task2_random_split_summary.csv"
)
OUTPUT_PATH = (
    "/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB"
    "/dna-tasks/Regression/interpretability/lineage_aware_data_path"
    "/task2_lineage_split_summary.csv"
)

# Drug names → uppercase column names in geno-pheno CSV and VCF mapping files
DRUGS = [
    "Rifampicin",
    "Isoniazid",
    "Ethambutol",
    "Pyrazinamide",
    "Streptomycin",
    "Moxifloxacin",
    "Levofloxacin",
    "Ethionamide",
    "Capreomycin",
    "Kanamycin",
]

DRUG_NAME_MAP = {
    "Rifampicin":   "RIFAMPICIN",
    "Isoniazid":    "ISONIAZID",
    "Ethambutol":   "ETHAMBUTOL",
    "Pyrazinamide": "PYRAZINAMIDE",
    "Streptomycin": "STREPTOMYCIN",
    "Moxifloxacin": "MOXIFLOXACIN",
    "Levofloxacin": "LEVOFLOXACIN",
    "Ethionamide":  "ETHIONAMIDE",
    "Capreomycin":  "CAPREOMYCIN",
    "Kanamycin":    "KANAMYCIN",
}

LINEAGES = [1, 2, 3, 4]
RESISTANCE_CONF = ["1) Assoc w R", "2) Assoc w R - Interim"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_who_catalogue():
    print("Loading WHO catalogue...")
    df = pd.read_csv(WHO_CATALOGUE_PATH)
    resistance = df[df["confidence"].isin(RESISTANCE_CONF)].copy()
    print(f"  {len(df)} total variants → {len(resistance)} high-confidence resistance")
    return resistance


def load_genotype_phenotype_data():
    """Returns dataframe indexed by integer row position (matching row_id in split files)."""
    print("Loading genotype-phenotype data...")
    df = pd.read_csv(GENOTYPE_PHENOTYPE_PATH, low_memory=False)
    df = df.reset_index(drop=True)   # row position == row_id in split manifests
    print(f"  {len(df)} isolates")
    return df


def load_random_split_summary():
    df = pd.read_csv(RANDOM_SPLIT_SUMMARY_PATH)
    return df.set_index("Drug")["Residues discovered"].to_dict()


def load_isolate_variant_file(isolate_id):
    path = os.path.join(VCF_WHO_MAPPED_DIR, f"{isolate_id}_variants.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Position extraction  (identical logic to random-split script)
# ---------------------------------------------------------------------------

def create_gene_position_key(gene, position):
    return f"{gene.strip('/').strip()}_{int(position)}"


def extract_who_position_from_variant(variant_row):
    gene = variant_row["gene"]
    pos = None
    if pd.notna(variant_row.get("rel_gapped_mutation_position_neg_strand")):
        pos = variant_row["rel_gapped_mutation_position_neg_strand"]
    elif pd.notna(variant_row.get("rel_gapped_mutation_position_pos_strand")):
        pos = variant_row["rel_gapped_mutation_position_pos_strand"]
    if pos is None:
        return None
    pos = int(pos) - 1 if pos >= 0 else int(pos)
    return create_gene_position_key(gene, pos)


def build_who_position_set_for_drug(drug, who_catalogue, sample_vcf_files=100):
    """Sample VCF files to build the full set of WHO positions for a drug."""
    vcf_files = [f for f in os.listdir(VCF_WHO_MAPPED_DIR) if f.endswith("_variants.csv")]
    random.seed(42)
    sample = random.sample(vcf_files, min(sample_vcf_files, len(vcf_files)))

    drug_conf_col = f"{drug}_confidence"
    positions = set()
    for fname in sample:
        try:
            df = pd.read_csv(os.path.join(VCF_WHO_MAPPED_DIR, fname), on_bad_lines="skip")
            if drug not in df.columns:
                continue
            for _, row in df[(df[drug] == "R") & (df[drug_conf_col].isin(RESISTANCE_CONF))].iterrows():
                key = extract_who_position_from_variant(row)
                if key:
                    positions.add(key)
        except Exception:
            continue
    return positions


# ---------------------------------------------------------------------------
# Per drug-lineage statistics
# ---------------------------------------------------------------------------

def compute_lineage_statistics(drug, lineage, geno_pheno_df, who_catalogue, all_who_positions):
    """
    Compute Table B statistics for one drug × heldout-lineage combination.
    Returns None if the training run was not feasible (no saved test_predictions.csv).
    """
    drug_upper = DRUG_NAME_MAP[drug]
    test_pred_path = os.path.join(
        TRAINING_OUTPUT_BASE, drug_upper, f"heldout_lineage_{lineage}", "test_predictions.csv"
    )

    if not os.path.exists(test_pred_path):
        print(f"  [{drug} L{lineage}] No test_predictions.csv – skipping (infeasible run)")
        return None

    test_preds = pd.read_csv(test_pred_path)  # columns: row_id, Lineage, y_true, y_pred

    # Map row_ids to isolate IDs
    test_row_ids = test_preds["row_id"].values
    test_isolates = geno_pheno_df.loc[test_row_ids, "isolate"].values
    test_labels   = test_preds["y_true"].values   # 0 = R, 1 = S

    print(f"\n  [{drug} L{lineage}] {len(test_isolates)} test isolates "
          f"(R={int((test_labels==0).sum())}, S={int((test_labels==1).sum())})")

    # Build position carriers only from test isolates
    drug_col      = drug
    drug_conf_col = f"{drug}_confidence"
    position_carriers = defaultdict(list)
    missing_files = 0

    for isolate_id, phenotype in tqdm(
        zip(test_isolates, test_labels),
        total=len(test_isolates),
        desc=f"    {drug} L{lineage}",
        leave=False,
    ):
        variant_df = load_isolate_variant_file(isolate_id)
        if variant_df is None:
            missing_files += 1
            continue
        if drug_col not in variant_df.columns:
            continue

        resistance_vars = variant_df[
            (variant_df[drug_col] == "R") &
            (variant_df[drug_conf_col].isin(RESISTANCE_CONF))
        ]
        for _, var_row in resistance_vars.iterrows():
            key = extract_who_position_from_variant(var_row)
            if key:
                position_carriers[key].append((isolate_id, float(phenotype)))

    if missing_files:
        print(f"    Missing VCF files: {missing_files}")

    # Aggregate
    test_carriers = sum(len(c) for c in position_carriers.values())
    test_r = sum(1 for carriers in position_carriers.values() for _, ph in carriers if ph == 0.0)
    test_s = sum(1 for carriers in position_carriers.values() for _, ph in carriers if ph == 1.0)

    counts = [len(c) for c in position_carriers.values()]
    if counts:
        min_c    = int(np.min(counts))
        median_c = float(np.median(counts))
        max_c    = int(np.max(counts))
    else:
        min_c = median_c = max_c = 0

    positions_zero = len(all_who_positions - set(position_carriers.keys()))

    return {
        "Drug":                         drug,
        "Held-out lineage":             lineage,
        "Residues discovered":          None,   # filled from random-split summary after loop
        "Test carriers":                test_carriers,
        "Test R":                       test_r,
        "Test S":                       test_s,
        "Positions w/ 0 test carriers": positions_zero,
        "Min test carriers/position":   min_c,
        "Median test carriers/position": median_c,
        "Max test carriers/position":   max_c,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Generating Task 2 Lineage Split Summary for DNA Regression (Table B)")
    print("=" * 80)

    who_catalogue    = load_who_catalogue()
    geno_pheno_df    = load_genotype_phenotype_data()
    residues_by_drug = load_random_split_summary()

    rows = []

    for drug in DRUGS:
        drug_upper = DRUG_NAME_MAP[drug]
        print(f"\n{'='*60}\nProcessing {drug}\n{'='*60}")

        # Build WHO position set once per drug (shared across lineages)
        all_who_positions = build_who_position_set_for_drug(drug, who_catalogue)
        print(f"  WHO positions for {drug}: {len(all_who_positions)}")

        for lineage in LINEAGES:
            stats = compute_lineage_statistics(
                drug, lineage, geno_pheno_df, who_catalogue, all_who_positions
            )
            if stats is None:
                continue
            stats["Residues discovered"] = residues_by_drug.get(drug, 0)
            rows.append(stats)

    if not rows:
        print("No results generated.")
        return

    cols = [
        "Drug",
        "Held-out lineage",
        "Residues discovered",
        "Test carriers",
        "Test R",
        "Test S",
        "Positions w/ 0 test carriers",
        "Min test carriers/position",
        "Median test carriers/position",
        "Max test carriers/position",
    ]
    result_df = pd.DataFrame(rows, columns=cols)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*80}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 80)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
