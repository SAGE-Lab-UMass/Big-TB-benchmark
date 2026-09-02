"""
Generate Task 2 Random Split Summary Statistics for DNA Regression

This script creates a summary table showing statistics about WHO-catalogued
resistance positions that appear in our genomic training data, similar to
Table A in the protein-level task2_random_vs_lineage_split.xlsx file.

Output columns:
- Drug: Antibiotic name
- Residues discovered: Number of unique nucleotide positions in WHO 2023 catalogue 
  (confidence 1 or 2) that appear in our dataset
- Total carriers: Cohort-wide count of isolate-position pairs where the isolate 
  carries a catalogued resistance variant
- Total R / Total S: Split by phenotype (resistant vs susceptible)
- Positions w/ 0 carriers: WHO positions with no carriers in the cohort
- Min/Median/Max carriers per position: Distribution statistics
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Configuration
WHO_CATALOGUE_PATH = "/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/data/WHO_catalogue_2023/WHO_resistance_variants_all_2023.csv"
GENOTYPE_PHENOTYPE_PATH = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/Regression_l2/input_data/reduced_prepared_data/combined_geno_pheno_df.csv"
SITE_INDICES_PATH = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/Regression_l2/input_data/reduced_prepared_data/site_indices.csv"
VCF_WHO_MAPPED_DIR = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/vcf_who_mapped_data"
OUTPUT_PATH = "/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Regression/interpretability/data/task2_random_split_summary.csv"

# Drug names (matching Table A)
DRUGS = [
    'Rifampicin',
    'Isoniazid', 
    'Ethambutol',
    'Pyrazinamide',
    'Streptomycin',
    'Moxifloxacin',
    'Levofloxacin',
    'Ethionamide',
    'Capreomycin',
    'Kanamycin'
]

# Drug name mappings (WHO catalogue uses different capitalization)
DRUG_NAME_MAP = {
    'Rifampicin': 'RIFAMPICIN',
    'Isoniazid': 'ISONIAZID',
    'Ethambutol': 'ETHAMBUTOL', 
    'Pyrazinamide': 'PYRAZINAMIDE',
    'Streptomycin': 'STREPTOMYCIN',
    'Moxifloxacin': 'MOXIFLOXACIN',
    'Levofloxacin': 'LEVOFLOXACIN',
    'Ethionamide': 'ETHIONAMIDE',
    'Capreomycin': 'CAPREOMYCIN',
    'Kanamycin': 'KANAMYCIN'
}


def load_who_catalogue():
    """Load WHO resistance catalogue and filter for high-confidence resistance variants."""
    print("Loading WHO catalogue...")
    who_df = pd.read_csv(WHO_CATALOGUE_PATH)
    
    # Filter for resistance associations (confidence 1 and 2)
    resistance_conf = ['1) Assoc w R', '2) Assoc w R - Interim']
    who_resistance = who_df[who_df['confidence'].isin(resistance_conf)].copy()
    
    print(f"Total WHO variants: {len(who_df)}")
    print(f"High-confidence resistance variants: {len(who_resistance)}")
    
    return who_resistance


def load_genotype_phenotype_data():
    """Load genotype-phenotype data."""
    print("\nLoading genotype-phenotype data...")
    geno_pheno_df = pd.read_csv(GENOTYPE_PHENOTYPE_PATH, low_memory=False)
    print(f"Total isolates: {len(geno_pheno_df)}")
    return geno_pheno_df


def load_site_indices():
    """Load site indices mapping genotype columns to gene positions."""
    print("\nLoading site indices...")
    site_indices = pd.read_csv(SITE_INDICES_PATH)
    print(f"Total genotype positions: {len(site_indices)}")
    return site_indices


def create_gene_position_key(gene, position):
    """Create standardized gene_position key format."""
    # Remove leading slash from gene if present
    gene_clean = gene.strip('/').strip()
    return f"{gene_clean}_{int(position)}"


def load_isolate_variant_file(isolate_id):
    """Load WHO variant mapping for a single isolate."""
    vcf_file = os.path.join(VCF_WHO_MAPPED_DIR, f"{isolate_id}_variants.csv")
    
    if os.path.exists(vcf_file):
        try:
            df = pd.read_csv(vcf_file, on_bad_lines='skip')
            return df
        except Exception as e:
            print(f"  Error reading {vcf_file}: {e}")
            return None
    return None


def extract_who_position_from_variant(variant_row, has_neg_strand=True):
    """
    Extract gene-relative position from VCF-WHO mapping row.
    
    For DNA-level analysis, we use rel_gapped_mutation_position which gives
    the gene-relative nucleotide position.
    """
    gene = variant_row['gene']
    
    # Use negative strand position if available, else positive strand
    if has_neg_strand and pd.notna(variant_row.get('rel_gapped_mutation_position_neg_strand')):
        position = variant_row['rel_gapped_mutation_position_neg_strand']
    elif pd.notna(variant_row.get('rel_gapped_mutation_position_pos_strand')):
        position = variant_row['rel_gapped_mutation_position_pos_strand']
    else:
        return None
    
    # Adjust position: subtract 1 for non-negative positions (0-indexing)
    if position >= 0:
        position = int(position) - 1
    else:
        position = int(position)
    
    return create_gene_position_key(gene, position)


def build_who_position_set_for_drug(drug, who_catalogue, sample_vcf_files=100):
    """
    Build a set of all WHO catalogue positions for a drug by examining VCF mapping files.
    
    This maps WHO catalogue genome coordinates to gene-relative positions by looking at
    a sample of VCF files that contain these variants.
    """
    print(f"  Building complete WHO position set for {drug}...")
    
    # Get WHO catalogue entries for this drug (resistance only)
    resistance_conf = ['1) Assoc w R', '2) Assoc w R - Interim']
    who_drug = who_catalogue[
        (who_catalogue['drug'].str.lower() == drug.lower())
    ].copy()
    
    # Track: genome_index -> gene_position mapping
    genome_to_gene_pos = {}
    who_positions = set()
    
    # Get list of VCF files to sample
    vcf_files = [f for f in os.listdir(VCF_WHO_MAPPED_DIR) if f.endswith('_variants.csv')]
    
    # Sample files to build the mapping (examining all 17k files would be slow)
    import random
    random.seed(42)
    sample_files = random.sample(vcf_files, min(sample_vcf_files, len(vcf_files)))
    
    drug_col_name = drug
    drug_conf_col = f"{drug}_confidence"
    
    for vcf_file in sample_files:
        vcf_path = os.path.join(VCF_WHO_MAPPED_DIR, vcf_file)
        try:
            df = pd.read_csv(vcf_path, on_bad_lines='skip')
            
            if drug_col_name not in df.columns:
                continue
            
            # Get resistance variants for this drug
            drug_variants = df[
                (df[drug_col_name] == 'R') & 
                (df[drug_conf_col].isin(resistance_conf))
            ].copy()
            
            for _, row in drug_variants.iterrows():
                position_key = extract_who_position_from_variant(row)
                if position_key:
                    who_positions.add(position_key)
        except:
            continue
    
    print(f"  Found {len(who_positions)} unique WHO positions from {len(sample_files)} sampled files")
    return who_positions


def compute_drug_statistics(drug, geno_pheno_df, who_catalogue):
    """
    Compute statistics for a single drug.
    
    Returns dict with:
    - residues_discovered: unique WHO positions in the cohort
    - total_carriers: total isolate-position pairs with resistance variants
    - total_r: carriers with resistant phenotype
    - total_s: carriers with susceptible phenotype
    - positions_zero_carriers: WHO positions with no carriers
    - min/median/max carriers per position
    """
    print(f"\n{'='*60}")
    print(f"Processing {drug}")
    print(f"{'='*60}")
    
    pheno_col = DRUG_NAME_MAP[drug]
    
    # Filter genotype-phenotype data for isolates with non-null phenotype
    drug_data = geno_pheno_df[geno_pheno_df[pheno_col].notna()].copy()
    print(f"Isolates with {drug} phenotype: {len(drug_data)}")
    
    # Get WHO catalogue entries for this drug
    who_drug = who_catalogue[who_catalogue['drug'].str.lower() == drug.lower()].copy()
    print(f"WHO resistance variants for {drug}: {len(who_drug)}")
    
    # Build complete set of WHO positions for this drug
    all_who_positions = build_who_position_set_for_drug(drug, who_catalogue)
    
    # Track carriers: {position: [(isolate_id, phenotype), ...]}
    position_carriers = defaultdict(list)
    
    # For each isolate with phenotype data
    missing_files = 0
    error_files = 0
    
    for _, isolate_row in tqdm(drug_data.iterrows(), total=len(drug_data), 
                               desc=f"Processing {drug} isolates"):
        isolate_id = isolate_row['isolate']
        phenotype = isolate_row[pheno_col]
        
        # Load variant file for this isolate
        variant_df = load_isolate_variant_file(isolate_id)
        
        if variant_df is None:
            missing_files += 1
            continue
        
        # Filter for resistance variants in this drug's WHO catalogue
        drug_col_name = drug  # Column name in VCF mapping file
        drug_conf_col = f"{drug}_confidence"
        
        if drug_col_name not in variant_df.columns:
            error_files += 1
            continue
        
        # Get resistance variants (R with confidence 1 or 2)
        resistance_conf = ['1) Assoc w R', '2) Assoc w R - Interim']
        resistance_variants = variant_df[
            (variant_df[drug_col_name] == 'R') & 
            (variant_df[drug_conf_col].isin(resistance_conf))
        ]
        
        # Extract positions and record carriers
        for _, var_row in resistance_variants.iterrows():
            position_key = extract_who_position_from_variant(var_row)
            if position_key:
                position_carriers[position_key].append((isolate_id, phenotype))
    
    print(f"  Missing mapping files: {missing_files}")
    print(f"  Files with errors: {error_files}")
    
    # Compute statistics
    residues_discovered = len(position_carriers)
    total_carriers = sum(len(carriers) for carriers in position_carriers.values())
    
    # Count R vs S carriers (dataset convention: 0 = resistant, 1 = susceptible)
    total_r = 0
    total_s = 0
    for carriers in position_carriers.values():
        for _, phenotype in carriers:
            if phenotype == 0.0:
                total_r += 1
            elif phenotype == 1.0:
                total_s += 1
    
    # Carrier distribution per position
    carrier_counts = [len(carriers) for carriers in position_carriers.values()]
    
    if carrier_counts:
        min_carriers = np.min(carrier_counts)
        median_carriers = np.median(carrier_counts)
        max_carriers = np.max(carrier_counts)
    else:
        min_carriers = median_carriers = max_carriers = 0
    
    # Compute positions with 0 carriers
    # These are WHO positions that exist in the catalogue but have no carriers in our cohort
    discovered_positions = set(position_carriers.keys())
    positions_zero_carriers = len(all_who_positions - discovered_positions)
    
    print(f"  Total WHO positions for {drug}: {len(all_who_positions)}")
    print(f"  Discovered positions (with carriers): {len(discovered_positions)}")
    print(f"  Positions with 0 carriers: {positions_zero_carriers}")
    
    stats = {
        'Drug': drug,
        'Residues discovered': residues_discovered,
        'Total carriers': total_carriers,
        'Total R': total_r,
        'Total S': total_s,
        'Positions w/ 0 carriers': positions_zero_carriers,
        'Min carriers/position': min_carriers,
        'Median carriers/position': median_carriers,
        'Max carriers/position': max_carriers
    }
    
    print(f"\nResults for {drug}:")
    for key, value in stats.items():
        if key != 'Drug':
            print(f"  {key}: {value}")
    
    return stats


def main():
    """Main execution function."""
    print("="*80)
    print("Generating Task 2 Random Split Summary for DNA Regression")
    print("="*80)
    
    # Load data
    who_catalogue = load_who_catalogue()
    geno_pheno_df = load_genotype_phenotype_data()
    site_indices = load_site_indices()
    
    # Compute statistics for each drug
    # Note: We load isolate mappings on-demand per drug to save memory
    results = []
    for drug in DRUGS:
        try:
            stats = compute_drug_statistics(drug, geno_pheno_df, who_catalogue)
            results.append(stats)
        except Exception as e:
            print(f"Error processing {drug}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create output dataframe
    results_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save results
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"{'='*80}")
    
    # Display results
    print("\nSummary Table:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
