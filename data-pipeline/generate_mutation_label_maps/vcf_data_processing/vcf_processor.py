NUCLEOTIDE_MUTATION_ANNOTATIONS = {
    'synonymous_variant', 
    'intergenic_region',
    'intragenic_variant',
    'feature_ablation', # the formatting is gene_name+deletion
    'bidirectional_gene_fusion',
    'gene_fusion',
    'duplication',
    'upstream_gene_variant',
    'non_coding_transcript_exon_variant',
    'initiator_codon_variant',
    'splice_region_variant',
    'stop_retained_variant',
}

# TODO: Fix this LoF category, the WHO looks different for these annotations
WHO_LoF_MUTATION_ANNOTATIONS = {
    'frameshift_variant',
    'stop_gained',
    'stop_lost',
    'start_lost',
    'feature_ablation',
    'inframe_deletion',
    'inframe_insertion',
}

# TODO: Fix this mixed mutation annotation in future WHO
MIXED_MUTATION_ANNOTATIONS = {
    'splice_region_variant',
    'start_retained_variant',
}

WHO_ANNOTATIONS = {
    'synonymous_variant', 
    'upstream_gene_variant',
    'stop_retained_variant', 
    'feature_ablation', 
    'missense_variant', 
    'non_coding_transcript_exon_variant',
    'frameshift',   # LOF starts here
    'stop_gained',
    'start_lost',
    'stop_lost',
    'inframe_insertion', 
    'inframe_deletion', 
    'initiator_codon_variant', 
    'LoF',
}

def is_snp_variant(annotation_str):
    """Returns True if all annotations in the string are a subset of NUCLEOTIDE_MUTATION_ANNOTATIONS."""
    annotations = set(annotation_str.split('&'))
    return annotations.issubset(NUCLEOTIDE_MUTATION_ANNOTATIONS)

def process_vcf(vcf_df):
    """Processes VCF file for mapping by renaming columns and filtering relevant mutations."""
    na_rows = vcf_df[vcf_df['POSITION'].isna()]
    vcf_df['POSITION'] = vcf_df['POSITION'].astype(int)

    # Classify variants based on annotation logic
    snp_mask = vcf_df['Annotation'].apply(is_snp_variant)

    snp_variants = vcf_df[snp_mask].copy()
    protein_variants = vcf_df[~snp_mask].copy()

    # Add vcf_variant column
    # If feature_ablation is present, use 'deletion' instead of HGVS.c
    snp_variants['vcf_variant'] = snp_variants.apply(
        lambda row: f"{row['Gene_Name']}_deletion"
        if 'feature_ablation' in row['Annotation'].split('&')
        else f"{row['Gene_Name']}_{row['HGVS.c']}",
        axis=1
    )
    
    # snp_variants['vcf_variant'] = snp_variants['Gene_Name'] + "_" + snp_variants['HGVS.c']
    protein_variants['vcf_variant'] = protein_variants['Gene_Name'] + "_" + protein_variants['HGVS.p']

    # Rename columns
    rename_dict = {
        'Gene_ID': 'GENE_ID', 'Gene_Name': 'GENE',
        'Annotation': 'MUTATION_EFFECT', 'HGVS.c': 'NUCLEOTIDE(S)_CHANGE',
        'HGVS.p': 'PROTEIN_CHANGE'
    }
    snp_variants.rename(columns=rename_dict, inplace=True)
    protein_variants.rename(columns=rename_dict, inplace=True)

    return protein_variants, snp_variants
















# NUCLEOTIDE_MUTATION_ANNOTATIONS = [
#     'synonymous_variant', 
#     'intergenic_region',
#     'intragenic_variant',
#     'feature_ablation',
#     'bidirectional_gene_fusion',
#     'gene_fusion',
#     'duplication',
#     'gene_variant',
# ]

# MIXED_MUTATION_ANNOTATIONS = [
#     'splice_region_variant',
#     'start_retained_variant',
#     'stop_retained_variant',
# ]

# def process_vcf(vcf_df):
#     """Processes VCF file for mapping by renaming columns and filtering relevant mutations."""
#     na_rows = vcf_df[vcf_df['POSITION'].isna()]
#     print("NaN rows:", na_rows)
#     vcf_df['POSITION'] = vcf_df['POSITION'].astype(int)
#     # vcf_df['Gene_Name'] = vcf_df['Gene_Name'].fillna('Unknown')

#     # Filter variants that cause protein changes
#     protein_variants = vcf_df[vcf_df['Annotation'] == 'missense_variant'].copy()
#     protein_variants['vcf_variant'] = protein_variants['Gene_Name'] + "_" + protein_variants['HGVS.p']

#     # Filter variants that do not cause protein changes
#     snp_variants = vcf_df[vcf_df['Annotation'] == 'synonymous_variant'].copy()
#     snp_variants['vcf_variant'] = snp_variants['Gene_Name'] + "_" + snp_variants['HGVS.c']
    
#     # Rename columns
#     rename_dict = {
#         'Gene_ID': 'GENE_ID', 'Gene_Name': 'GENE',
#         'Annotation': 'MUTATION_EFFECT', 'HGVS.c': 'NUCLEOTIDE(S)_CHANGE',
#         'HGVS.p': 'PROTEIN_CHANGE'
#     }
#     protein_variants.rename(columns=rename_dict, inplace=True)
#     snp_variants.rename(columns=rename_dict, inplace=True)
    
#     return protein_variants, snp_variants
