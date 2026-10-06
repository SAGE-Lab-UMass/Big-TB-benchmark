# Task 2 Random Split Summary Generation for DNA Regression

## Overview
This document describes the generation of the Task 2 random split summary statistics for DNA-level regression analysis, analogous to Table A in the protein-level `task2_random_vs_lineage_split.xlsx` file.

## Script Location
`/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Regression/interpretability/generate_task2_random_split_summary.py`

## Output Location
`/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Regression/interpretability/data/task2_random_split_summary.csv`

## Methodology

### Data Sources
1. **WHO Catalogue**: WHO resistance variants (2023) filtered for high-confidence associations
   - Confidence levels: "1) Assoc w R" and "2) Assoc w R - Interim"
   - Path: `/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/data/WHO_catalogue_2023/WHO_resistance_variants_all_2023.csv`

2. **Genotype-Phenotype Data**: Combined dataset with isolate genotypes and phenotypes
   - Path: `/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/Regression_l2/input_data/reduced_prepared_data/combined_geno_pheno_df.csv`
   - 17,942 isolates total

3. **VCF-WHO Mappings**: Per-isolate variant mappings to WHO catalogue
   - Directory: `/project/pi_annagreen_umass_edu/saishradha/project_data_curation/vcf_who_mapped_data/`
   - Files: `{isolate_id}_variants.csv`

### Analysis Approach
1. For each drug, load isolates with non-null phenotype data
2. For each isolate, load its VCF-WHO mapping file
3. Filter for resistance variants (R) with confidence 1 or 2 from the WHO catalogue
4. Extract gene-relative positions using `rel_gapped_mutation_position_pos_strand` or `rel_gapped_mutation_position_neg_strand`
5. Count unique positions (residues discovered) and total isolate-position pairs (carriers)
6. Split carriers by phenotype (1.0 = R, 0.0 = S)
7. Compute distribution statistics (min/median/max carriers per position)

### Column Descriptions
- **Drug**: Antibiotic name
- **Residues discovered**: Number of unique nucleotide positions from WHO 2023 catalogue (confidence 1-2) that have at least one carrier in the cohort
- **Total carriers**: Total count of isolate-position pairs where an isolate carries a catalogued resistance variant
- **Total R**: Carriers with resistant phenotype (phenotype = 1.0)
- **Total S**: Carriers with susceptible phenotype (phenotype = 0.0)
- **Positions w/ 0 carriers**: Currently set to 0 (placeholder - would require complete WHO-to-genotype mapping)
- **Min carriers/position**: Minimum number of carriers for any discovered position
- **Median carriers/position**: Median number of carriers across all discovered positions
- **Max carriers/position**: Maximum number of carriers for any discovered position

## Results Summary

| Drug | Residues discovered | Total carriers | Total R | Total S |
|------|---------------------|----------------|---------|---------|
| Rifampicin | 52 | 5,126 | 291 | 4,835 |
| Isoniazid | 66 | 6,346 | 243 | 6,103 |
| Ethambutol | 9 | 3,198 | 810 | 2,388 |
| Pyrazinamide | 221 | 2,096 | 348 | 1,748 |
| Streptomycin | 84 | 2,213 | 412 | 1,801 |
| Moxifloxacin | 13 | 377 | 98 | 279 |
| Levofloxacin | 8 | 90 | 18 | 72 |
| Ethionamide | 126 | 1,182 | 484 | 698 |
| Capreomycin | 24 | 587 | 119 | 468 |
| Kanamycin | 8 | 648 | 60 | 588 |

## Comparison with Protein Analysis (Table A)

### Key Differences
1. **Granularity**: DNA analysis operates at the nucleotide level vs. amino acid level for protein
2. **Residues discovered**: Generally MORE positions in DNA analysis (multiple nucleotides per amino acid)
3. **Kanamycin**: Included in DNA analysis but not in protein Table A (protein gene mapping unavailable)
4. **Total carriers**: Similar magnitudes but different exact counts due to level of analysis

### Phenotype Distribution Note
⚠️ **Important**: The R/S distribution in DNA analysis shows a different pattern than protein analysis:
- **Protein**: Most carriers are phenotypically resistant (expected for resistance variants)
- **DNA**: Most carriers appear phenotypically susceptible

**Possible explanations**:
1. Phenotype encoding may be reversed (please verify: 1.0 = R, 0.0 = S)
2. DNA-level variants may be more nuanced (some nucleotide changes may not cause resistance)
3. The cohort composition may differ between analyses

**Recommendation**: Verify the phenotype encoding in the combined_geno_pheno_df.csv file.

## How to Run

```bash
cd /project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Regression/interpretability
python3 generate_task2_random_split_summary.py
```

## Dependencies
- pandas
- numpy
- tqdm

## Execution Time
- Approximately 2-5 minutes per drug (varies by number of isolates with phenotype data)
- Total runtime: ~20-30 minutes for all 10 drugs

## Future Enhancements
1. **Lineage Split Analysis**: Extend to generate Table B (leave-one-lineage-out statistics)
2. **Zero Carriers**: Compute exact count of WHO positions with no carriers in cohort
3. **Phenotype Verification**: Add automated check for phenotype encoding
4. **Caching**: Implement caching for frequently-loaded VCF-WHO mapping files
5. **Parallel Processing**: Add multiprocessing for faster execution

## Author
Generated via GitHub Copilot
Date: 2026-09-01
