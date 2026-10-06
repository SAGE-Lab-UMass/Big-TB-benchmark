# Data Setup Guide for Evo2 LoRA Finetuning

## Overview

This guide explains how to prepare the required data files for running Evo2 LoRA finetuning on MTB drug resistance prediction.

## Required Data Files

### 1. Lineage Annotations

**File**: `BIG_TB_isolates_with_lineages.csv`

**Description**: Contains MTB lineage annotations for each isolate.

**Required columns**:
- `ROLLINGDB_ID`: Isolate identifier
- `Lineage`: MTB lineage (1, 2, 3, or 4)

**Default location**: `../../BIG_TB_isolates_with_lineages.csv` (relative to Evo2/)

**Example**:
```csv
ROLLINGDB_ID,Lineage
SAMEA104394571,2
SAMEA104394572,4
SAMEA104394573,1
```

**How to override**: Use `--lineage-csv /path/to/lineages.csv`

---

### 2. Genomic FASTA Files

**Directory**: Aligned gene sequences in FASTA format

**Description**: Per-gene FASTA files containing aligned DNA sequences for MTB genes associated with drug resistance.

**File naming convention**: `<gene_name>_aligned.fasta` or `<gene_name>.fasta`

**Required genes** (depends on drug):
- For ISONIAZID: `inhA.fasta`, `katG.fasta`
- For RIFAMPICIN: `rpoB.fasta`, `rpoC.fasta`
- For ETHAMBUTOL: `embC.fasta`, `embA.fasta`, `embB.fasta`
- For PYRAZINAMIDE: `pncA.fasta`
- For STREPTOMYCIN: `rpsL.fasta`, `rrs.fasta`, `gid.fasta`
- For AMIKACIN: `rrs.fasta`, `eis.fasta`
- For KANAMYCIN: `rrs.fasta`
- For CAPREOMYCIN: `rrs.fasta`, `rrl.fasta`, `tlyA.fasta`
- For LEVOFLOXACIN: `gyrB.fasta`, `gyrA.fasta`
- For MOXIFLOXACIN: `gyrB.fasta`, `gyrA.fasta`
- For ETHIONAMIDE: `inhA.fasta`, `ethA.fasta`, `ethR.fasta`

**Default location**: `data/aligned_fasta`

**Example FASTA format**:
```fasta
>SAMEA104394571
ATGCGATCGATCGATCGATCG...
>SAMEA104394572
ATGCGATCGATCGATCGATCG...
```

**How to override**: Use `--fasta-dir /path/to/fasta`

---

### 3. Genotype-Phenotype Data

**File**: `geno_pheno_full_combined.csv`

**Description**: Maps isolate IDs to drug resistance phenotypes.

**Required columns**:
- `Unnamed: 0` or first column: Isolate identifier (matches FASTA headers and lineage file)
- Drug columns: One column per drug (ISONIAZID, RIFAMPICIN, etc.) with values:
  - `0` = Resistant (R)
  - `1` = Susceptible (S)

**Location**: `data/multidrug_classification/training/geno_pheno_full_combined.csv`

**Example**:
```csv
,ISONIAZID,RIFAMPICIN,ETHAMBUTOL
SAMEA104394571,0,0,1
SAMEA104394572,1,0,0
SAMEA104394573,1,1,1
```

**How to override**: Use `--geno-pheno-csv /path/to/geno_pheno.csv`

---

## Data Preparation Checklist

Before running LoRA finetuning, ensure:

- [ ] Lineage CSV exists and contains all isolates
- [ ] FASTA files exist for all required genes for your target drug
- [ ] FASTA headers match isolate IDs in lineage CSV
- [ ] Genotype-phenotype CSV exists with drug resistance labels
- [ ] All three files use consistent isolate identifiers

---

## Testing Your Setup

Run a dry-run to verify data loading:

```bash
cd finetuning/lineage_holdout

python train_evo2_lora.py \
    --drug ISONIAZID \
    --heldout-lineage 2 \
    --dry-run \
    --geno-pheno-csv /path/to/your/geno_pheno.csv \
    --lineage-csv /path/to/your/lineages.csv \
    --fasta-dir /path/to/your/fasta
```

This will print split statistics without training, allowing you to verify:
- Data files load successfully
- Split sizes are reasonable
- Class balance is acceptable

---

## Common Issues

### Issue: "No FASTA file found for gene X"
**Solution**: Ensure the gene FASTA file exists in your `--fasta-dir` and matches the naming convention.

### Issue: "Isolate ID not found in lineage map"
**Solution**: Check that isolate IDs are consistent across all three data sources (FASTA headers, lineage CSV, geno-pheno CSV).

### Issue: "Insufficient samples for training"
**Solution**: This may occur if the held-out lineage contains most samples for a drug. Try a different lineage or different drug.

---

## Example: Complete Setup

```bash
# 1. Organize your data
mkdir -p /path/to/mtb_data/{fasta,metadata}

# 2. Place lineage annotations
cp BIG_TB_isolates_with_lineages.csv /path/to/mtb_data/metadata/

# 3. Place FASTA files
cp *.fasta /path/to/mtb_data/fasta/

# 4. Place genotype-phenotype data
# (Already in repository at data/multidrug_classification/training/)

# 5. Test dry-run
cd finetuning/lineage_holdout
python train_evo2_lora.py \
    --drug ISONIAZID \
    --heldout-lineage 2 \
    --lineage-csv /path/to/mtb_data/metadata/BIG_TB_isolates_with_lineages.csv \
    --fasta-dir /path/to/mtb_data/fasta \
    --dry-run

# 6. Run actual training
python train_evo2_lora.py \
    --drug ISONIAZID \
    --heldout-lineage 2 \
    --lineage-csv /path/to/mtb_data/metadata/BIG_TB_isolates_with_lineages.csv \
    --fasta-dir /path/to/mtb_data/fasta \
    --epochs 30
```
