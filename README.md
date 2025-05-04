
# GeRM: a benchmark dataset for Genomic Resistance prediction and interpretability in *Mycobacterium tuberculosis*

This repository provides a reproducible benchmark for genotype-to-phenotype prediction of drug resistance in *M. tuberculosis*, combining classical and deep learning models on DNA and protein sequences. It includes end-to-end code for:
- DNA to protein translation,
- Sequence Data preparation for ML pipeline,
- Model training and evaluation,
- Per-residue interpretability using occlusion.

---

##  Dataset Overview

We work with gene- or operon-specific datasets extracted from whole-genome sequencing data of *M. tuberculosis* isolates, annotated with binary drug resistance phenotypes.

- Genes covered: 'rpoB','rpsL','tlyA','pncA','eis','gid','katG','inhA','embA','embB', 'embC','gyrB', 'gyrA', 'ethA', 'ethR'
- Drug phenotypes: RIF, INH, ETH, PZA, STM, CM, AMC, KAN,EMB.
- Variant annotations aligned with the **WHO mutation catalog**.

Each gene-specific CSV file includes:
- `Filename`: Isolate filenames,
- `Sequence`: Nucleotide Sequence,
- `Phenotype`: Resistant or Susceptible Phenotypes
- `seq len`: Nucleotide Sequence length (aligned)
- `Protein_Sequence`: translated amino acid string (with optional masking of ambiguous residues)
- `Frameshift_Mutation`: boolean flag for quality filtering


---

## Pipeline Overview

### 1.  Protein Translation from Variant FASTA
Scripts to extract and translate gene/operon regions from aligned variant FASTA files.

- Handles CDS slicing, operon-aware boundaries
- Gap-aware coordinate mapping with validation against MycoBrowser


### 2. Model Training

####  Classical ML
- Logistic Regression, Ridge, Lasso (variant encodings)
- KNN, SVR, Random Forest (on ESM embeddings)

####  Deep Learning
- 1D CNNs and Transformers (trained on one-hot encoded sequences)

Each model is evaluated:
- Per antibiotic
- Across different sequence representations

### 3.  Interpretability

#### Leave-One-Residue-Out Occlusion
- Compute per-residue importance by re-evaluating model performance after masking each amino acid
- Precision, recall, and F1@k measured against WHO causal variant sets

#### Outputs
- Heatmaps of residue importance
- Top-k residue lists
- Precision-recall-F1 plots stratified by gene, drug, and model

---

## Directory Structure

```


````

---

## Getting Started

### Dependencies
Install via pip:
```bash
pip install -r requirements.txt
````

Includes:

* `biopython`, `pandas`, `numpy`
* `scikit-learn`, `torch`, `transformers`, `fair-esm`
* `matplotlib`, `seaborn`

### ESM Setup (optional)

To generate ESM2 embeddings:

```bash
git clone https://github.com/facebookresearch/esm
cd esm
pip install -e .
```

---

## Reproducing Results

1. **Prepare data:**

   * Extract variant-aligned CDS regions (see `translation/`)
   * Generate protein sequences and label files

2. **Generate features:**


3. **Train models:**



4. **Compute interpretability:**



---

## Citation

If you use this benchmark in your work, please cite:

```
[placeholder]. GeRM: a benchmark dataset for Genomic Resistance prediction and interpretability in Mycobacterium tuberculosis.
```

---

## Acknowledgments

* WHO mutation catalog (2023)
* ESM protein language models (Meta AI)
* Research support from SAGE Lab @ UMass Amherst

---



