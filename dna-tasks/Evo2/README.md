# Evo2 for MTB Drug Resistance Prediction

This repository contains code for Evo2-based approaches to Mycobacterium tuberculosis (MTB) drug resistance prediction, including:

1. **Embedding generation** for zero-shot downstream models
2. **LoRA fine-tuning** for end-to-end supervised learning on held-out lineages

## Repository Structure

```
Evo2/
├── finetuning/
│   ├── modules/                  # Vendored finetuning utilities
│   │   ├── dataloader/
│   │   │   └── locus_order.py    # Drug-to-loci mappings
│   │   └── downstream_cnn_model.py
│   ├── lineage_holdout/          # LoRA fine-tuning implementation
│   │   ├── train_evo2_lora.py    # Main training script
│   │   ├── lineage_split.py      # Lineage-aware data splitting
│   │   ├── lora_data.py          # Data loading utilities
│   │   ├── run_evo2_finetuning_per_lineage.sh  # SLURM launcher
│   │   └── LORA_README.md        # Detailed LoRA documentation
│   └── verify_setup.py           # Finetuning setup validator
│
├── evo2_embed_gen/               # Embedding generation package
│   ├── model/                    # Evo2 model wrapper
│   ├── data/                     # FASTA data utilities
│   └── embeddings/               # Embedding CLI
│
├── evo2_downstream/              # Zero-shot downstream training
│   ├── config.py                 # Path configuration
│   ├── train.py                  # Classifier training
│   └── eval.py                   # Evaluation
│
├── data/
│   └── multidrug_classification/
│       └── training/
│           └── geno_pheno_full_combined.csv  # Genotype-phenotype data
│
├── setup_evo2_env.sh            # Environment setup
├── requirements.txt             # Base dependencies
├── requirements_lora.txt        # LoRA-specific dependencies
└── README.md                    # This file
```

## External Data Requirements

The following data files must be provided separately:

### 1. Lineage Annotations
**File**: `BIG_TB_isolates_with_lineages.csv`
**Default location**: `../../BIG_TB_isolates_with_lineages.csv` (relative to Evo2/)
**Contains**: MTB lineage annotations for each isolate
**Override via**: `--lineage-csv /path/to/lineages.csv`

### 2. Genomic FASTA Files
**Directory**: Aligned gene sequences in FASTA format
**Default location**: `data/aligned_fasta`
**Contains**: Per-gene FASTA files (e.g., `rpoB_aligned.fasta`, `katG_aligned.fasta`, etc.)
**Override via**: `--fasta-dir /path/to/fasta`

### 3. Genotype-Phenotype Data
**File**: `geno_pheno_full_combined.csv`
**Location**: `data/multidrug_classification/training/geno_pheno_full_combined.csv`
**Contains**: Isolate IDs and drug resistance phenotypes
**Override via**: `--geno-pheno-csv /path/to/geno_pheno.csv`

## Quick Start

### 1. Environment Setup

```bash
cd path/to/Evo2

# Create base Evo2 environment
bash setup_evo2_env.sh

# Install LoRA dependencies
conda activate <your-evo2-env>
pip install -r requirements_lora.txt
```

### 2. LoRA Fine-tuning

See [finetuning/lineage_holdout/LORA_README.md](finetuning/lineage_holdout/LORA_README.md) for detailed documentation.

**Quick example** (single drug, single lineage holdout):

```bash
cd finetuning/lineage_holdout

python train_evo2_lora.py \
    --drug ISONIAZID \
    --heldout-lineage 2 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-lr 1e-4 \
    --classifier-lr 1e-3 \
    --epochs 30 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --use-amp \
    --gradient-checkpointing
```

**Batch training** (all 4 lineage holdouts):

```bash
cd finetuning/lineage_holdout
sbatch run_evo2_finetuning_per_lineage.sh ISONIAZID 1  # Lineage 1
sbatch run_evo2_finetuning_per_lineage.sh ISONIAZID 2  # Lineage 2
sbatch run_evo2_finetuning_per_lineage.sh ISONIAZID 3  # Lineage 3
sbatch run_evo2_finetuning_per_lineage.sh ISONIAZID 4  # Lineage 4
```

### 3. Custom Data Paths

If your data is in different locations, override the defaults:

```bash
python train_evo2_lora.py \
    --drug RIFAMPICIN \
    --heldout-lineage 2 \
    --geno-pheno-csv /path/to/your/geno_pheno.csv \
    --lineage-csv /path/to/your/lineages.csv \
    --fasta-dir /path/to/your/fasta_files \
    --output-dir /path/to/output
```

### 4. Dry-run Mode

Check data splits without training:

```bash
python train_evo2_lora.py \
    --drug ISONIAZID \
    --heldout-lineage 2 \
    --dry-run
```

## Self-Contained Deployment

This repository is **self-contained** for LoRA fine-tuning:

- ✅ **Finetuning utility modules vendored** in `finetuning/modules/`
- ✅ **No external repository dependencies** for model code
- ✅ **Configurable data paths** via command-line arguments

Only the **data files** (FASTA, lineages, phenotypes) need to be provided separately.

## Embedding Generation (Zero-shot)

For zero-shot embedding generation:

```bash
# Single gene embedding
sbatch run_embed_gen_sbatch.sh

# Or non-SLURM:
bash run_evo2_embed_gen.sh
```

Default outputs are saved to:
```
embeddings/zero-shot/token/train/new/
```

## Supported Drugs

The following drugs are supported (defined in `finetuning/modules/dataloader/locus_order.py`):

- ISONIAZID
- RIFAMPICIN
- ETHAMBUTOL
- PYRAZINAMIDE
- STREPTOMYCIN
- KANAMYCIN
- AMIKACIN
- CAPREOMYCIN
- LEVOFLOXACIN
- MOXIFLOXACIN
- ETHIONAMIDE

Each drug is mapped to its associated resistance genes automatically.

