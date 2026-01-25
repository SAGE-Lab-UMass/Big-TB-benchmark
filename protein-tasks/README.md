# Protein Tasks

This folder holds protein-level modeling, interpretability, and utilities built around ESM embeddings and baseline models for *M. tuberculosis* drug resistance.

## Layout
- `esm_models/` – main ESM workflows (CNN model, dataset loaders, SHAP, significance testing notebooks/scripts, utilities).
- `one_hot_encoded/` – one-hot CNN/transformer baselines.
- `regression/` – linear-model CV and precision/recall vs WHO catalog.
- `protein_translation/` – variant-to-protein translation utilities.
- `old_experiments/` – legacy notebooks kept for reference.
- `reorganize_cv_folds.py` – helper to tidy CV artifacts into drug/fold_* subdirs.
- `data/` – embeddings, labels, and results (not versioned; paths assumed to exist).

## Environment
- Python 3.10+, PyTorch (CUDA preferred), scikit-learn, shap, pandas, numpy, tqdm.
- Set `PYTHONPATH` to this folder when running scripts directly:
  ```bash
  cd protein-tasks
  export PYTHONPATH=$(pwd)
  ```

## Data expectations
- ESM token/mean/PCA embeddings under `data/latest/embeddings/{gene}/token[/PCA|/MEAN]`.
- Sequence CSVs under `data/latest/sequence_data_csv/{gene}_{DRUG}_combined_sequence_data.csv` with `Filename` and `Phenotype`.
- Results written under `data/latest/results/...` and `data/latest/cross_val/...`.
- Absolute paths in scripts currently point to `/project/pi_annagreen_umass_edu/mahbuba/Data-Curation-for-MTB/protein-tasks/...`; adjust if running elsewhere.

## Common workflows
### Train/evaluate CNNs (ESM)
- Use `esm_models/significance_testing.py` or the associated notebook.
- Picks up embeddings via `data_utils.py` / `esm_test_dataclasses.py`, trains `ProteinCNN1x1`, saves fold metrics and model weights under `data/latest/results/prediction/esm/{drug}_dim{K}`.

### SHAP interpretability (per residue)
- Run `esm_models/shap_esm.py` after training; it loads saved models, deduplicates datasets, splits background/explain sets, and writes SHAP pickles plus metadata under `data/latest/results/interpretability/{mode}_{dim}`.

### Regression baselines
- `regression/significance_testing_regression.py` runs outer CV across Lasso/Ridge/LogReg on pre-built feature matrices in `data/latest/feature_matrix_labels/`, writes per-fold metrics, coeffs, and PR@k vs WHO catalog.

### One-hot baselines
- `one_hot_encoded/*.py` and notebooks mirror the CNN/transformer pipelines using one-hot encodings.

### Organize CV artifacts
- To clean loose CV outputs into `drug/fold_*` layout:
  ```bash
  python reorganize_cv_folds.py --root data/latest/cross_val/<run_name>      # dry run
  python reorganize_cv_folds.py --root data/latest/cross_val/<run_name> --apply
  ```

## Notes and tips
- Special embedding dirs (e.g., gyrA/gyrB for levo/moxi, ethA/ethR/inhA for ethionamide) are resolved via `SPECIAL_DIRS` in `esm_models/data_utils.py`; update if you move data.
- `esm_models/significance_testing.py` imports helper datasets from `esm_models/utility/esm_sig_test_dataclasses.py`.
- Checkpoint and large result files are ignored by `.gitignore`; keep raw data outside version control.
