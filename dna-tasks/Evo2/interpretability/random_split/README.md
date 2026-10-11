# Random-split Evo2 interpretability

Mirrors [interpretability/lineage_aware_split](../lineage_aware_split), but
for Evo2 `DNABERTCNN` classifiers trained with a **zero-shot random split**
(`training_output/zero_shot/random_split`) instead of a lineage-aware
leave-one-lineage-out split. SHAP importance is computed the same way
(`shap.DeepExplainer` over the CNN, ranking `"<gene>_<position>"` features by
max |SHAP| across explained isolates, mapped to WHO/VCF resistant features
for MAP/MAR), but two things differ from `lineage_aware_split`:

- **Background/explainer selection**: random-split training does not hold
  out a lineage, so there is no need for a per-model background
  reconstructed from that model's exact training subset. Instead, the
  background and explainer are chosen together in **one** stratified split
  drawn directly from the drug's deduplicated dataset, matching
  [SD-CNN/interpretability](../../../SD-CNN/interpretability)'s
  `utils.py::compute_shap_values_strat` (20% stratified background capped at
  160 samples; the remainder is the explainer pool). The explainer pool is
  then additionally capped at **360** samples using the same stratified
  subsampling style as `lineage_aware_split`'s fixed explainer selection.
- **Model selection**: random-split training produces several independent
  folds per drug (`training_output/zero_shot/random_split/<DRUG>/saved_models/evo2/token/<DRUG>/seed_42/fold_<N>/`).
  Instead of evaluating every fold, a **single** representative fold is
  chosen automatically: the fold whose training history
  (`classification_results/.../DNABERTCNN_fold<N>_history.csv`) has the
  highest `val_auc` value. If no fold for that drug has a training history
  with a usable `val_auc` column (this happens when a drug's history CSVs
  were never generated), the first available fold is used instead. The
  chosen fold is frozen to `outputs/shap_values/<DRUG>/fold_selection.json`
  the first time it is resolved, so parallel explainer-shard array tasks
  agree on the same fold and cache files without re-deriving the choice.

Checkpoint selection within a fold defaults to `model_filename: auto`, which
prefers the final saved checkpoint (`DNABERTCNN.pt`) and falls back to the
early-stopping checkpoint (`DNABERTCNN_best_model.pt`) -- matching the
convention used for the zero-shot random-split evaluation jobs, since not
every fold has a final `DNABERTCNN.pt`.

## Usage

Run one configured drug directly:

```bash
/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python \
  run_interpret_evo2_random_split.py parameter_files/shap_interpret_random_split.yaml --drug STREPTOMYCIN
```

Run every drug found under `training_output/zero_shot/random_split`:

```bash
./run_all_drugs.sh
```

For the SLURM workflow, run `submit_random_split_interpret.sh`; it submits a
prep job (selects the fold, caches dedup indices and the background/explainer
split), a dependent explainer-shard array job, and a final merge job:

```bash
./submit_random_split_interpret.sh AMIKACIN

# Override the automatic fold selection:
FOLD=3 ./submit_random_split_interpret.sh AMIKACIN

# Override the number of parallel explainer shards (default: 4):
SHARD_COUNT=8 ./submit_random_split_interpret.sh AMIKACIN
```

## Outputs

For each drug, outputs are written below `outputs/`:

- `map_mar/map_mar_<DRUG>.csv`: one row per k value for the selected fold.
- `map_mar/relevant_features_<DRUG>.csv`: WHO/VCF resistant features used as ground truth.
- `shap_values/<DRUG>/fold_selection.json`: the fold chosen, the checkpoint used, the best `val_auc` found (or `null` on fallback), and the selection reason.
- `shap_values/<DRUG>/fold_<N>/ranked_shap.csv`: all `<gene>_<position>` features ranked by max |SHAP|.
- `shap_values/<DRUG>/fold_<N>/selected_features.csv`: top `top_n_positions` ranked features fed into MAP/MAR.
- `shap_values/<DRUG>/fold_<N>/background_samples.csv` / `explainer_samples.csv`: the frozen background/explainer sample IDs and labels.
- `shap_values/<DRUG>/fold_<N>/sample_selection_summary.csv`: sample counts and background label counts.
- `dedup_geno_data/<DRUG>_full_dedup_indices.npy`: cached deduplication indices for the drug's full dataset.

The k values and metric definitions (`P@k`, `R@k`, `MAP@k`, `Hits@k` for
k=1, 5, 10) are identical to `lineage_aware_split`. Full per-sample SHAP
matrices are disabled by default (`save_shap_values: false` in the parameter
file) since they can be large; enable when needed.
