# Lineage-aware Evo2 interpretability

Mirrors the lineage-aware regression interpretability workflow at
[Regression/interpretability/lineage_aware_split](../../Regression/interpretability/lineage_aware_split),
but computes SHAP importance for the Evo2 `DNABERTCNN` downstream classifier
on token embeddings, following the same approach used at
[DNABERT-2/interpretability](../../DNABERT-2/interpretability) (`shap.DeepExplainer`
over the CNN, ranking `"<gene>_<position>"` features by max |SHAP| across
explained isolates). Every eligible held-out lineage is evaluated with the
exact model checkpoint trained for that lineage's test split. A lineage is
eligible when both its saved model (`DNABERTCNN.pt` by default -- the exact
checkpoint `eval_lineage_holdout.py` loads via `--saved_model_name` to
produce `test_set_auc_<DRUG>.csv`, not an intermediate early-stopping
checkpoint) and `test_set_auc_<DRUG>.csv` exist under
`training_output/lineage_aware_holdout/<DRUG>/`, at `seed_42` (the seed used
for all lineage-holdout evaluation runs).

## What differs from DNABERT-2's random-split interpretability

- **Input data processing**: embeddings are loaded through the Evo2
  `finetuning/modules` dataset classes (`TokenMemmapMap` /
  `MultiGeneConcatDataset`) instead of DNABERT-2's memmap loaders, and the
  Evo2/EVO2_DRUG_INDEX phenotype column mapping is enforced via
  `evo2_downstream.train.enforce_evo2_drug_index_mapping`. Because Evo2
  tokenizes one nucleotide per token (no BPE merging), a SHAP-important token
  position already equals the WHO/VCF gapped-alignment position -- no
  tokenizer offset-mapping step is required.
- **Per-lineage model loading**: instead of one model per drug, every
  eligible `heldout_lineage_<N>` checkpoint for the drug is loaded and
  explained separately (same logic difference as the Regression
  lineage-aware split vs. its random-split counterpart).
- **SHAP cohorts**: one phenotype-stratified explainer is selected per drug
  from the full deduplicated labelled dataset (seed 42, maximum 360) and is
  reused for every eligible lineage. Each lineage gets a separate background
  reconstructed from the exact 80% inner-training subset used to fit that
  model (excluding its 20% validation subset). Background
  candidates are deduplicated by `(embedding, phenotype)`, made disjoint from
  the explainer by both sample ID and signature, and capped at 160 with no
  lower bound.
- **Bounded-memory explanation**: explainer samples are loaded and evaluated
  in streaming batches, while the selected background is processed in
  weighted chunks. This avoids placing the complete explainer tensor,
  background tensor, and every full SHAP tensor on the GPU at once.
- **Parallel explainer shards**: the SLURM launcher defaults to four GPU tasks
  per lineage. Each task explains a disjoint quarter of the fixed explainer;
  the combine job takes the per-position maximum across shards, which is
  equivalent to taking the maximum across the complete explainer set.

## Usage

Run one configured drug:

```bash
/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python \
  run_interpret_evo2_lineage_aware.py parameter_files/shap_interpret_lineage_aware.yaml --drug STREPTOMYCIN
```

Run every drug that has at least one eligible lineage model:

```bash
./run_all_drugs.sh
```

`run_all_drugs.sh` uses the project's `dnabert_s` training environment by
default (the same environment used to train/evaluate Evo2 downstream
models). Override it with `PYTHON=/path/to/environment/bin/python
./run_all_drugs.sh`. SHAP's `DeepExplainer` needs a GPU in practice; run this
on a GPU node/allocation (same as `evaluate_lineage_holdout_classifier.sh`).
For the SLURM workflow, run `submit_lineage_aware_interpret.sh`; it submits a
CPU prep job, a dependent lineage-by-shard A100 array, and a final CPU merge.
Set `SHARD_COUNT` to override the default of four shards per lineage.

## Outputs

For each drug, outputs are written below `outputs/`:

- `map_mar/map_mar_<DRUG>_by_lineage.csv`: one row per eligible lineage and k value.
- `map_mar/map_mar_<DRUG>_mean.csv`: means across eligible lineages for every k value.
- `map_mar/relevant_features_<DRUG>.csv`: WHO/VCF resistant features used as ground truth.
- `shap_values/<DRUG>/heldout_lineage_<N>/ranked_shap.csv`: all `<gene>_<position>` features ranked by max |SHAP|.
- `shap_values/<DRUG>/heldout_lineage_<N>/selected_features.csv`: top `top_n_positions` ranked features fed into MAP/MAR.
- `shap_values/<DRUG>/explainer_samples.csv`: fixed drug-level explainer IDs and labels.
- `shap_values/<DRUG>/heldout_lineage_<N>/background_samples.csv`: exact lineage-specific background IDs and labels.
- `shap_values/<DRUG>/heldout_lineage_<N>/sample_selection_summary.csv`: sample counts and background label counts.
- `dedup_geno_data/<DRUG>_full_dedup_indices.npy`: cached deduplication indices for the drug's full dataset.

The k values and metric definitions (`P@k`, `R@k`, `MAP@k`, `Hits@k` for
k=1, 5, 10) are identical to the Regression lineage-aware split. Full
per-sample SHAP matrices are disabled by default (`save_shap_values: false`
in the parameter file) since they can be large; enable when needed.
