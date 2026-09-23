# Finetuned Evo2 interpretability

Lineage-aware SHAP interpretability for the LoRA-finetuned Evo2 models under
`training_output/lora_finetuned/<DRUG>/final/heldout_lineage_<N>/best`
(`lora_adapter.pt` + `classifier_head.pt` + `training_config.json`, produced by
`finetuning/lineage_holdout/train_evo2_lora.py`). Every eligible held-out
lineage is evaluated with the exact finetuned checkpoint saved for that
lineage's held-out test split -- the `best` checkpoint selected by validation
ROC-AUC during LoRA training, not an intermediate resume checkpoint.

Unlike the zero-shot Evo2 downstream classifier, the finetuned model computes
its own token hidden states from raw DNA sequence (frozen Evo2 backbone +
trainable LoRA adapters), which are then fed to a `DNABERTCNN` classifier
head. This directory therefore loads raw aligned FASTA sequences directly
(via `finetuning/lineage_holdout/utils/lora_data.py`) instead of precomputed
token memmaps, reconstructs each finetuned model
(`Evo2LoRAClassifier` + `load_adapter_and_classifier`), and runs
`shap.DeepExplainer` over the classifier head using materialized hidden-state
tensors as SHAP's model input.

A lineage is eligible when `training_config.json`, `lora_adapter.pt`,
`classifier_head.pt` (under `.../heldout_lineage_<N>/best/`), and
`test_metrics.json` (under `.../heldout_lineage_<N>/`) all exist. Set
`require_test_metrics: false` in the parameter file to relax the last
requirement.

## Background/explainer sample selection

Sample selection matches
[interpretability/lineage_aware_split](../lineage_aware_split): one
phenotype-stratified explainer is selected per drug from the full
deduplicated (by sequence + label) dataset (seed 42, 20% stratified split,
capped at 360), and reused for every eligible lineage. Each lineage's
background is reconstructed from the exact inner-training subset used to fit
that lineage's model (lineage-aware train/test split, then the same
stratified validation split used during training, excluding the held-out
lineage and the validation subset), deduplicated against the fixed explainer
by sequence/phenotype signature and capped at 160.

## Usage

Run one configured drug directly:

```bash
/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/dnabert_s/bin/python \
  run_interpret_evo2_finetuned.py parameter_files/shap_interpret_finetuned.yaml --drug ISONIAZID
```

For the SLURM workflow, run `submit_finetuned_interpret.sh`; it discovers
every eligible held-out lineage for the drug and submits a CPU prep job, a
dependent lineage-by-shard A100 array, and a final CPU merge job:

```bash
./submit_finetuned_interpret.sh ISONIAZID

# Override the number of parallel explainer shards per lineage (default: 4):
SHARD_COUNT=8 ./submit_finetuned_interpret.sh ISONIAZID
```

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
- `dedup_sequence_data/<DRUG>_full_dedup_indices.npy`: cached deduplication indices for the drug's full sequence dataset.

`save_shap_values: false` in the parameter file disables persisting the full
per-sample SHAP matrix (can be large); enable it when needed.
