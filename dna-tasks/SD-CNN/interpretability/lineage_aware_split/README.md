# Lineage-aware SD-CNN interpretability

This directory mirrors the random-split SD-CNN SHAP and MAP/MAR workflow. For
each drug it evaluates every lineage marked `feasible=true` in
`all_drugs_split_counts.csv`, using that lineage's finalized
`saved_model/sd-cnn_model_lineage_holdout.h5` model (the same model used to
create `test_predictions.csv`). Stale models left under ineligible lineage
directories are ignored.

For isoniazid, both `INH` and `ISONIAZID` training-output folders are checked;
the folder with the newest finalized lineage model is selected. With the current
outputs this is `INH`.

Run one drug:

```bash
/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/cnn/bin/python \
  run_interpret_lineage_aware.py \
  parameter_files/lineage_aware_interpretability.yaml \
  --drug AMIKACIN
```

Run all drugs with at least one eligible lineage:

```bash
./run_all_drugs.sh
```

Set `PYTHON=/path/to/python` to use a different environment. Add `--force` to a
single-drug command to invalidate matching per-lineage SHAP summary caches.

For each drug the requested files are written below `outputs/map_mar/`:

- `map_mar_<DRUG>_by_lineage.csv`: one row for every eligible lineage and k.
- `map_mar_<DRUG>_mean.csv`: mean metrics across eligible lineages for each k.

The k values and metric definitions match random split: 1, 5, and 10, with
`P@k`, `R@k`, `MAP@k`, and `Hits@k`. Per-lineage mean absolute SHAP values,
selected features, and plots are stored under
`outputs/shap_values/<DRUG>/heldout_lineage_<N>/`.
