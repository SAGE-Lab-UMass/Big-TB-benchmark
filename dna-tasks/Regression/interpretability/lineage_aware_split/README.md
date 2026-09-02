# Lineage-aware logistic-regression interpretability

This directory mirrors the random-split regression interpretability workflow,
but evaluates every eligible held-out lineage with the exact
`LogisticRegressionCV.model` used for that lineage's test run. A lineage is
eligible when both its saved model and `test_predictions.csv` exist.

Run one configured drug:

```bash
/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/cnn/bin/python \
  run_interpret_logreg_lineage_aware.py parameter_files/logreg_iters1000.txt
```

Run every drug that has at least one eligible lineage model:

```bash
./run_all_drugs.sh
```

`run_all_drugs.sh` uses the project's `cnn` environment by default. Override it
when needed with `PYTHON=/path/to/environment/bin/python ./run_all_drugs.sh`.

For each drug, the requested outputs are written below `outputs/map_mar/`:

- `map_mar_<DRUG>_by_lineage.csv`: one row per eligible lineage and k value.
- `map_mar_<DRUG>_mean.csv`: means across eligible lineages for every k value.

The k values and metric definitions are the same as random split: 1, 5, and 10,
with `P@k`, `R@k`, `MAP@k`, and `Hits@k`. Per-lineage mean absolute SHAP values
and selected feature rankings are stored below `outputs/shap_values/<DRUG>/`.
Full SHAP matrices are disabled by default because saving one for every lineage
can consume many gigabytes; set `save_shap_values: true` in the parameter file
when those matrices are required.
