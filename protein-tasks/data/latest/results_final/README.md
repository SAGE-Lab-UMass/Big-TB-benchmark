# results_final (protein)

Organized copies of the actual result files behind each numbered table/figure in the current
manuscript (`latex_paper/BigTB_revision_summer26.tex`), so it's clear which real, script-generated
file backs each reported number. One folder per table/figure, named after its current number and
label. Files inside each folder are untouched copies of the originals under `protein-tasks/data/latest/`
(paths noted below); nothing here is a new computation, only a reorganization for traceability.

- `table3_combined_auc_summary/` -- Task 1 headline AUC (Table 3, protein half). `Combined2_test_aucs_Table.csv`
  is the combined table; the other 4 files are its per-model raw sources.
- `table4_perc_discovered_combined/` -- Task 2 headline Precision@10/Recall@10 (Table 4, protein half),
  plus its new pooled significance test vs. LogReg.
- `table5_combined_lineage_mean_auc/` -- Task 1 lineage-holdout AUC (Table 5, protein half): raw
  per-drug/per-architecture `all_lineage_summary.csv` files, plus the new lineage-vs-random significance test.
- `table6_lineage_shap_combined/` -- Task 2 lineage-holdout Precision@10/Recall@10 delta (Table 6, protein
  half), plus its significance test.
- `table16_per_drug_vs_baseline_protein/` -- the per-drug/per-fold CV significance test (formerly Table 16,
  now removed from the manuscript -- see note below).
- `table17_pooled_model_vs_logreg_protein/` -- Task 1 pooled significance test vs. LogReg, with Holm-Bonferroni.
- `table18_shap_cnn_detailed_trim/` -- WHO variant recovery detail table.
- `figure2_prediction_cv_result/`, `figure4b_esm_variants_ablation/`, `figure6_esm_embeddings_ablation_heatmap/`
  -- source data for the corresponding figures.

Note: Table 16 (and its DNA counterpart, Table 15) were removed from the manuscript on 2026-09-21 --
with only 5 CV folds, that per-drug/fold test can never reach p<0.05 regardless of effect size, and the
pooled test (Table 17) now covers this properly. The file is kept here for reference/traceability, not
because it's still cited in the paper.

For DNA: this folder only covers the protein side. The equivalent DNA tables/figures (Table 3 DNA half,
Table 4 DNA half, Table 5 DNA half, Table 6 DNA half, Figure 4a/5 DNA embedding ablation) still need
their own `results_final`-style folder built the same way -- one subfolder per table/figure, containing
the actual script-generated file(s) behind it, not a re-derivation.
