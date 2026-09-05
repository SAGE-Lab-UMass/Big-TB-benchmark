"""Task 2 (WHO-catalogued resistance variant recovery) under lineage holdout,
for the Ref-Alt / SHAP logistic-regression pipeline.

NOTE - output not used for reporting: this script explains SHAP on the
HELD-OUT LINEAGE's own test partition, which was a useful engineering step
(and mirrors lineage_holdout_task2_cnn.py/_transformer.py/_esm.py's design)
but is NOT the metric the team agreed to report. Per the 2026-09 agreement
with Sai/Anna/Mahbuba, Task 2's lineage-aware result uses the FIXED
random-split (Table 3) explain set with the lineage-trained model swapped
in - see controlled_shap_task2_shap_regression.py for that actual
deliverable. This file's own Precision@k/Recall@k output should not be
presented as "the" Task 2 lineage-aware result.

SHAP: shap.LinearExplainer with an Independent (interventional) masker.
Residues from all of a multi-gene drug are ranked together in ONE combined
list (max |SHAP| per feature column across the explained samples), matching
the CNN/Transformer/ESM combined-ranking methodology in interp_pr_utils.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROTEIN_TASKS_DIR))
sys.path.insert(0, str(THIS_DIR))

from lineage_split_utils import DEFAULT_MIN_CLASS_COUNT, MAJOR_LINEAGES
from regression_utils import DRUG2GENES, gene_slices, load_catalog
from lineage_holdout_regression import _prepare_lineage_annotated_subset
from shap_pr_utils import (
    build_hit_variants,
    druglevel_gold_excl,
    greedy_topk_global,
    linear_shap_explain,
    stratified_bg_explain_indices,
)

OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/regression_task2_shap'
WHO_CATALOG = PROTEIN_TASKS_DIR / 'data/filtered_variants_output.csv'
ALLOWED_CONF = ['1) Assoc w R', '2) Assoc w R - Interim']
MODEL_NAME = 'logreg_shap_independent'

ELIGIBLE_DRUGS = ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'streptomycin',
                  'capreomycin', 'moxifloxacin', 'ethionamide']


def run_lineage_task2_shap_for_drug(drug: str, heldout_lineage: str, k_vals=(1, 5, 10),
                                     seed: int = 42, bg_frac: float = 0.10, max_bg: int = 160):
    if drug not in DRUG2GENES:
        raise NotImplementedError(f'{drug} is not supported by the regression SHAP lineage runner')

    X, y, manifest = _prepare_lineage_annotated_subset(drug)
    test_mask = manifest['Lineage'].astype(str) == str(heldout_lineage)
    train_mask = ~test_mask

    train_labels = manifest.loc[train_mask, 'saved_label']
    test_labels = manifest.loc[test_mask, 'saved_label']
    train_r = int((train_labels == 'R').sum())
    train_s = int((train_labels == 'S').sum())
    test_r = int((test_labels == 'R').sum())
    test_s = int((test_labels == 'S').sum())

    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te = X[test_idx]

    model = LogisticRegressionCV(
        cv=3, scoring='roc_auc', max_iter=5000,
        Cs=[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
        class_weight='balanced',
    )
    model.fit(X_tr, y_tr)

    bg_idx, _ = stratified_bg_explain_indices(y_tr, bg_frac=bg_frac, seed=seed, max_bg=max_bg)
    phi = linear_shap_explain(model, X_tr[bg_idx], X_te, masker_mode='independent')

    scores_global = np.abs(phi).max(axis=0)
    rank_df = (pd.DataFrame({
        'Residue_Position': np.arange(scores_global.shape[0]),
        'MaxAbsSHAP': scores_global,
    }).sort_values('MaxAbsSHAP', ascending=False).reset_index(drop=True))

    slices = gene_slices(drug, X.shape[1])
    who_df = load_catalog(WHO_CATALOG, ALLOWED_CONF)
    catalog_full = pd.read_csv(WHO_CATALOG)
    catalog_full['aa_pos_0idx'] = catalog_full['aa_pos'].astype(int) - 1
    gold_global, excl_global = druglevel_gold_excl(slices, who_df, catalog_full)
    n_true = len(gold_global)

    out_dir = OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(out_dir / f'{drug}_{MODEL_NAME}_ranked_SHAP_DRUG.csv', index=False)

    rows = []
    for k in k_vals:
        topk = greedy_topk_global(rank_df, k, exclude=excl_global)
        tp = len(gold_global & set(topk))
        prec = tp / len(topk) if topk else 0.0
        rec = tp / n_true if n_true else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) else 0.0
        hit_vars = build_hit_variants(topk, slices, who_df) or ['None']
        rows.append({
            'drug': drug, 'gene': 'ALL', 'model': MODEL_NAME, 'k': k,
            'Total_Resistance_Positions': n_true, 'TP': tp,
            'precision': prec, 'recall': rec, 'F1': f1,
            'identified_variants': ', '.join(hit_vars),
            'heldout_lineage': str(heldout_lineage),
            'train_n': int(train_mask.sum()), 'train_r': train_r, 'train_s': train_s,
            'test_n': int(test_mask.sum()), 'test_r': test_r, 'test_s': test_s,
        })

    pr_df = pd.DataFrame(rows)
    pr_df.to_csv(out_dir / f'PR_{drug}_heldout_lineage_{heldout_lineage}.csv', index=False)
    print(f'[ok] {drug} held-out lineage {heldout_lineage}: wrote {out_dir / f"PR_{drug}_heldout_lineage_{heldout_lineage}.csv"}')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', choices=ELIGIBLE_DRUGS, default=None)
    parser.add_argument('--heldout-lineage', default=None, choices=list(MAJOR_LINEAGES))
    parser.add_argument('--k-vals', default='1,5,10')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    drugs = [args.drug] if args.drug else ELIGIBLE_DRUGS
    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)
    k_vals = tuple(int(k) for k in args.k_vals.split(','))

    for drug in drugs:
        for heldout in heldouts:
            try:
                run_lineage_task2_shap_for_drug(drug, heldout, k_vals=k_vals, seed=args.seed)
            except Exception as e:
                print(f'[error] {drug} held-out lineage {heldout}: {e!r}; continuing')

    pr_files = sorted(OUT_ROOT.glob('*/heldout_lineage_*/PR_*.csv'))
    if pr_files:
        combined = pd.concat([pd.read_csv(f) for f in pr_files], ignore_index=True)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUT_ROOT / 'combined_lineage_task2_shap_precision_recall.csv', index=False)
        print(f'[done] wrote {OUT_ROOT / "combined_lineage_task2_shap_precision_recall.csv"} ({len(pr_files)} combos)')


if __name__ == '__main__':
    main()
