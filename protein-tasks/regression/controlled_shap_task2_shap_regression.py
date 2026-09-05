"""Controlled SHAP comparison for the Ref-Alt/SHAP logistic-regression Task 2
lineage holdout - mirrors one_hot_encoded/controlled_shap_task2_cnn.py's
design exactly. Background = lineage model's own train partition minus overlap
with the fixed explain set; explain set = Table 3's exact original isolate
identities (recovered deterministically via original_explain_set.py).

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
from original_explain_set import recover_original_explain_filenames

OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/regression_task2_shap_controlled'
WHO_CATALOG = PROTEIN_TASKS_DIR / 'data/filtered_variants_output.csv'
ALLOWED_CONF = ['1) Assoc w R', '2) Assoc w R - Interim']
MODEL_NAME = 'logreg_shap_independent'
ELIGIBLE_DRUGS = ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'streptomycin']
BG_FRAC = 0.10
MAX_BG = 160
SEED = 42


def run_controlled_for_drug(drug: str, heldout_lineage: str, k_vals=(1, 5, 10), seed: int = SEED):
    X, y, manifest = _prepare_lineage_annotated_subset(drug)
    test_mask = manifest['Lineage'].astype(str) == str(heldout_lineage)
    train_mask = ~test_mask
    train_idx = np.flatnonzero(train_mask.to_numpy())

    explain_filenames = set(recover_original_explain_filenames(drug))
    filenames = manifest['Filename'].astype(str).to_numpy()
    fid_to_idx = {fid: i for i, fid in enumerate(filenames)}

    explain_local_idx = np.array([fid_to_idx[f] for f in explain_filenames if f in fid_to_idx])
    missing = len(explain_filenames) - len(explain_local_idx)
    if missing:
        print(f'[warn] {drug} held-out lineage {heldout_lineage}: {missing}/{len(explain_filenames)} '
              f'original explain-set isolates not found in lineage-annotated cohort')

    train_fid_set = set(filenames[train_idx])
    bg_candidate_fids = train_fid_set - explain_filenames
    bg_pool_idx = np.array([fid_to_idx[f] for f in bg_candidate_fids])

    X_tr_full, y_tr_full = X[train_idx], y[train_idx]
    model = LogisticRegressionCV(
        cv=3, scoring='roc_auc', max_iter=5000,
        Cs=[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
        class_weight='balanced',
    )
    model.fit(X_tr_full, y_tr_full)

    y_bg_pool = y[bg_pool_idx]
    bg_sub_idx, _ = stratified_bg_explain_indices(y_bg_pool, bg_frac=BG_FRAC, seed=seed, max_bg=MAX_BG)
    X_bg = X[bg_pool_idx[bg_sub_idx]]
    X_explain = X[explain_local_idx]

    slices = gene_slices(drug, X.shape[1])
    who_df = load_catalog(WHO_CATALOG, ALLOWED_CONF)
    catalog_full = pd.read_csv(WHO_CATALOG)
    catalog_full['aa_pos_0idx'] = catalog_full['aa_pos'].astype(int) - 1

    phi = linear_shap_explain(model, X_bg, X_explain, masker_mode='independent')
    scores_global = np.abs(phi).max(axis=0)
    rank_df = (pd.DataFrame({
        'Residue_Position': np.arange(scores_global.shape[0]),
        'MaxAbsSHAP': scores_global,
    }).sort_values('MaxAbsSHAP', ascending=False).reset_index(drop=True))
    gold_global, excl_global = druglevel_gold_excl(slices, who_df, catalog_full)
    n_true = len(gold_global)

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
            'shap_explain_scope': 'table3_matched',
            'explain_n': len(explain_local_idx), 'background_pool_n': len(X_bg),
        })

    out_dir = OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f'PR_{drug}_heldout_lineage_{heldout_lineage}_controlled.csv', index=False)
    print(f'[ok] {drug} held-out lineage {heldout_lineage}: controlled-SHAP PR written '
          f'(explain_n={len(explain_local_idx)}, bg_n={len(X_bg)})')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', choices=ELIGIBLE_DRUGS, default=None)
    parser.add_argument('--heldout-lineage', default=None, choices=list(MAJOR_LINEAGES))
    parser.add_argument('--k-vals', default='1,5,10')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    drugs = [args.drug] if args.drug else ELIGIBLE_DRUGS
    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)
    k_vals = tuple(int(k) for k in args.k_vals.split(','))

    all_pr = []
    for drug in drugs:
        for heldout in heldouts:
            try:
                rows = run_controlled_for_drug(drug, heldout, k_vals=k_vals, seed=args.seed)
                all_pr.extend(rows)
            except Exception as e:
                print(f'[error] {drug} held-out lineage {heldout}: {e!r}; continuing')

    if all_pr:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_pr).to_csv(OUT_ROOT / 'combined_controlled_shap_regression.csv', index=False)
        print(f'[done] wrote {OUT_ROOT / "combined_controlled_shap_regression.csv"}')


if __name__ == '__main__':
    main()
