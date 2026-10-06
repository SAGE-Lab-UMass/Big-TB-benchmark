"""DO NOT COMMIT / SUPERSEDED - kept locally for reference only.

This script ranks residues by raw Lasso/Ridge/LogisticRegression coefficient
magnitude on a StandardScaler-ed feature matrix. That is NOT the pipeline
behind the paper's actual Table 3 "LogReg (Ref-Alt)" row, which uses a single
un-scaled LogisticRegressionCV ranked by SHAP (see
lineage_holdout_task2_shap_regression.py / controlled_shap_task2_shap_regression.py,
which do match Table 3 exactly - verified digit-for-digit). This file's
numbers are internally self-consistent but do not correspond to anything
published, so they should not be committed, reported, or confused with the
real regression Task 2 results.

Original docstring, for context: reuses the same train/test split and model
family as lineage_holdout_regression.py (leave-one-major-lineage-out), and
the same Precision@k/Recall@k methodology as
regression_utils.evaluate_topk_precision_recall (bona-fide positions = WHO
confidence 1-2 AND intersectional==True). No feasibility threshold is
applied; every (drug, held-out lineage, model, k) combination is computed,
including folds with zero or near-zero test-side carrier support.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROTEIN_TASKS_DIR))
sys.path.insert(0, str(THIS_DIR))

from lineage_split_utils import DEFAULT_MIN_CLASS_COUNT, MAJOR_LINEAGES
from regression_utils import (
    DRUG2GENES,
    compute_residue_scores,
    encode_labels,
    evaluate_topk_precision_recall,
    gene_slices,
    load_catalog,
    load_feature_matrix_and_labels,
)
from lineage_holdout_regression import _load_manifest, _prepare_lineage_annotated_subset

OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/regression_task2'
WHO_CATALOG = PROTEIN_TASKS_DIR / 'data/filtered_variants_output.csv'
ALLOWED_CONF = ['1) Assoc w R', '2) Assoc w R - Interim']

ELIGIBLE_DRUGS = ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'streptomycin',
                  'capreomycin', 'moxifloxacin', 'ethionamide']


def _extract_raw_coefs(pipeline):
    scaler = pipeline.named_steps['scaler']
    est = pipeline.named_steps['est']
    coef_std = est.coef_.ravel() if hasattr(est, 'coef_') else est.coef_
    scale = np.where(scaler.scale_ != 0, scaler.scale_, 1.0)
    return coef_std / scale


def _build_models(seed: int = 42):
    return {
        'lasso': Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('est', LassoCV(max_iter=10000, cv=5, random_state=seed, alphas=[0.001, 0.01, 0.1, 1, 10, 100])),
        ]),
        'ridge': Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('est', RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10])),
        ]),
        'logreg': Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('est', LogisticRegressionCV(
                cv=5, scoring='roc_auc', max_iter=5000,
                Cs=[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                class_weight='balanced', solver='liblinear', refit=True,
            )),
        ]),
    }


def run_lineage_task2_for_drug(drug: str, heldout_lineage: str, k_vals=(1, 5, 10), seed: int = 42):
    if drug not in DRUG2GENES:
        raise NotImplementedError(f'{drug} is not supported by the regression lineage runner')

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
    X_tr = X[train_idx]
    y_tr = y[train_idx]

    slices = gene_slices(drug, X.shape[1])
    who_df = load_catalog(WHO_CATALOG, ALLOWED_CONF)

    out_dir = OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_rows = []
    for name, pipe in _build_models(seed).items():
        try:
            pipe.fit(X_tr, y_tr)
            coef_raw = _extract_raw_coefs(pipe)
        except np.linalg.LinAlgError as e:
            # RidgeCV's SVD-based solver occasionally fails to converge on
            # small/skewed lineage-holdout training folds. Skip this model
            # for this fold rather than losing every other (drug, lineage)
            # combination in the grid.
            print(f'[warn] {drug} held-out lineage {heldout_lineage}: {name} fit failed ({e}); skipping')
            continue
        np.save(out_dir / f'{name}_coefs.npy', coef_raw)

        for gene, (start, end) in slices.items():
            scores = compute_residue_scores(coef_raw[start:end])
            full_scores = pd.DataFrame({
                'Residue_Position': np.arange(start, end) - start,
                'Importance': scores,
                'Model': name,
                'Gene': gene,
            })
            full_scores.to_csv(out_dir / f'full_residue_scores_{gene}_{drug}_{name}.csv', index=False)

            rows = evaluate_topk_precision_recall(drug, gene, scores, who_df, k_vals=k_vals, model=name)
            for row in rows:
                row.update({
                    'heldout_lineage': str(heldout_lineage),
                    'train_n': int(train_mask.sum()), 'train_r': train_r, 'train_s': train_s,
                    'test_n': int(test_mask.sum()), 'test_r': test_r, 'test_s': test_s,
                })
            pr_rows.extend(rows)

    pr_df = pd.DataFrame(pr_rows)
    pr_df.to_csv(out_dir / f'PR_{drug}_heldout_lineage_{heldout_lineage}.csv', index=False)
    print(f'[ok] {drug} held-out lineage {heldout_lineage}: wrote {out_dir / f"PR_{drug}_heldout_lineage_{heldout_lineage}.csv"}')
    return pr_rows


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
                run_lineage_task2_for_drug(drug, heldout, k_vals=k_vals, seed=args.seed)
            except Exception as e:
                # Never let one (drug, lineage) combination's failure abort
                # the rest of the grid.
                print(f'[error] {drug} held-out lineage {heldout}: {e!r}; continuing')

    # Rebuild the combined table from whatever PR_*.csv files exist on disk,
    # so it reflects every combination that succeeded even if this run only
    # covered a subset (e.g. a rerun after a partial failure).
    pr_files = sorted(OUT_ROOT.glob('*/heldout_lineage_*/PR_*.csv'))
    if pr_files:
        combined = pd.concat([pd.read_csv(f) for f in pr_files], ignore_index=True)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUT_ROOT / 'combined_lineage_task2_precision_recall.csv', index=False)
        print(f'[done] wrote {OUT_ROOT / "combined_lineage_task2_precision_recall.csv"} ({len(pr_files)} combos)')


if __name__ == '__main__':
    main()
