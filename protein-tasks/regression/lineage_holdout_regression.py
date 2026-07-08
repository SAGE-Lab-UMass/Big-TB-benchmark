"""Run lineage-aware holdout evaluation for regression baselines.

Regression uses saved feature matrices rather than sequence-table datasets, so
this runner consumes recovered row manifests to map matrix rows back to isolate
IDs and lineage calls. The split rule matches the one-hot and ESM runners: test
is one major lineage, train is every other lineage-annotated isolate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROTEIN_TASKS_DIR))

from lineage_split_utils import DEFAULT_MIN_CLASS_COUNT, MAJOR_LINEAGES
from regression_utils import DRUG2GENES, encode_labels, load_feature_matrix_and_labels

# train contains all non-held-out lineaged isolates.
OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/regression'
MANIFEST_DIR = PROTEIN_TASKS_DIR / 'data/latest/feature_matrix_labels/row_manifests'
SUPPORTED_DRUGS = set(DRUG2GENES)


def _write_aggregate_summary(drug: str) -> None:
    """Rebuild the per-drug aggregate table from split-level summaries."""
    out_dir = OUT_ROOT / drug
    summary_files = sorted(out_dir.glob('heldout_lineage_*/summary.csv'))
    if not summary_files:
        return
    df = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True)
    if 'heldout_lineage' in df.columns:
        sort_cols = ['heldout_lineage']
        if 'model' in df.columns:
            sort_cols.append('model')
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(out_dir / 'all_lineage_summary.csv', index=False)


def _auc_safe(y_true, y_score):
    """Return NaN instead of crashing if a test split has one class."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def _bootstrap_auc_ci(y, p, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        boots.append(roc_auc_score(yy, pp))
    if not boots:
        return np.nan, (np.nan, np.nan)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(boots)), (float(lo), float(hi))


def _extract_raw_coefs(pipeline):
    """Map standardized model coefficients back to the raw feature scale."""
    scaler = pipeline.named_steps['scaler']
    est = pipeline.named_steps['est']
    coef_std = est.coef_.ravel() if hasattr(est, 'coef_') else est.coef_
    scale = np.where(scaler.scale_ != 0, scaler.scale_, 1.0)
    return coef_std / scale


def _load_manifest(drug: str) -> pd.DataFrame:
    """Load the recovered matrix-row to isolate-row manifest for one drug."""
    manifest_path = MANIFEST_DIR / f'{drug.upper()}_row_manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing row manifest for {drug}: {manifest_path}')
    df = pd.read_csv(manifest_path)
    df['Filename'] = df['Filename'].astype(str)
    df['saved_label'] = df['saved_label'].astype(str)
    df['Lineage'] = df['Lineage'].astype('string')
    return df


def _prepare_lineage_annotated_subset(drug: str):
    """Align matrix rows, labels, and lineage calls for split construction."""
    X_full, _ = load_feature_matrix_and_labels(drug)
    manifest = _load_manifest(drug)
    # Drop only rows without a lineage call. Minor or ambiguous lineage labels
    # remain eligible for training when they are not the held-out test lineage.
    annotated = manifest[manifest['Lineage'].notna()].copy()
    annotated = annotated.sort_values('matrix_row_idx').reset_index(drop=True)
    y = encode_labels(annotated['saved_label'].to_numpy())
    X = X_full[annotated['matrix_row_idx'].to_numpy()]
    if X.shape[0] != len(annotated):
        raise ValueError(f'{drug}: X rows {X.shape[0]} != manifest rows {len(annotated)}')
    return X, y, annotated


def _build_models(seed: int = 42):
    """Build the same regression baseline family used in the protein benchmark."""
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


def run_lineage_holdout_for_drug(drug: str, heldout_lineage: str, min_class_count: int = DEFAULT_MIN_CLASS_COUNT, seed: int = 42, dry_run: bool = False):
    if drug not in SUPPORTED_DRUGS:
        raise NotImplementedError(f'{drug} is not supported by the regression lineage runner')

    X, y, manifest = _prepare_lineage_annotated_subset(drug)
    # Held-out groups are exact top-level major-lineage labels. Everything
    # else with a lineage annotation, including minor/ambiguous labels, trains.
    test_mask = manifest['Lineage'].astype(str) == str(heldout_lineage)
    train_mask = ~test_mask

    train_labels = manifest.loc[train_mask, 'saved_label']
    test_labels = manifest.loc[test_mask, 'saved_label']
    train_r = int((train_labels == 'R').sum())
    train_s = int((train_labels == 'S').sum())
    test_r = int((test_labels == 'R').sum())
    test_s = int((test_labels == 'S').sum())
    # A reportable split needs both classes in both partitions: the model must
    # have enough signal to fit and the held-out AUC must be interpretable.
    feasible = min(train_r, train_s, test_r, test_s) >= min_class_count

    result = {
        'drug': drug,
        'heldout_lineage': str(heldout_lineage),
        'train_n': int(train_mask.sum()),
        'train_r': train_r,
        'train_s': train_s,
        'test_n': int(test_mask.sum()),
        'test_r': test_r,
        'test_s': test_s,
        'feasible': feasible,
        'ambiguous_rows_total': int(manifest['assignment_ambiguous'].sum()) if 'assignment_ambiguous' in manifest.columns else 0,
        'cross_lineage_ambiguous_rows_total': int(manifest['cross_lineage_duplicate_bucket'].sum()) if 'cross_lineage_duplicate_bucket' in manifest.columns else 0,
        'ambiguous_rows_test': int(manifest.loc[test_mask, 'assignment_ambiguous'].sum()) if 'assignment_ambiguous' in manifest.columns else 0,
        'cross_lineage_ambiguous_rows_test': int(manifest.loc[test_mask, 'cross_lineage_duplicate_bucket'].sum()) if 'cross_lineage_duplicate_bucket' in manifest.columns else 0,
    }
    if not feasible:
        print(f'[skip] {drug} held-out lineage {heldout_lineage}: underpowered')
        return [result]

    if dry_run:
        print(f'[dry-run] {drug} held-out lineage {heldout_lineage}: {result}')
        return [result]

    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    out_dir = OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.loc[test_mask].to_csv(out_dir / 'test_manifest.csv', index=False)

    rows = []
    for name, pipe in _build_models(seed).items():
        pipe.fit(X_tr, y_tr)
        if name == 'logreg':
            y_score = pipe.predict_proba(X_te)[:, 1]
        else:
            y_score = pipe.predict(X_te)
        auc = _auc_safe(y_te, y_score)
        acc = accuracy_score(y_te, (y_score >= 0.5).astype(int)) if not np.isnan(auc) else np.nan
        pooled_mean, (ci_lo, ci_hi) = _bootstrap_auc_ci(y_te, y_score, seed=seed)

        # Store isolate-level predictions so later tables can be audited back
        # to the exact held-out lineage examples.
        preds = pd.DataFrame({
            'Filename': manifest.loc[test_mask, 'Filename'].tolist(),
            'Lineage': manifest.loc[test_mask, 'Lineage'].astype(str).tolist(),
            'prob': y_score,
            'label': y_te,
        })
        preds.to_csv(out_dir / f'{name}_test_preds.csv', index=False)

        coef_raw = _extract_raw_coefs(pipe)
        np.save(out_dir / f'{name}_coefs.npy', coef_raw)

        row = dict(result)
        row.update({
            'model': name,
            'auc': auc,
            'acc': acc,
            'pooled_auc': pooled_mean,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
        })
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / 'summary.csv', index=False)
    print(f'[ok] {drug} held-out lineage {heldout_lineage}: wrote {out_dir / "summary.csv"}')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', required=True, choices=sorted(SUPPORTED_DRUGS))
    parser.add_argument('--heldout-lineage', default=None, choices=['1', '2', '3', '4'])
    parser.add_argument('--min-class-count', type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)
    for heldout in heldouts:
        run_lineage_holdout_for_drug(
            args.drug,
            heldout,
            min_class_count=args.min_class_count,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        out_dir = OUT_ROOT / args.drug
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_aggregate_summary(args.drug)


if __name__ == '__main__':
    main()
