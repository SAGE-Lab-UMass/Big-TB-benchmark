"""Aggregate lineage-holdout result tables into manuscript-facing summaries.

This script rebuilds per-drug `all_lineage_summary.csv` files from the split-
level `summary.csv` outputs written by each runner, then creates combined tables
that are easier to inspect and commit than the full artifact tree.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROTEIN_TASKS_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train'
COMBINED_DIR = RESULTS_ROOT / 'combined'

MODEL_SPECS = {
    'cnn': {
        'summary_glob': 'cnn/*/heldout_lineage_*/summary.csv',
        'aggregate_root_glob': 'cnn/*',
        'family': 'cnn',
        'model_label': 'cnn',
    },
    'transformer': {
        'summary_glob': 'transformer/*/heldout_lineage_*/summary.csv',
        'aggregate_root_glob': 'transformer/*',
        'family': 'transformer',
        'model_label': 'transformer',
    },
    'esm': {
        'summary_glob': 'esm/*/*/heldout_lineage_*/summary.csv',
        'aggregate_root_glob': 'esm/*/*',
        'family': 'esm',
        'model_label': 'esm',
    },
    'regression': {
        'summary_glob': 'regression/*/heldout_lineage_*/summary.csv',
        'aggregate_root_glob': 'regression/*',
        'family': 'regression',
        'model_label': None,
    },
}


def _sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ['heldout_lineage', 'model'] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def rebuild_per_drug_aggregates() -> None:
    """Rewrite every all_lineage_summary.csv from split-level summaries."""
    for spec in MODEL_SPECS.values():
        for root in sorted(RESULTS_ROOT.glob(spec['aggregate_root_glob'])):
            if not root.is_dir():
                continue
            summary_files = sorted(root.glob('heldout_lineage_*/summary.csv'))
            if not summary_files:
                continue
            df = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True)
            df = _sort_dataframe(df)
            df.to_csv(root / 'all_lineage_summary.csv', index=False)


def build_long_table() -> pd.DataFrame:
    """Collect all reportable split summaries into one long-format table."""
    frames = []
    for spec in MODEL_SPECS.values():
        for path in sorted(RESULTS_ROOT.glob(spec['summary_glob'])):
            df = pd.read_csv(path)
            df['model_family'] = spec['family']
            if spec['model_label'] is not None and 'model' not in df.columns:
                df['model'] = spec['model_label']
            if spec['family'] == 'esm':
                parts = path.parts
                mode_dir = parts[-3]
                if '_' in mode_dir:
                    mode, in_dim = mode_dir.rsplit('_', 1)
                    df['mode'] = mode
                    df['in_dim'] = int(in_dim)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    preferred_cols = [
        'drug', 'model_family', 'model', 'mode', 'in_dim', 'heldout_lineage',
        'train_n', 'train_r', 'train_s', 'test_n', 'test_r', 'test_s',
        'feasible', 'auc', 'acc', 'sens', 'spec', 'pooled_boot_auc_mean',
        'pooled_auc', 'ci_lo', 'ci_hi', 'last_val_auc', 'last_val_acc',
        'ambiguous_rows_total', 'cross_lineage_ambiguous_rows_total',
        'ambiguous_rows_test', 'cross_lineage_ambiguous_rows_test',
    ]
    ordered = [c for c in preferred_cols if c in long_df.columns]
    leftovers = [c for c in long_df.columns if c not in ordered]
    long_df = long_df[ordered + leftovers]
    return long_df.sort_values(['drug', 'model_family', 'model', 'heldout_lineage']).reset_index(drop=True)


def build_mean_auc_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Create a wide per-drug mean AUC summary for manuscript drafting."""
    if long_df.empty:
        return long_df
    mean_df = (
        long_df.groupby(['drug', 'model_family', 'model'], dropna=False)['auc']
        .mean()
        .reset_index(name='mean_auc')
    )

    rows = []
    for drug, group in mean_df.groupby('drug'):
        row = {'drug': drug}
        cnn = group[(group['model_family'] == 'cnn')]['mean_auc']
        trf = group[(group['model_family'] == 'transformer')]['mean_auc']
        esm = group[(group['model_family'] == 'esm')]['mean_auc']
        reg = group[group['model_family'] == 'regression']
        row['cnn_mean_auc'] = float(cnn.iloc[0]) if not cnn.empty else pd.NA
        row['transformer_mean_auc'] = float(trf.iloc[0]) if not trf.empty else pd.NA
        row['esm_mean_auc'] = float(esm.iloc[0]) if not esm.empty else pd.NA
        for model_name in ['logreg', 'ridge', 'lasso']:
            sub = reg[reg['model'] == model_name]['mean_auc']
            row[f'regression_{model_name}_mean_auc'] = float(sub.iloc[0]) if not sub.empty else pd.NA
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values('drug').reset_index(drop=True)


def main() -> None:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    rebuild_per_drug_aggregates()
    long_df = build_long_table()
    if long_df.empty:
        print('[warn] No lineage summary files found.')
        return
    long_path = COMBINED_DIR / 'lineage_holdout_per_split_results.csv'
    long_df.to_csv(long_path, index=False)

    mean_auc_df = build_mean_auc_table(long_df)
    mean_path = COMBINED_DIR / 'lineage_holdout_mean_auc_by_drug.csv'
    mean_auc_df.to_csv(mean_path, index=False)
    print(f'[ok] wrote {long_path}')
    print(f'[ok] wrote {mean_path}')


if __name__ == '__main__':
    main()
