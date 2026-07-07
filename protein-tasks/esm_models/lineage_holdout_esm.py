"""Run leave-one-major-lineage-out evaluation for ESM embedding CNN models.

The saved ESM datasets are indexed by isolate IDs. This runner builds the same
canonical lineage split manifests as the one-hot models, then maps split IDs to
embedding-dataset indices so the foundation-model comparison uses identical
held-out isolates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROTEIN_TASKS_DIR))

from lineage_split_utils import build_and_save_drug_splits, DEFAULT_MIN_CLASS_COUNT
from significance_testing import (
    DRUG2GENES,
    load_dataset_for_cv,
    train_token_split,
    _eval_subset,
)

#train contains all non-held-out lineaged isolates.
OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/esm'
SEED = 42
MODE2DIM = {
    'full': 320,
    'pca': 10,
    'mean': 1,
}


def _write_aggregate_summary(drug: str, mode: str, in_dim: int) -> None:
    """Rebuild the per-drug aggregate table from split-level summaries."""
    out_dir = OUT_ROOT / drug / f'{mode}_{in_dim}'
    summary_files = sorted(out_dir.glob('heldout_lineage_*/summary.csv'))
    if not summary_files:
        return
    df = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True)
    if 'heldout_lineage' in df.columns:
        df = df.sort_values('heldout_lineage').reset_index(drop=True)
    df.to_csv(out_dir / 'all_lineage_summary.csv', index=False)


def run_lineage_holdout_for_drug(
    drug: str,
    heldout_lineage: str,
    mode: str,
    in_dim: int,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
    batch_size: int = 32,
    n_epochs: int = 20,
    lr: float = 5e-4,
    freeze_bias_frac: float = 0.25,
    dry_run: bool = False,
) -> dict:
    # Build the isolate-level split before loading embeddings; this prevents
    # embedding shard order from defining or perturbing train/test membership.
    df, splits = build_and_save_drug_splits(drug, DRUG2GENES, min_class_count=min_class_count)
    split = splits[str(heldout_lineage)]
    result = {
        'drug': drug,
        'mode': mode,
        'in_dim': in_dim,
        'heldout_lineage': str(heldout_lineage),
        'train_n': split['train_counts']['n'],
        'train_r': split['train_counts']['r'],
        'train_s': split['train_counts']['s'],
        'test_n': split['test_counts']['n'],
        'test_r': split['test_counts']['r'],
        'test_s': split['test_counts']['s'],
        'feasible': split['feasible'],
    }
    if not split['feasible']:
        print(f"[skip] {drug} held-out lineage {heldout_lineage}: underpowered")
        return result

    full_ds, label_map, gene_names, per_gene_len = load_dataset_for_cv(None, drug, mode, in_dim)
    # ESM datasets expose stable isolate IDs. Mapping by ID rather than row
    # number keeps the ESM split aligned with one-hot and regression models.
    dataset_ids = list(getattr(full_ds, 'ids', []))
    id_to_idx = {fid: idx for idx, fid in enumerate(dataset_ids)}
    train_idx = [id_to_idx[fid] for fid in split['train_ids'] if fid in id_to_idx]
    test_idx = [id_to_idx[fid] for fid in split['test_ids'] if fid in id_to_idx]
    result.update({
        'train_n_dataset': len(train_idx),
        'test_n_dataset': len(test_idx),
    })
    if min(len(train_idx), len(test_idx)) == 0:
        print(f"[skip] {drug} held-out lineage {heldout_lineage}: no overlap with embedding dataset")
        return result

    if dry_run:
        print(f"[dry-run] {drug} {mode} held-out lineage {heldout_lineage}: {result}")
        return result

    # The held-out lineage is used directly as the evaluation set for this
    # robustness analysis; no examples from it enter model fitting.
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)
    out_dir = OUT_ROOT / drug / f'{mode}_{in_dim}' / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, _, hist = train_token_split(
        gene=None,
        drug=drug,
        mode=mode,
        in_dim=in_dim,
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=lr,
        freeze_bias_frac=freeze_bias_frac,
        out_root=str(out_dir),
        train_ds=train_ds,
        val_ds=test_ds,
        per_gene_len=per_gene_len,
        gene_names=gene_names,
        compute_shap=False,
    )

    probe_n = min(100, len(train_ds))
    l_pad = max(train_ds[i][0].shape[1] for i in range(probe_n))
    probs, gold = _eval_subset(model, test_ds, batch_size, 'cuda' if torch.cuda.is_available() else 'cpu', l_pad)
    pd.DataFrame({'prob': probs, 'label': gold}).to_csv(out_dir / 'test_preds.csv', index=False)

    auc = float('nan')
    if len(np.unique(gold)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(gold, probs))
    result.update({
        'auc': auc,
        'last_val_auc': float(hist['val_auc'].iloc[-1]),
        'last_val_acc': float(hist['val_acc'].iloc[-1]),
    })
    pd.DataFrame([result]).to_csv(out_dir / 'summary.csv', index=False)
    print(f"[ok] {drug} {mode} held-out lineage {heldout_lineage}: auc={auc:.3f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', required=True, choices=sorted(DRUG2GENES))
    parser.add_argument('--mode', default='full', choices=['full', 'pca', 'mean'])
    parser.add_argument('--in-dim', type=int, default=None)
    parser.add_argument('--heldout-lineage', default=None, choices=['1', '2', '3', '4'])
    parser.add_argument('--min-class-count', type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    in_dim = args.in_dim if args.in_dim is not None else MODE2DIM[args.mode]
    _, splits = build_and_save_drug_splits(args.drug, DRUG2GENES, min_class_count=args.min_class_count)
    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(splits.keys())
    for heldout in heldouts:
        run_lineage_holdout_for_drug(
            args.drug,
            heldout,
            args.mode,
            in_dim,
            min_class_count=args.min_class_count,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        out_dir = OUT_ROOT / args.drug / f'{args.mode}_{in_dim}'
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_aggregate_summary(args.drug, args.mode, in_dim)


if __name__ == '__main__':
    main()
