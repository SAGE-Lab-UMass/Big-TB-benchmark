"""Run leave-one-major-lineage-out evaluation for the one-hot transformer.

This script reuses the standard training code but replaces random folds with
canonical isolate-level lineage splits from `lineage_split_utils`. The held-out
lineage is used only as the final test set; no validation or augmentation rows
are drawn from it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROTEIN_TASKS_DIR))

from lineage_split_utils import build_and_save_drug_splits, DEFAULT_MIN_CLASS_COUNT
from transformer import ProteinTransformer
from transformer_utils import ProteinDataset, bootstrap_auc_ci, set_seed, build_drug_df
from significance_testing_transformer import _train_one_fold, DRUG2GENES, N_EPOCHS, LR, BATCH_SIZE, DEVICE, SEED

#train contains all non-held-out lineaged isolates.
OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/transformer'


def _write_aggregate_summary(drug: str) -> None:
    """Rebuild the per-drug aggregate table from split-level summaries."""
    out_dir = OUT_ROOT / drug
    summary_files = sorted(out_dir.glob('heldout_lineage_*/summary.csv'))
    if not summary_files:
        return
    df = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True)
    if 'heldout_lineage' in df.columns:
        df = df.sort_values('heldout_lineage').reset_index(drop=True)
    df.to_csv(out_dir / 'all_lineage_summary.csv', index=False)


def run_lineage_holdout_for_drug(drug: str, heldout_lineage: str, min_class_count: int = DEFAULT_MIN_CLASS_COUNT, dry_run: bool = False) -> dict:
    # Recompute the canonical isolate-ID split from shared inputs. Because the
    # split is defined by isolate IDs, each model family receives the same train/test
    # isolates for a given drug and held-out lineage.
    df, splits = build_and_save_drug_splits(drug, DRUG2GENES, min_class_count=min_class_count)
    split = splits[str(heldout_lineage)]
    result = {
        'drug': drug,
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

    if dry_run:
        print(f"[dry-run] {drug} held-out lineage {heldout_lineage}: {result}")
        return result

    # Build the full model dataset after split construction, then map isolate
    # IDs to dataset indices. This keeps splitting independent of row order.
    full_ds = ProteinDataset(
        df['Protein_Sequence'].tolist(),
        (df['Phenotype'] == 'R').astype(int).tolist(),
    )
    id_to_idx = {fid: idx for idx, fid in enumerate(df['Filename'].astype(str).tolist())}
    train_idx = [id_to_idx[fid] for fid in split['train_ids'] if fid in id_to_idx]
    test_idx = [id_to_idx[fid] for fid in split['test_ids'] if fid in id_to_idx]
    train_ds = Subset(full_ds, train_idx)
    test_ds = Subset(full_ds, test_idx)

    model = ProteinTransformer().to(DEVICE)
    curve_df, probs, gold, metrics = _train_one_fold(
        model, train_ds, test_ds,
        n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, device=DEVICE,
    )

    out_dir = OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(out_dir / 'training_curve.csv', index=False)
    pd.DataFrame({'prob': probs, 'label': gold}).to_csv(out_dir / 'test_preds.csv', index=False)
    torch.save(model.state_dict(), out_dir / f'{drug}_transformer.pt')

    boot_mean, (ci_lo, ci_hi) = bootstrap_auc_ci(np.asarray(gold), np.asarray(probs), n_boot=5000, alpha=0.05, seed=SEED)
    result.update({
        'auc': metrics['auc'],
        'acc': metrics['acc'],
        'sens': metrics['sens'],
        'spec': metrics['spec'],
        'pooled_boot_auc_mean': boot_mean,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
    })
    pd.DataFrame([result]).to_csv(out_dir / 'summary.csv', index=False)
    print(f"[ok] {drug} held-out lineage {heldout_lineage}: auc={metrics['auc']:.3f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', required=True, choices=sorted(DRUG2GENES))
    parser.add_argument('--heldout-lineage', default=None, choices=['1', '2', '3', '4'])
    parser.add_argument('--min-class-count', type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    set_seed(SEED)
    _, splits = build_and_save_drug_splits(args.drug, DRUG2GENES, min_class_count=args.min_class_count)
    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(splits.keys())
    for heldout in heldouts:
        run_lineage_holdout_for_drug(args.drug, heldout, min_class_count=args.min_class_count, dry_run=args.dry_run)

    if not args.dry_run:
        out_dir = OUT_ROOT / args.drug
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_aggregate_summary(args.drug)


if __name__ == '__main__':
    main()
