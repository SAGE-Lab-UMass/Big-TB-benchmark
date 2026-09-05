"""Controlled SHAP comparison for ESM Task 2 lineage holdout - mirrors
one_hot_encoded/controlled_shap_task2_cnn.py's design exactly (agreed
2026-09-02, see that file's docstring for the full rationale). Background =
lineage model's own train partition minus overlap with the fixed explain
set; explain set = the exact isolate identities from the (verified-correct)
Table 3 SHAP methodology, recovered via original_explain_set.py - NOT a
resample, and NOT the stale on-disk `{drug}_dim320_shap_all_meta.json` pool,
which was found to have a different (likely outdated) R/S composition than
what the current, live compute_shap_for_drug() code actually produces.

Verified: recover_original_explain_filenames('rifampicin') reproduces the
identical 712-isolate/472R-240S pool that CNN's dedup produces - confirming
the unified background/explain design documented in the paper is correctly
implemented in the current codebase, even though it doesn't match whatever
generated the currently-published Table 3 ESM numbers (a separate, open
issue - see original_explain_set.py's docstring).

Retrains from scratch (the original ESM lineage run never saved a
checkpoint) and, in the same run, computes SHAP with the corrected
background/explain design.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROTEIN_TASKS_DIR))
sys.path.insert(0, str(THIS_DIR))

from lineage_split_utils import build_and_save_drug_splits, DEFAULT_MIN_CLASS_COUNT, MAJOR_LINEAGES
from significance_testing import DRUG2GENES, load_dataset_for_cv, train_token_split
from shap_esm import Wrapped
from interp_pr_utils import load_catalog_normalized, precision_recall_from_shap
from original_explain_set import recover_original_explain_filenames

OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/esm_task2_controlled_shap'
WHO_CATALOG = PROTEIN_TASKS_DIR / 'data/filtered_variants_output.csv'
ELIGIBLE_DRUGS = ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'streptomycin']
MODE2DIM = {'full': 320, 'pca': 10, 'mean': 1}
SEED = 42
BG_SIZE = 100


def _shap_explain(model, background_ds, explain_ds, l_pad, background_size,
                   per_gene_lengths, gene_names, device, seed=SEED):
    model = model.to(device).eval()
    rng = random.Random(seed)

    def _padded(ds, idx):
        x, y = ds[idx]
        if x.shape[1] < l_pad:
            x = torch.nn.functional.pad(x, (0, l_pad - x.shape[1]))
        return x, y

    b_n = min(background_size, len(background_ds))
    bg_idx = rng.sample(range(len(background_ds)), b_n)
    background = torch.stack([_padded(background_ds, i)[0] for i in bg_idx]).to(device)

    xs = torch.stack([_padded(explain_ds, i)[0] for i in range(len(explain_ds))]).to(device)
    ys = [int(_padded(explain_ds, i)[1]) for i in range(len(explain_ds))]

    explainer = shap.DeepExplainer(Wrapped(model), [background])
    sv = explainer.shap_values([xs], check_additivity=False)[0]
    imp = np.abs(sv).sum(axis=1)

    out = {'sample_idx': list(range(len(explain_ds))), 'label': ys, 'importance_full': list(imp)}
    if per_gene_lengths is not None:
        cuts = np.cumsum([0] + per_gene_lengths)
        for gi, g in enumerate(gene_names):
            out[f'importance_{g}'] = [imp[n, cuts[gi]:cuts[gi + 1]] for n in range(imp.shape[0])]
    return pd.DataFrame(out)


def run_controlled_shap_for_drug(drug: str, heldout_lineage: str, mode: str, in_dim: int, k_vals=(1, 5, 10),
                                  min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
                                  batch_size: int = 32, n_epochs: int = 20, lr: float = 5e-4,
                                  freeze_bias_frac: float = 0.25):
    df, splits = build_and_save_drug_splits(drug, DRUG2GENES, min_class_count=min_class_count)
    split = splits[str(heldout_lineage)]
    if not split['feasible']:
        print(f'[skip] {drug} held-out lineage {heldout_lineage}: underpowered')
        return None

    full_ds, label_map, gene_names, per_gene_len = load_dataset_for_cv(None, drug, mode, in_dim)
    dataset_ids = list(getattr(full_ds, 'ids', []))
    id_to_idx = {fid: idx for idx, fid in enumerate(dataset_ids)}
    train_idx = [id_to_idx[fid] for fid in split['train_ids'] if fid in id_to_idx]
    test_idx = [id_to_idx[fid] for fid in split['test_ids'] if fid in id_to_idx]
    if min(len(train_idx), len(test_idx)) == 0:
        print(f'[skip] {drug} held-out lineage {heldout_lineage}: no overlap with embedding dataset')
        return None

    explain_filenames = set(recover_original_explain_filenames(drug, mode, in_dim))
    train_ids_set = {dataset_ids[i] for i in train_idx}
    bg_candidate_ids = train_ids_set - explain_filenames
    bg_idx_pool = [id_to_idx[fid] for fid in bg_candidate_ids]

    explain_idx = [id_to_idx[fid] for fid in explain_filenames if fid in id_to_idx]
    missing = len(explain_filenames) - len(explain_idx)
    if missing:
        print(f'[warn] {drug} held-out lineage {heldout_lineage}: {missing}/{len(explain_filenames)} '
              f'original explain-set isolates not found in current embedding dataset')

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)
    bg_pool_ds = torch.utils.data.Subset(full_ds, bg_idx_pool)
    explain_ds = torch.utils.data.Subset(full_ds, explain_idx)

    out_dir = OUT_ROOT / drug / f'{mode}_{in_dim}' / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _, hist = train_token_split(
        gene=None, drug=drug, mode=mode, in_dim=in_dim, batch_size=batch_size, n_epochs=n_epochs,
        lr=lr, freeze_bias_frac=freeze_bias_frac, out_root=str(out_dir),
        train_ds=train_ds, val_ds=test_ds, per_gene_len=per_gene_len, gene_names=gene_names,
        compute_shap=False,
    )

    probe_n = min(100, len(train_ds))
    l_pad = max(train_ds[i][0].shape[1] for i in range(probe_n))

    genes = DRUG2GENES[drug]
    gene_lengths_arg = per_gene_len if len(genes) > 1 else None
    gene_names_arg = gene_names if len(genes) > 1 else None

    shap_df = _shap_explain(
        model, bg_pool_ds, explain_ds, l_pad,
        background_size=min(BG_SIZE, len(bg_pool_ds)),
        per_gene_lengths=gene_lengths_arg, gene_names=gene_names_arg, device=device,
    )
    shap_df.to_pickle(out_dir / f'{drug}_dim{in_dim}_heldout_lineage_{heldout_lineage}_controlled_shap.pkl', protocol=4)

    catalog = load_catalog_normalized(WHO_CATALOG)
    pr_rows = precision_recall_from_shap(drug, genes, shap_df, catalog, k_vals=k_vals)
    for row in pr_rows:
        row.update({
            'heldout_lineage': str(heldout_lineage), 'model': f'esm_{mode}{in_dim}',
            'shap_explain_scope': 'table3_matched', 'explain_n': len(explain_ds),
            'background_pool_n': len(bg_pool_ds),
        })
    pd.DataFrame(pr_rows).to_csv(out_dir / f'PR_{drug}_heldout_lineage_{heldout_lineage}_controlled.csv', index=False)
    print(f'[ok] {drug} {mode} held-out lineage {heldout_lineage}: controlled-SHAP PR written '
          f'(explain_n={len(explain_ds)}, bg_pool_n={len(bg_pool_ds)})')
    return pr_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', choices=ELIGIBLE_DRUGS, default='rifampicin')
    parser.add_argument('--mode', default='full', choices=['full', 'pca', 'mean'])
    parser.add_argument('--in-dim', type=int, default=None)
    parser.add_argument('--heldout-lineage', default=None, choices=list(MAJOR_LINEAGES))
    parser.add_argument('--min-class-count', type=int, default=DEFAULT_MIN_CLASS_COUNT)
    args = parser.parse_args()

    in_dim = args.in_dim if args.in_dim is not None else MODE2DIM[args.mode]
    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)

    all_pr = []
    for heldout in heldouts:
        try:
            rows = run_controlled_shap_for_drug(args.drug, heldout, args.mode, in_dim, min_class_count=args.min_class_count)
            if rows:
                all_pr.extend(rows)
        except Exception as e:
            print(f'[error] {args.drug} held-out lineage {heldout}: {e!r}; continuing')

    if all_pr:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_pr).to_csv(OUT_ROOT / f'combined_controlled_shap_{args.drug}_esm_{args.mode}{in_dim}.csv', index=False)
        print(f'[done] wrote combined_controlled_shap_{args.drug}_esm_{args.mode}{in_dim}.csv')


if __name__ == '__main__':
    main()
