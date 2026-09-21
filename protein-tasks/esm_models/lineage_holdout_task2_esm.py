"""Task 2 (WHO-catalogued resistance variant recovery) under lineage holdout,
for ESM embedding CNN models.

NOTE - output not used for reporting: per the 2026-09 agreement with
Sai/Anna/Mahbuba, Task 2's lineage-aware result uses the FIXED random-split
(Table 3) explain set with the lineage-trained model swapped in, not this
script's held-out-lineage explain set - see controlled_shap_task2_esm.py for
that actual deliverable. Unlike the CNN/Transformer versions, this script
isn't a training dependency for the controlled one (ESM's controlled
script retrains from scratch since no checkpoint is saved here) - so this
file's own Precision@k/Recall@k output is unused entirely, not just
unreported.

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
from significance_testing import DRUG2GENES, load_dataset_for_cv, train_token_split, _eval_subset
from data_utils import pad_collate
from shap_esm import Wrapped
from interp_pr_utils import load_catalog_normalized, precision_recall_from_shap

OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/esm_task2'
WHO_CATALOG = PROTEIN_TASKS_DIR / 'data/filtered_variants_output.csv'
ELIGIBLE_DRUGS = ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'streptomycin',
                  'capreomycin', 'moxifloxacin', 'ethionamide']
MODE2DIM = {'full': 320, 'pca': 10, 'mean': 1}
SEED = 42
BG_SIZE = 100
EXPL_SAMPLES = 200


def _shap_per_residue_lineage(model, train_ds, test_ds, l_pad, background_size, explain_samples,
                               per_gene_lengths, gene_names, device, seed=SEED):
    """Background sampled from train_ds, explained samples from test_ds (the
    held-out lineage) — mirrors one_hot_encoded/cnn_utils.shap_per_residue's
    train/val split semantics, adapted for ESM's padded variable-length inputs."""
    model = model.to(device).eval()
    rng = random.Random(seed)

    def _padded(ds, idx):
        x, y = ds[idx]
        if x.shape[1] < l_pad:
            x = torch.nn.functional.pad(x, (0, l_pad - x.shape[1]))
        return x, y

    b_n = min(background_size, len(train_ds))
    bg_idx = rng.sample(range(len(train_ds)), b_n)
    background = torch.stack([_padded(train_ds, i)[0] for i in bg_idx]).to(device)

    e_n = min(explain_samples, len(test_ds))
    ex_idx = rng.sample(range(len(test_ds)), e_n)
    xs = torch.stack([_padded(test_ds, i)[0] for i in ex_idx]).to(device)
    ys = [int(_padded(test_ds, i)[1]) for i in ex_idx]

    explainer = shap.DeepExplainer(Wrapped(model), [background])
    sv = explainer.shap_values([xs], check_additivity=False)[0]  # (E, C, L)
    imp = np.abs(sv).sum(axis=1)  # (E, L)

    out = {'sample_idx': ex_idx, 'label': ys, 'importance_full': list(imp)}
    if per_gene_lengths is not None:
        cuts = np.cumsum([0] + per_gene_lengths)
        for gi, g in enumerate(gene_names):
            out[f'importance_{g}'] = [imp[n, cuts[gi]:cuts[gi + 1]] for n in range(imp.shape[0])]
    return pd.DataFrame(out)


def run_lineage_task2_for_drug(drug: str, heldout_lineage: str, mode: str, in_dim: int, k_vals=(1, 5, 10),
                                min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
                                batch_size: int = 32, n_epochs: int = 20, lr: float = 5e-4,
                                freeze_bias_frac: float = 0.25, dry_run: bool = False) -> dict:
    df, splits = build_and_save_drug_splits(drug, DRUG2GENES, min_class_count=min_class_count)
    split = splits[str(heldout_lineage)]
    result = {
        'drug': drug, 'mode': mode, 'in_dim': in_dim, 'heldout_lineage': str(heldout_lineage),
        'train_n': split['train_counts']['n'], 'train_r': split['train_counts']['r'], 'train_s': split['train_counts']['s'],
        'test_n': split['test_counts']['n'], 'test_r': split['test_counts']['r'], 'test_s': split['test_counts']['s'],
        'feasible': split['feasible'],
    }
    if not split['feasible']:
        print(f'[skip] {drug} held-out lineage {heldout_lineage}: underpowered')
        return result

    full_ds, label_map, gene_names, per_gene_len = load_dataset_for_cv(None, drug, mode, in_dim)
    dataset_ids = list(getattr(full_ds, 'ids', []))
    id_to_idx = {fid: idx for idx, fid in enumerate(dataset_ids)}
    train_idx = [id_to_idx[fid] for fid in split['train_ids'] if fid in id_to_idx]
    test_idx = [id_to_idx[fid] for fid in split['test_ids'] if fid in id_to_idx]
    result.update({'train_n_dataset': len(train_idx), 'test_n_dataset': len(test_idx)})
    if min(len(train_idx), len(test_idx)) == 0:
        print(f'[skip] {drug} held-out lineage {heldout_lineage}: no overlap with embedding dataset')
        return result

    if dry_run:
        print(f'[dry-run] {drug} {mode} held-out lineage {heldout_lineage}: {result}')
        return result

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)
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
    probs, gold = _eval_subset(model, test_ds, batch_size, device, l_pad)
    pd.DataFrame({'prob': probs, 'label': gold}).to_csv(out_dir / 'test_preds.csv', index=False)

    auc = float('nan')
    if len(np.unique(gold)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(gold, probs))
    result.update({'auc': auc, 'last_val_auc': float(hist['val_auc'].iloc[-1]), 'last_val_acc': float(hist['val_acc'].iloc[-1])})

    genes = DRUG2GENES[drug]
    shap_df = _shap_per_residue_lineage(
        model, train_ds, test_ds, l_pad,
        background_size=min(BG_SIZE, len(train_ds)), explain_samples=min(EXPL_SAMPLES, len(test_ds)),
        per_gene_lengths=per_gene_len if len(genes) > 1 else None,
        gene_names=gene_names if len(genes) > 1 else None,
        device=device,
    )
    shap_df.to_pickle(out_dir / f'{drug}_dim{in_dim}_heldout_lineage_{heldout_lineage}_shap.pkl', protocol=4)

    catalog = load_catalog_normalized(WHO_CATALOG)
    pr_rows = precision_recall_from_shap(drug, genes, shap_df, catalog, k_vals=k_vals)
    for row in pr_rows:
        row.update({'heldout_lineage': str(heldout_lineage), 'model': f'esm_{mode}{in_dim}'})
    pd.DataFrame(pr_rows).to_csv(out_dir / f'PR_{drug}_heldout_lineage_{heldout_lineage}.csv', index=False)

    pd.DataFrame([result]).to_csv(out_dir / 'summary.csv', index=False)
    print(f"[ok] {drug} {mode} held-out lineage {heldout_lineage}: auc={auc:.3f}")
    return {**result, 'pr_rows': pr_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', choices=ELIGIBLE_DRUGS, default=None)
    parser.add_argument('--mode', default='full', choices=['full', 'pca', 'mean'])
    parser.add_argument('--in-dim', type=int, default=None)
    parser.add_argument('--heldout-lineage', default=None, choices=list(MAJOR_LINEAGES))
    parser.add_argument('--min-class-count', type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    in_dim = args.in_dim if args.in_dim is not None else MODE2DIM[args.mode]
    drugs = [args.drug] if args.drug else ELIGIBLE_DRUGS
    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)

    all_pr = []
    for drug in drugs:
        for heldout in heldouts:
            res = run_lineage_task2_for_drug(drug, heldout, args.mode, in_dim, min_class_count=args.min_class_count, dry_run=args.dry_run)
            if not args.dry_run and res.get('pr_rows'):
                all_pr.extend(res['pr_rows'])

    if all_pr:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_pr).to_csv(OUT_ROOT / f'combined_lineage_task2_precision_recall_esm_{args.mode}{in_dim}.csv', index=False)
        print(f'[done] wrote combined_lineage_task2_precision_recall_esm_{args.mode}{in_dim}.csv')


if __name__ == '__main__':
    main()
