"""Task 2 (WHO-catalogued resistance variant recovery) under lineage holdout,
for the one-hot CNN. Trains on the leave-one-major-lineage-out split from
lineage_holdout_cnn.py, computes SHAP on the held-out lineage test set, and
converts it to Precision@k/Recall@k via interp_pr_utils (same methodology as
the random-split Table 5 CNN rows), so results are directly comparable.
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
sys.path.insert(0, str(THIS_DIR))

from lineage_split_utils import build_and_save_drug_splits, DEFAULT_MIN_CLASS_COUNT, MAJOR_LINEAGES
from cnn_model import ProteinCNN1x1
from cnn_utils import ProteinDataset, bootstrap_auc_ci, set_seed, shap_per_residue
from significance_testing_cnn import _train_one_fold, DRUG2GENES, N_EPOCHS, LR, BATCH_SIZE, DEVICE, SEED
from interp_pr_utils import load_catalog_normalized, precision_recall_from_shap

OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/cnn_task2'
WHO_CATALOG = PROTEIN_TASKS_DIR / 'data/filtered_variants_output.csv'
ELIGIBLE_DRUGS = ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'streptomycin',
                  'capreomycin', 'moxifloxacin', 'ethionamide']
BG_SIZE = 100
EXPL_SAMPLES = 200


def run_lineage_task2_for_drug(drug: str, heldout_lineage: str, k_vals=(1, 5, 10),
                                min_class_count: int = DEFAULT_MIN_CLASS_COUNT, dry_run: bool = False) -> dict:
    df, splits = build_and_save_drug_splits(drug, DRUG2GENES, min_class_count=min_class_count)
    split = splits[str(heldout_lineage)]
    result = {
        'drug': drug, 'heldout_lineage': str(heldout_lineage),
        'train_n': split['train_counts']['n'], 'train_r': split['train_counts']['r'], 'train_s': split['train_counts']['s'],
        'test_n': split['test_counts']['n'], 'test_r': split['test_counts']['r'], 'test_s': split['test_counts']['s'],
        'feasible': split['feasible'],
    }
    if not split['feasible']:
        print(f'[skip] {drug} held-out lineage {heldout_lineage}: underpowered')
        return result
    if dry_run:
        print(f'[dry-run] {drug} held-out lineage {heldout_lineage}: {result}')
        return result

    full_ds = ProteinDataset(df['Protein_Sequence'].tolist(), (df['Phenotype'] == 'R').astype(int).tolist())
    id_to_idx = {fid: idx for idx, fid in enumerate(df['Filename'].astype(str).tolist())}
    train_idx = [id_to_idx[fid] for fid in split['train_ids'] if fid in id_to_idx]
    test_idx = [id_to_idx[fid] for fid in split['test_ids'] if fid in id_to_idx]
    train_ds = Subset(full_ds, train_idx)
    test_ds = Subset(full_ds, test_idx)

    genes = DRUG2GENES[drug]
    per_gene_lengths = None
    gene_names = None
    if len(genes) > 1:
        seq_meta = pd.read_csv(PROTEIN_TASKS_DIR / 'data/catalog/protein_sequences.csv')
        per_gene_lengths = [len(seq_meta.loc[seq_meta['gene'] == g, 'protein_sequence'].values[0]) for g in genes]
        gene_names = genes

    model = ProteinCNN1x1(seq_len=full_ds.seq_len, in_dim=20).to(DEVICE)
    curve_df, probs, gold, metrics = _train_one_fold(
        model, train_ds, test_ds, n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, device=DEVICE,
    )

    out_dir = OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(out_dir / 'training_curve.csv', index=False)
    pd.DataFrame({'prob': probs, 'label': gold}).to_csv(out_dir / 'test_preds.csv', index=False)
    torch.save(model.state_dict(), out_dir / f'{drug}_cnn.pt')

    boot_mean, (ci_lo, ci_hi) = bootstrap_auc_ci(np.asarray(gold), np.asarray(probs), n_boot=5000, alpha=0.05, seed=SEED)
    result.update({
        'auc': metrics['auc'], 'acc': metrics['acc'], 'sens': metrics['sens'], 'spec': metrics['spec'],
        'pooled_boot_auc_mean': boot_mean, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
    })

    shap_df = shap_per_residue(
        model=model, train_ds=train_ds, val_ds=test_ds,
        background_size=min(BG_SIZE, len(train_ds)), explain_samples=min(EXPL_SAMPLES, len(test_ds)),
        per_gene_lengths=per_gene_lengths, gene_names=gene_names, device=DEVICE,
    )
    shap_df.to_pickle(out_dir / f'{drug}_heldout_lineage_{heldout_lineage}_shap.pkl', protocol=4)

    catalog = load_catalog_normalized(WHO_CATALOG)
    pr_rows = precision_recall_from_shap(drug, genes, shap_df, catalog, k_vals=k_vals)
    for row in pr_rows:
        row.update({'heldout_lineage': str(heldout_lineage), 'model': 'cnn'})
    pd.DataFrame(pr_rows).to_csv(out_dir / f'PR_{drug}_heldout_lineage_{heldout_lineage}.csv', index=False)

    pd.DataFrame([result]).to_csv(out_dir / 'summary.csv', index=False)
    print(f"[ok] {drug} held-out lineage {heldout_lineage}: auc={metrics['auc']:.3f}")
    return {**result, 'pr_rows': pr_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', choices=ELIGIBLE_DRUGS, default=None)
    parser.add_argument('--heldout-lineage', default=None, choices=list(MAJOR_LINEAGES))
    parser.add_argument('--min-class-count', type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    set_seed(SEED)
    drugs = [args.drug] if args.drug else ELIGIBLE_DRUGS
    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)

    all_pr = []
    for drug in drugs:
        for heldout in heldouts:
            res = run_lineage_task2_for_drug(drug, heldout, min_class_count=args.min_class_count, dry_run=args.dry_run)
            if not args.dry_run and res.get('pr_rows'):
                all_pr.extend(res['pr_rows'])

    if all_pr:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_pr).to_csv(OUT_ROOT / 'combined_lineage_task2_precision_recall_cnn.csv', index=False)
        print(f'[done] wrote {OUT_ROOT / "combined_lineage_task2_precision_recall_cnn.csv"}')


if __name__ == '__main__':
    main()
