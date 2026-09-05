"""Controlled SHAP comparison for Transformer Task 2 lineage holdout . Background = lineage model's own train
partition minus explain-set overlap; explain set = Table 3's exact original
isolate identities (recovered via original_explain_set.py), not a resample.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Subset

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROTEIN_TASKS_DIR))
sys.path.insert(0, str(THIS_DIR))

from lineage_split_utils import build_and_save_drug_splits, DEFAULT_MIN_CLASS_COUNT, MAJOR_LINEAGES
from transformer import ProteinTransformer
from transformer_utils import ProteinDataset, shap_per_residue
from significance_testing_transformer import DRUG2GENES, DEVICE
from interp_pr_utils import load_catalog_normalized, precision_recall_from_shap
from original_explain_set import recover_original_explain_filenames

LINEAGE_OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/transformer_task2'
CONTROLLED_OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/transformer_task2_controlled_shap'
SHAP_SOURCE_DIR = PROTEIN_TASKS_DIR / 'data/latest/results/interpretability/transformer'
WHO_CATALOG = PROTEIN_TASKS_DIR / 'data/filtered_variants_output.csv'
ELIGIBLE_DRUGS = ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'streptomycin']
BG_SIZE = 100
SEED = 42


def run_controlled_shap_for_drug(drug: str, heldout_lineage: str, k_vals=(1, 5, 10),
                                  min_class_count: int = DEFAULT_MIN_CLASS_COUNT, seed: int = SEED):
    ckpt_path = LINEAGE_OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}' / f'{drug}_transformer.pt'
    if not ckpt_path.exists():
        print(f'[skip] {drug} held-out lineage {heldout_lineage}: no checkpoint at {ckpt_path}')
        return None

    df, splits = build_and_save_drug_splits(drug, DRUG2GENES, min_class_count=min_class_count)
    split = splits[str(heldout_lineage)]
    if not split['feasible']:
        print(f'[skip] {drug} held-out lineage {heldout_lineage}: underpowered')
        return None

    genes = DRUG2GENES[drug]
    explain_filenames = set(recover_original_explain_filenames(drug, genes, SHAP_SOURCE_DIR))

    full_ds = ProteinDataset(df['Protein_Sequence'].tolist(), (df['Phenotype'] == 'R').astype(int).tolist())
    filenames = df['Filename'].astype(str).tolist()
    id_to_idx = {fid: idx for idx, fid in enumerate(filenames)}

    train_ids = [fid for fid in split['train_ids'] if fid in id_to_idx]
    bg_candidate_ids = [fid for fid in train_ids if fid not in explain_filenames]
    bg_idx_pool = [id_to_idx[fid] for fid in bg_candidate_ids]

    explain_idx = [id_to_idx[fid] for fid in explain_filenames if fid in id_to_idx]
    missing = len(explain_filenames) - len(explain_idx)
    if missing:
        print(f'[warn] {drug} held-out lineage {heldout_lineage}: {missing}/{len(explain_filenames)} '
              f'original explain-set isolates not found in current dataset')

    explain_ds = Subset(full_ds, explain_idx)
    bg_pool_ds = Subset(full_ds, bg_idx_pool)

    per_gene_lengths = None
    gene_names = None
    if len(genes) > 1:
        seq_meta = pd.read_csv(PROTEIN_TASKS_DIR / 'data/catalog/protein_sequences.csv')
        per_gene_lengths = [len(seq_meta.loc[seq_meta['gene'] == g, 'protein_sequence'].values[0]) for g in genes]
        gene_names = genes

    model = ProteinTransformer().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    random.seed(seed)

    shap_df = shap_per_residue(
        model=model, train_ds=bg_pool_ds, val_ds=explain_ds,
        background_size=min(BG_SIZE, len(bg_pool_ds)), explain_samples=len(explain_ds),
        per_gene_lengths=per_gene_lengths, gene_names=gene_names, device=DEVICE,
    )

    out_dir = CONTROLLED_OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}'
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_df.to_pickle(out_dir / f'{drug}_heldout_lineage_{heldout_lineage}_controlled_shap.pkl', protocol=4)

    catalog = load_catalog_normalized(WHO_CATALOG)
    pr_rows = precision_recall_from_shap(drug, genes, shap_df, catalog, k_vals=k_vals)
    for row in pr_rows:
        row.update({
            'heldout_lineage': str(heldout_lineage), 'model': 'transformer',
            'shap_explain_scope': 'table3_matched', 'explain_n': len(explain_ds),
            'background_pool_n': len(bg_pool_ds),
        })
    pd.DataFrame(pr_rows).to_csv(out_dir / f'PR_{drug}_heldout_lineage_{heldout_lineage}_controlled.csv', index=False)
    print(f'[ok] {drug} held-out lineage {heldout_lineage}: controlled-SHAP PR written '
          f'(explain_n={len(explain_ds)}, bg_pool_n={len(bg_pool_ds)})')
    return pr_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', choices=ELIGIBLE_DRUGS, default=None)
    parser.add_argument('--heldout-lineage', default=None, choices=list(MAJOR_LINEAGES))
    parser.add_argument('--k-vals', default='1,5,10')
    args = parser.parse_args()

    drugs = [args.drug] if args.drug else ELIGIBLE_DRUGS
    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)
    k_vals = tuple(int(k) for k in args.k_vals.split(','))

    all_pr = []
    for drug in drugs:
        for heldout in heldouts:
            try:
                pr_rows = run_controlled_shap_for_drug(drug, heldout, k_vals=k_vals)
                if pr_rows:
                    all_pr.extend(pr_rows)
            except Exception as e:
                print(f'[error] {drug} held-out lineage {heldout}: {e!r}; continuing')

    if all_pr:
        CONTROLLED_OUT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_pr).to_csv(CONTROLLED_OUT_ROOT / 'combined_controlled_shap_precision_recall_transformer.csv', index=False)
        print(f'[done] wrote {CONTROLLED_OUT_ROOT / "combined_controlled_shap_precision_recall_transformer.csv"}')


if __name__ == '__main__':
    main()
