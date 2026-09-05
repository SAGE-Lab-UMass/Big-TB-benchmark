"""Controlled SHAP comparison for CNN Task 2 lineage holdout : isolates whether the lineage-split Task 2 numbers differ
from random-split (Table 3) because of a genuine training-distribution
effect, holding the SHAP explain set fixed and identical to Table 3's own.

Design (agreed with Sai/Anna/Mahbuba):
  - Background = drawn from the lineage-holdout model's OWN training
    partition (the 3 lineages it was actually trained on) - background must
    represent this model's real reference distribution, not random split's.
  - Explain set = the EXACT isolate identities from Table 3's own random-split
    SHAP explain set (recovered via original_explain_set.py), NOT a fresh
    random draw and NOT capped to some fixed n - size is drug-specific
    (e.g. 641 isolates for rifampicin), matching what Table 3 actually used.
  - Background/explain overlap is removed by excluding any isolate that's in
    the fixed explain set from the background draw (background can shrink
    below its target size for small train partitions; SHAP's own guidance
    treats background in the ~100-1000 range as reasonable, see
    https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html).

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
from cnn_model import ProteinCNN1x1
from cnn_utils import ProteinDataset, shap_per_residue
from significance_testing_cnn import DRUG2GENES, DEVICE
from interp_pr_utils import load_catalog_normalized, precision_recall_from_shap
from original_explain_set import recover_original_explain_filenames

LINEAGE_OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/cnn_task2'
CONTROLLED_OUT_ROOT = PROTEIN_TASKS_DIR / 'data/latest/lineage_ood_all_train/cnn_task2_controlled_shap'
SHAP_SOURCE_DIR = PROTEIN_TASKS_DIR / 'data/latest/results/interpretability/cnn'
WHO_CATALOG = PROTEIN_TASKS_DIR / 'data/filtered_variants_output.csv'
ELIGIBLE_DRUGS = ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'streptomycin']
BG_SIZE = 100
SEED = 42


def run_controlled_shap_for_drug(drug: str, heldout_lineage: str, k_vals=(1, 5, 10),
                                  min_class_count: int = DEFAULT_MIN_CLASS_COUNT, seed: int = SEED):
    ckpt_path = LINEAGE_OUT_ROOT / drug / f'heldout_lineage_{heldout_lineage}' / f'{drug}_cnn.pt'
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
    # background: this model's own training isolates, MINUS anything that's
    # also in the fixed explain set (keeps background/explain disjoint).
    bg_candidate_ids = [fid for fid in train_ids if fid not in explain_filenames]
    bg_idx_pool = [id_to_idx[fid] for fid in bg_candidate_ids]

    # explain: the exact Table 3 explain-set isolates, restricted to those
    # present in the current dataset (should be ~all of them).
    explain_idx = [id_to_idx[fid] for fid in explain_filenames if fid in id_to_idx]
    missing = len(explain_filenames) - len(explain_idx)
    if missing:
        print(f'[warn] {drug} held-out lineage {heldout_lineage}: {missing}/{len(explain_filenames)} '
              f'original explain-set isolates not found in current dataset')

    train_ds = Subset(full_ds, [id_to_idx[fid] for fid in train_ids])
    explain_ds = Subset(full_ds, explain_idx)
    bg_pool_ds = Subset(full_ds, bg_idx_pool)

    per_gene_lengths = None
    gene_names = None
    if len(genes) > 1:
        seq_meta = pd.read_csv(PROTEIN_TASKS_DIR / 'data/catalog/protein_sequences.csv')
        per_gene_lengths = [len(seq_meta.loc[seq_meta['gene'] == g, 'protein_sequence'].values[0]) for g in genes]
        gene_names = genes

    model = ProteinCNN1x1(seq_len=full_ds.seq_len, in_dim=20).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    rng = random.Random(seed)
    random.seed(seed)  # shap_per_residue uses the module-level `random`

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
            'heldout_lineage': str(heldout_lineage), 'model': 'cnn',
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
        pd.DataFrame(all_pr).to_csv(CONTROLLED_OUT_ROOT / 'combined_controlled_shap_precision_recall_cnn.csv', index=False)
        print(f'[done] wrote {CONTROLLED_OUT_ROOT / "combined_controlled_shap_precision_recall_cnn.csv"}')


if __name__ == '__main__':
    main()
