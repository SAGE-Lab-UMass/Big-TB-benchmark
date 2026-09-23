"""Recovers the exact isolate identities of the original random-split
SHAP explain set, for the one-hot CNN/Transformer pipeline.

Both CNN and Transformer draw from the identical pool and split, reconstructed as:
  1. merge each gene's sequence CSV -> one row per isolate (frameshift/R-S filtered)
  2. train_test_split(test_size=0.2, random_state=42, stratify=Phenotype)
  3. concat(train, test) -> deduplicate using the cached dedup_ohe/*_dedup_indices.npy
  4. the saved *_shap_all.pkl's `sample_idx` indexes directly into that
     deduplicated pool - those rows ARE the original explain set.
 reconstructed labels match the saved shap_all.pkl labels 100%
for rifampicin (N_unique=712 matches meta.json exactly).
"""
from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = THIS_DIR.parent
SEQ_DIR = PROTEIN_TASKS_DIR / 'data/latest/sequence_data_csv'


def _build_drug_df(drug: str, genes: list[str]) -> pd.DataFrame:
    gene_dfs = []
    for g in genes:
        path = SEQ_DIR / f'{g}_{drug.upper()}_combined_sequence_data.csv'
        d = pd.read_csv(path)
        d = d[(d['Frameshift_Mutation'] == 0) & (d['Phenotype'].isin(['R', 'S']))].copy()
        d = d[['Filename', 'Protein_Sequence', 'Phenotype']].rename(columns={'Protein_Sequence': f'seq_{g}'})
        gene_dfs.append(d)
    merged = reduce(lambda a, b: pd.merge(a, b, on=['Filename', 'Phenotype'], how='inner'), gene_dfs)
    return merged


def recover_original_explain_filenames(drug: str, genes: list[str], shap_dir: Path) -> list[str]:
    """Returns the list of isolate Filenames that made up the original
    random-split (Table 3) SHAP explain set for this drug."""
    df = _build_drug_df(drug, genes)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Phenotype'])
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    dedup_idx = np.load(shap_dir / 'dedup_ohe' / f'{drug}_ohe_full_dedup_indices.npy')
    full_unique = full_df.iloc[dedup_idx].reset_index(drop=True)

    shap_all = pd.read_pickle(shap_dir / f'{drug}_cnn_shap_all.pkl' if 'cnn' in str(shap_dir)
                               else shap_dir / f'{drug}_transformer_shap_all.pkl')
    sample_idx = shap_all['sample_idx'].to_numpy()
    return full_unique.iloc[sample_idx]['Filename'].astype(str).tolist()
