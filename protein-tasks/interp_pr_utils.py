"""Shared SHAP-to-Precision/Recall@k conversion, used by the CNN, Transformer,
and ESM lineage-holdout Task 2 runners so all three report against the same
WHO bona-fide/exclusion definition used in Table 5 and the regression runner
(regression_utils.py / lineage_holdout_task2.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED_CONF = ['1) Assoc w R', '2) Assoc w R - Interim']


def load_catalog_normalized(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    catalog['aa_pos_0idx'] = catalog['aa_pos'].astype(int) - 1
    catalog['_gene_norm'] = catalog['gene'].astype(str).str.lower().str.strip()
    return catalog


def who_sets_for_genes(catalog: pd.DataFrame, genes):
    genes_norm = {g.lower().strip() for g in genes}
    sub = catalog[catalog['_gene_norm'].isin(genes_norm)].copy()

    is_bona = sub['confidence'].isin(ALLOWED_CONF) & (sub['intersectional'] == True)
    gold = set(zip(sub.loc[is_bona, '_gene_norm'], sub.loc[is_bona, 'aa_pos_0idx']))

    is_unc_or_noti = (sub['confidence'] == '3) Uncertain significance') | (sub['intersectional'] != True)
    excl_all = set(zip(sub.loc[is_unc_or_noti, '_gene_norm'], sub.loc[is_unc_or_noti, 'aa_pos_0idx']))
    excl = excl_all - gold
    return gold, excl


def combined_rank_from_shap(shap_df: pd.DataFrame, genes) -> pd.DataFrame:
    rows = []
    multi = len(genes) > 1
    for g in genes:
        col = f'importance_{g}' if multi else 'importance_full'
        if col not in shap_df.columns:
            print(f'  [warn] missing column {col}; skipping gene {g}')
            continue
        stacks = np.stack([np.asarray(v).squeeze() for v in shap_df[col]], axis=0)
        maximp = np.abs(stacks).max(axis=0)
        lg = len(maximp)
        rows.append(pd.DataFrame({
            'gene': [g] * lg,
            'aa_pos_0idx': np.arange(lg, dtype=int),
            'score': maximp.astype(float),
        }))
    if not rows:
        return pd.DataFrame(columns=['gene', 'aa_pos_0idx', 'score'])
    comb = pd.concat(rows, ignore_index=True)
    comb['_gene_norm'] = comb['gene'].astype(str).str.lower().str.strip()
    comb = comb.sort_values(['score', '_gene_norm', 'aa_pos_0idx'], ascending=[False, True, True]).reset_index(drop=True)
    comb['rank1'] = np.arange(1, len(comb) + 1)
    return comb


def greedy_topk_pairs(rank_df: pd.DataFrame, k: int, excl_pairs: set):
    chosen = []
    for _, r in rank_df.iterrows():
        pair = (r['_gene_norm'], int(r['aa_pos_0idx']))
        if pair in excl_pairs:
            continue
        chosen.append(pair)
        if len(chosen) == k:
            break
    return chosen


def precision_recall_from_shap(drug: str, genes, shap_df: pd.DataFrame, catalog: pd.DataFrame, k_vals=(1, 5, 10)) -> list:
    """Same methodology as pr_from_shap_COMBINED_cnn.py, generalized for any
    model family's shap_df (CNN, Transformer, or ESM all share the
    importance_full / importance_<gene> column convention)."""
    rank_df = combined_rank_from_shap(shap_df, genes)
    if rank_df.empty:
        return []

    gold_pairs, excl_pairs = who_sets_for_genes(catalog, genes)
    k_gold = len(gold_pairs)

    rows = []
    for k in k_vals:
        topk_pairs = greedy_topk_pairs(rank_df, k, excl_pairs)
        k_eff = len(topk_pairs)
        tp = len(set(topk_pairs) & gold_pairs)
        prec = tp / k_eff if k_eff else 0.0
        rec = tp / k_gold if k_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) else 0.0

        if k_eff and k_gold:
            cat_pick = catalog[
                catalog['_gene_norm'].isin([g.lower() for g in genes])
                & catalog['confidence'].isin(ALLOWED_CONF)
                & (catalog['intersectional'] == True)
            ].copy()
            cat_pick['_pair'] = list(zip(cat_pick['_gene_norm'], cat_pick['aa_pos_0idx'].astype(int)))
            hits_list = (cat_pick[cat_pick['_pair'].isin(set(topk_pairs))].drop_duplicates(['_pair']))['variant'].astype(str).tolist() or ['None']
        else:
            hits_list = ['None']

        rows.append({
            'drug': drug, 'gene': 'ALL', 'k_req': k, 'k_eff': k_eff,
            'total_res_pos': k_gold, 'TP': tp,
            'precision': prec, 'recall': rec, 'F1': f1,
            'hit_variants': ', '.join(hits_list),
        })
    return rows
