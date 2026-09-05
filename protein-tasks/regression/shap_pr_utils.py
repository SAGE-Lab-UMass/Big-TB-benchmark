"""SHAP + combined-ranking Precision@k/Recall@k utilities for the Ref-Alt
logistic-regression pipeline 
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


def dedup_Xy(X, y):
    Xc = np.ascontiguousarray(X)
    view = Xc.view([('', Xc.dtype)] * Xc.shape[1])
    _, idx = np.unique(view, return_index=True)
    idx = np.sort(idx)
    return Xc[idx], np.asarray(y)[idx], idx


def stratified_bg_explain_indices(y, bg_frac=0.10, seed=42, max_bg=None):
    """Disjoint (bg_idx, ex_idx); BG ~= bg_frac of the pool, stratified by
    class when both classes are present, optionally capped at max_bg."""
    y = np.asarray(y).astype(int)
    N = len(y)
    idx = np.arange(N)

    if len(np.unique(y)) > 1:
        bg_idx, ex_idx = train_test_split(idx, train_size=bg_frac, stratify=y, random_state=seed)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        cut = max(1, int(round(bg_frac * N)))
        bg_idx, ex_idx = perm[:cut], perm[cut:]

    if max_bg is not None and len(bg_idx) > max_bg:
        bg_idx, _ = train_test_split(
            bg_idx, train_size=max_bg,
            stratify=y[bg_idx] if len(np.unique(y[bg_idx])) > 1 else None,
            random_state=seed,
        )
        mask = np.ones(N, dtype=bool)
        mask[bg_idx] = False
        ex_idx = np.where(mask)[0]

    return np.array(bg_idx), np.array(ex_idx)


def linear_shap_explain(model_fitted, X_bg, X_explain, masker_mode="independent"):
    """shap.LinearExplainer over a pre-chosen background/explain partition
    (the lineage-holdout variant: background = train partition, explain =
    held-out-lineage test partition, instead of the random-split notebook's
    pooled-then-resplit background/explain sample)."""
    masker = shap.maskers.Impute(X_bg) if masker_mode == "impute" else shap.maskers.Independent(X_bg)
    expl = shap.LinearExplainer(model_fitted, masker)
    phi = expl.shap_values(X_explain)
    return phi


def greedy_topk_global(rank_df, k, exclude):
    chosen = []
    for pos in rank_df["Residue_Position"]:
        if pos in exclude:
            continue
        chosen.append(pos)
        if len(chosen) == k:
            break
    return chosen


def druglevel_gold_excl(gene_slices, who_df_full, catalog_full):
    """Global (0-based, concatenated-matrix) gold/excluded position sets for
    the whole drug. who_df_full must already be filtered to bona-fide
    (confidence in {1,2} AND intersectional==True); catalog_full is the raw,
    unfiltered WHO catalogue used to detect uncertain/non-intersectional
    positions to exclude from the greedy top-k walk."""
    gold, excl = set(), set()
    for g, (start, end) in gene_slices.items():
        rows_bona = who_df_full[who_df_full["gene"].str.lower() == g.lower()]
        rows_all = catalog_full[catalog_full["gene"].str.lower() == g.lower()]
        gold_local = set(rows_bona["aa_pos_0idx"].astype(int).tolist())
        mask_unc = rows_all["confidence"].eq("3) Uncertain significance")
        mask_not_i = rows_all["intersectional"] != True
        excl_local = set(rows_all.loc[mask_unc | mask_not_i, "aa_pos_0idx"].astype(int).tolist()) - gold_local
        gold |= {start + p for p in gold_local}
        excl |= {start + p for p in excl_local}
    return gold, excl


def build_hit_variants(top_global_positions, gene_slices, who_df):
    """Variant strings (unique by site) for the global top-k picks."""
    hits = []
    for pos in top_global_positions:
        for g, (start, end) in gene_slices.items():
            if start <= pos < end:
                local_pos0 = pos - start
                m = (who_df["gene"].str.lower() == g.lower()) & (who_df["aa_pos_0idx"].astype(int) == local_pos0)
                vs = who_df.loc[m, "variant"].drop_duplicates().tolist()
                if vs:
                    hits.extend(vs)
                break
    seen, uniq = set(), []
    for v in hits:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq
