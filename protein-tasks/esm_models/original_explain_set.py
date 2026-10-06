"""Recovers the exact isolate identities of the SHAP explain set used in the original ESM lineage run, for any of the five drugs, using the SAME live code
as compute_shap_for_drug() in shap_esm.py: build_train_test_split -> build the
embedding dataset -> dedup_and_save_indices -> stratified_bg_explain_indices_from_ds.
for rifampicin this reproduces the identical 712-isolate
pool (472R/240S) that CNN's dedup produces - confirming the intended unified
background/explain methodology really is shared across architectures. 
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from data_utils import build_train_test_split
from esm_test_dataclasses import (
    MeanMemmapMap,
    MeanMultiGeneConcatDataset,
    MultiGeneConcatDataset,
    PcaMemmapMap,
    PcaMultiGeneConcatDataset,
    TokenMemmapMap,
    embeddings_root,
)
from shap_esm import dedup_and_save_indices, multi_drugs, single_drugs, stratified_bg_explain_indices_from_ds

SEED = 42
BG_FRAC = 0.10
MAX_BG = 160


def _make_dataset(drug: str, mode: str, in_dim: int, files, labels):
    label_map = dict(zip(files, labels))
    if drug in multi_drugs:
        genes = multi_drugs[drug]
        if mode == 'full':
            return MultiGeneConcatDataset(genes, drug, label_map)
        elif mode == 'mean':
            metas = []
            for g in genes:
                data_path = embeddings_root(g, drug)
                metas += [Path(p) for p in glob.glob(f'{data_path}/token/MEAN/*_pcmean_meta.npz')]
            return MeanMultiGeneConcatDataset(genes, metas, label_map)
        elif mode == 'pca':
            return PcaMultiGeneConcatDataset(genes, drug, label_map, k=in_dim)
    else:
        gene = single_drugs[drug][0]
        data_path = embeddings_root(gene, drug)
        if mode == 'full':
            metas = [p for p in glob.glob(f'{data_path}/token/*_meta.npz') if '_pc' not in p]
            return TokenMemmapMap(metas, label_map)
        elif mode == 'mean':
            metas = [p for p in glob.glob(f'{data_path}/token/MEAN/*_pcmean_meta.npz')]
            return MeanMemmapMap(metas, label_map)
        elif mode == 'pca':
            metas = glob.glob(f'{data_path}/token/PCA/*_pc{in_dim}_meta.npz')
            return PcaMemmapMap(metas, label_map, k=in_dim)


def _filename_for_index(ds, idx: int) -> str:
    """MultiGeneConcatDataset (and its PCA/mean variants) expose a flat
    `.ids` list indexed directly by position. Single-gene Token/Pca/Mean
    MemmapMap datasets instead store (ids, memmap) blocks plus a
    (block_idx, row_idx) `.lookup` table."""
    if hasattr(ds, 'ids'):
        return str(ds.ids[idx])
    bidx, ridx = ds.lookup[idx]
    ids, _ = ds.blocks[bidx]
    return str(ids[ridx])


def recover_original_explain_filenames(drug: str, mode: str = 'full', in_dim: int = 320) -> list[str]:
    (train_files, y_train), (test_files, y_test) = build_train_test_split(drug)
    full_label_map = {**dict(zip(train_files, y_train)), **dict(zip(test_files, y_test))}
    full_ds = _make_dataset(drug, mode, in_dim, list(full_label_map.keys()), list(full_label_map.values()))

    dedup_idx = dedup_and_save_indices(full_ds, f'{drug}_full')

    import torch
    pool_subset = torch.utils.data.Subset(full_ds, dedup_idx)
    _, explain_idx, _ = stratified_bg_explain_indices_from_ds(pool_subset, bg_frac=BG_FRAC, seed=SEED, max_bg=MAX_BG)

    # explain_idx indexes into pool_subset -> map to full_ds indices -> Filenames
    full_ds_indices = [dedup_idx[int(i)] for i in explain_idx]
    return [_filename_for_index(full_ds, i) for i in full_ds_indices]
