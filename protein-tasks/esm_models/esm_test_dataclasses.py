#required imports
from sklearn.metrics import roc_auc_score
import glob, os, math, sys, json, time, gc, random, shap
import numpy as np, pandas as pd
from pathlib import Path
# import tqdm
import torch
import time
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

single_drugs = {
    "rifampicin" : ["rpoB"],
    "pyrazinamide": ["pncA"],
    "capreomycin" : ["tlyA"],
    "amikacin"    : ["eis"]
}

multi_drugs = {
    "streptomycin": ["rpsL", "gid"],
    "isoniazid"   : ["katG", "inhA"],
    "ethionamide" : ["ethA", "ethR","inhA"],
    "ethambutol"  : ["embC","embA","embB"],
    "moxifloxacin": ["gyrA", "gyrB"],
    "levofloxacin": ["gyrA", "gyrB"]
}

all_drugs = {**single_drugs, **multi_drugs}   # merge dicts

# -------------------------------------------------------------------
SPECIAL_DIRS = {
    ("gyrA", "levofloxacin") : "data/latest/embeddings/gyrA_LEV",
    ("gyrB", "levofloxacin") : "data/latest/embeddings/gyrB_LEV",
    ("gyrA", "moxifloxacin") : "data/latest/embeddings/gyrA_MOX",
    ("gyrB", "moxifloxacin") : "data/latest/embeddings/gyrB_MOX",
    ("ethA", "ethionamide")  : "data/latest/embeddings/ethA_ETH",
    ("ethR", "ethionamide")  : "data/latest/embeddings/ethR_ETH",
    ("inhA", "ethionamide")  : "data/latest/embeddings/inhA_ETH",
    
}

def embeddings_root(gene: str, drug: str | None = None) -> Path:
    return Path(SPECIAL_DIRS.get((gene, drug),
                                 f"data/latest/embeddings/{gene}"))


# -------------------------
# helper: pad-collate
# -------------------------

## Step 3: multi gene data processing
# ──────────────────────────────────────────────────────────────
# Dataset #1 : raw 320-dim token embeddings, concatenated length-wise
# ──────────────────────────────────────────────────────────────
class MultiGeneConcatDataset(torch.utils.data.Dataset):
    """
    Concatenates per-gene token tensors on demand:
        genes = ["katG","inhA"]  →  x shape (320 , L_katG+L_inhA) →  (320 , 740+269)
    """
    def __init__(self, genes, drug, label_map):
        self.genes  = genes
        self.label  = label_map
        self.drug   = drug
        
        # build a list of mem-map blocks per gene: open mem-maps per gene
        self.blocks = {}              # gene → [(ids , mm), …]
        for g in genes:
            data_path = embeddings_root(g, drug) 
            metas = sorted(Path(f"{data_path}/token").glob("*_meta.npz"))
            gene_blocks  = []
            for mp in metas:
                meta = np.load(mp, allow_pickle=True)
                mm   = np.memmap(str(mp).replace("_meta.npz", ".mmap"),
                                 dtype="float16", mode="r",
                                 shape=tuple(meta["shape"]))
                gene_blocks.append((meta["identifier"].astype(str), mm))
            self.blocks[g] = gene_blocks
        # common isolate ids
        id_sets = [set(id_ for blk in self.blocks[g] for id_ in blk[0]) for g in genes]
        self.ids = sorted(set.intersection(*id_sets))  #global index
        # skip unlabeled isolates 
        self.ids = [i for i in self.ids if i in label_map]

    def __len__(self): return len(self.ids)

    # small helper: fetch one isolate from one gene
    def _row(self, gene, ident):
        for ids, mm in self.blocks[gene]:
            w = np.where(ids == ident)[0]
            if w.size:                                  # (L , 320) ➜ (320 , L)
                return torch.from_numpy(mm[w[0]].astype("float32")).t()   # → (320 , L)
        raise KeyError(f"{ident} missing in {gene}") 

    def __getitem__(self, idx):
        ident = self.ids[idx]
        parts = [self._row(g, ident) for g in self.genes]      # list of (320 , Lg)
        x     = torch.cat(parts, dim=1)                  # length-concat concat on length axis → (320 , ∑L)
        y     = torch.tensor(self.label[ident], dtype=torch.float32)
        return x, y



# ──────────────────────────────────────────────────────────────
# Dataset #2 : the same idea, but after PCA compression to `k` dims
# ──────────────────────────────────────────────────────────────
class PcaMultiGeneConcatDataset(torch.utils.data.Dataset):
    """
    Returns PCA-compressed token tensors.
        genes = ["rpsL","gid"], k = 10  →
            x  shape (k , L_rpsL+L_gid)
    """
    def __init__(self, genes, drug,label_map, k):
        self.genes, self.k = genes, k
        self.label_map = label_map
        self.drug = drug

        # open PCA-compressed mem-maps  (live under .../token/PCA/)
        self.blocks = {}           # gene → list[(ids, memmap)]
        for g in genes:
            data_path = embeddings_root(g, drug) 
            metas = sorted(
                Path(f"{data_path}/token/PCA")
                .glob(f"*_pc{k}_meta.npz")          # ← look only in PCA/
            )
            if not metas:
                raise FileNotFoundError(f"No *_pc{k}_meta.npz for {g}")
            g_blocks = []
            for mp in metas:
                meta = np.load(mp, allow_pickle=True)
                mmap_path = mp.parent / mp.name.replace("_meta.npz", ".mmap")
                mm = np.memmap(mmap_path, dtype="float16",
                               mode="r", shape=tuple(meta["shape"]))
                g_blocks.append((meta["identifier"].astype(str), mm))
            self.blocks[g] = g_blocks

        # identifiers present in *all* genes
        id_sets = [set(id_ for blk in self.blocks[g] for id_ in blk[0])
                   for g in genes]
        self.ids = sorted(set.intersection(*id_sets))
        # skip unlabeled isolates 
        self.ids = [i for i in self.ids if i in label_map]

    def __len__(self): return len(self.ids)

    def _row(self, g, ident):
        for ids, mm in self.blocks[g]:
            w = np.where(ids == ident)[0]
            if w.size:
                return torch.from_numpy(mm[w[0]].astype("float32")).t()  # (k , Lg)
        raise KeyError(f"{ident} missing in {g}")

    def __getitem__(self, idx):
        ident = self.ids[idx]
        parts = [self._row(g, ident) for g in self.genes] # (k , Lg)
        x = torch.cat(parts, dim=1)                       # length-concat (k , ∑L)
        y = torch.tensor(self.label_map[ident], dtype=torch.float32)
        return x, y


class MeanMultiGeneConcatDataset(torch.utils.data.Dataset):
    """
    Concatenate (k=1) mean-compressed tokens for several genes.
    Each sample → tensor shape (1 , ΣL_gene)   and a label.
    """
    def __init__(self, genes, meta_paths, label_map):
        self.genes  = genes
        self.blocks = {}
        self.ids    = None

        for g in genes:
            g_meta = [p for p in meta_paths if p.name.startswith(f"{g}_")]
            blks   = []
            for mp in g_meta:
                m  = np.load(mp, allow_pickle=True)
                mmap_path = Path(str(mp).replace("_pcmean_meta.npz", "_pcmean.mmap"))
                mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=tuple(m["shape"]))
                blks.append((m["identifier"].astype(str), mm))
            self.blocks[g] = blks

            ids_here = set(id_ for ids,_ in blks for id_ in ids)
            self.ids = ids_here if self.ids is None else self.ids & ids_here

        # Only keep ids that also exist in the label map
        self.ids = sorted(self.ids & set(label_map.keys()))
        self.label = {k: label_map[k] for k in self.ids}

    def __len__(self): return len(self.ids)

    def _row(self, g, ident):
        for ids, mm in self.blocks[g]:
            idx = np.where(ids == ident)[0]
            if idx.size:
                # (L,1) → (1,L)
                return torch.from_numpy(mm[idx[0]].astype("float32")).t()
        raise KeyError(f"{ident} missing in {g}")

    def __getitem__(self, idx):
        ident = self.ids[idx]
        if ident not in self.label:
            raise KeyError(f"Missing key: {ident}")
        parts = [self._row(g, ident) for g in self.genes]
        x = torch.cat(parts, dim=1)  # (1, ΣL)
    
        y_val = self.label[ident]
        try:
            y = torch.tensor(int(y_val), dtype=torch.float32)
        except Exception as e:
            raise TypeError(
                f"Label for {ident} is {y_val!r} (type {type(y_val)}), cannot convert to int"
            ) from e
    
        return x, y




## step 4: memory map file loading

def pad_collate(batch, L_PAD):
    """Right-pad every sequence in the mini-batch to the same length."""
    xs, ys = zip(*batch)
    xs_pad = [F.pad(x, (0, L_PAD - x.shape[1])) if x.shape[1] < L_PAD else x
              for x in xs]
    return torch.stack(xs_pad), torch.stack(ys)

# ─────────────────────────────────────────────────────────
#  Map-style datasets that stream from float-16 mem-maps
# ─────────────────────────────────────────────────────────
#
# Why two classes?
#   • TokenMemmapMap → raw 320-channel ESM tokens
#   • PcaMemmapMap   → PCA-compressed version (k channels)
#
# Both expose   __len__   and   __getitem__   so they plug straight
# into a PyTorch DataLoader.
# --------------------------------------------------------

class TokenMemmapMap(torch.utils.data.Dataset):
    def __init__(self, meta_paths, label_dict):
        self.blocks = []          # (ids, memmap)
        self.lookup = []          # (block_idx, row_idx)
        self.label_dict = label_dict

        for bidx, meta_path in enumerate(meta_paths):
            meta = np.load(meta_path, allow_pickle=True)
            ids  = meta["identifier"].astype(str)   # ensure str
            shape = tuple(meta["shape"])            # (n, L, 320)
            mmap_path = meta_path.replace("_meta.npz", ".mmap")
            mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=shape)

            self.blocks.append((ids, mm))

            # only keep rows whose IDs exist in label_dict
            for r in range(shape[0]):
                if ids[r] in label_dict:
                    self.lookup.append((bidx, r))

    def __len__(self): 
        return len(self.lookup)

    def __getitem__(self, idx):
        bidx, ridx = self.lookup[idx]
        ids, mm = self.blocks[bidx]
        seq_id   = ids[ridx]
        x = torch.from_numpy(mm[ridx].astype("float32")).t()   # (320, L)
        y = torch.tensor(self.label_dict[seq_id], dtype=torch.float32)
        return x, y




class PcaMemmapMap(torch.utils.data.Dataset):          # PCA-K
    """
    PCA-compressed matrices.
      • in_dim = k (e.g. 10)
      • each row shape: (L , k)  → we transpose to (k , L)
    """
    def __init__(self, meta_paths, label_dict, k):
        self.blocks = []   # [(ids, memmap_k), …]
        self.lookup = []   # list of (block_index, row_index)
        self.k = k
        self.label = label_dict

        for bidx, p in enumerate(meta_paths):
            m  = np.load(p, allow_pickle=True)
            ids = m["identifier"].astype(str)
            mm = np.memmap(
                p.replace("_meta.npz", ".mmap"),
                dtype="float16", mode="r",
                shape=tuple(m["shape"])   # (N, L, k)
            )
            self.blocks.append((ids, mm))

            # Only keep rows that exist in label_dict
            for r, id_ in enumerate(ids):
                if id_ in self.label:
                    self.lookup.append((bidx, r))

    def __len__(self):  
        return len(self.lookup)

    def __getitem__(self, idx):
        b, r = self.lookup[idx]
        ids, mm = self.blocks[b]
        ident = ids[r]
        x = torch.from_numpy(mm[r].astype("float32")).t()   # (K, L)
        y = torch.tensor(self.label[ident], dtype=torch.float32)
        return x, y


class MeanMemmapMap(torch.utils.data.Dataset):
    """
    Loads the *_pcmean.mmap chunks produced above.
    Each item → (tensor [1,L] float32 ,  label)
    """
    def __init__(self, meta_paths, label_dict):
        self.blocks, self.lookup = [], []
        self.label = label_dict

        for bidx, p in enumerate(meta_paths):
            m  = np.load(p, allow_pickle=True)
            mmap_path = Path(str(p).replace("_pcmean_meta.npz", "_pcmean.mmap"))
            mm = np.memmap(mmap_path, dtype="float16", mode="r", shape=tuple(m["shape"]))
            ids = m["identifier"].astype(str)   # make sure all string
            self.blocks.append((ids, mm))

            # only keep rows with known labels
            for r, id_ in enumerate(ids):
                if id_ in self.label:
                    self.lookup.append((bidx, r))

    def __len__(self):  
        return len(self.lookup)

    def __getitem__(self, idx):
        b, r = self.lookup[idx]
        ids, mm = self.blocks[b]
        ident = ids[r]
        x = torch.from_numpy(mm[r].astype("float32")).T      # → (1 , L)
        y = torch.tensor(self.label[ident], dtype=torch.float32)
        return x, y
