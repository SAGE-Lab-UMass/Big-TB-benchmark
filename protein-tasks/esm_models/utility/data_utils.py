
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
import torch
import glob, os, math, sys,json,time,gc
import numpy as np, pandas as pd
from pathlib import Path


import random, shap
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ## Step 3: multi gene data processing


from tqdm.auto import tqdm
#declare gene-drug map
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

# ──────────────────────────────────────────────────────────────
# helper ─ build a {isolate-id → label} dict for a *set* of genes
#          0 = susceptible (S) , 1 = resistant (R)
# ──────────────────────────────────────────────────────────────

def build_label_map(genes,drug):
    """
    genes  : ["gyrA", "gyrB"]
    returns: { isolate_id -> 0/1 }  (0 = S, 1 = R)
    """
    label_map = {}

    for g in genes:
        csv = Path(f"data/latest/sequence_data_csv/{g}_{drug.upper()}_combined_sequence_data.csv")
        df  = pd.read_csv(csv, usecols=["Filename", "Phenotype"])
        df["id"]    = df["Filename"].astype(str)
        df["label"] = (df["Phenotype"] == "R").astype(int)
        # if an isolate appears in both files, assert the labels agree
        for _id, lbl in zip(df["id"], df["label"]):
            if _id in label_map and label_map[_id] != lbl:
                raise ValueError(f"Phenotype disagreement for {_id} between genes")
            label_map[_id] = lbl

    return label_map

# -------------------------
# helper: pad-collate
# -------------------------

def pad_collate(batch, L_PAD):
    """Right-pad every sequence in the mini-batch to the same length."""
    xs, ys = zip(*batch)
    xs_pad = [F.pad(x, (0, L_PAD - x.shape[1])) if x.shape[1] < L_PAD else x
              for x in xs]
    return torch.stack(xs_pad), torch.stack(ys)




def build_train_test_split(drug, test_size=0.2, seed=42):
    """Load sequence CSV(s) for a drug, split into train/test, return filenames + labels."""

    if drug in single_drugs:
        gene = single_drugs[drug][0]
        paths = [f"data/latest/sequence_data_csv/{gene}_{drug.upper()}_combined_sequence_data.csv"]
    else:
        genes = multi_drugs[drug]
        paths = [f"data/latest/sequence_data_csv/{g}_{drug.upper()}_combined_sequence_data.csv"
                 for g in genes]

    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")
        dfs.append(pd.read_csv(p, usecols=["Filename", "Phenotype"]))

    ph = pd.concat(dfs, ignore_index=True)

    files  = ph["Filename"].astype(str).values
    labels = (ph["Phenotype"] == "R").astype(int).tolist()   # ← cast to list of ints

    X_train, X_test, y_train, y_test = train_test_split(
        files, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # Force Python int
    y_train = [int(y) for y in y_train]
    y_test  = [int(y) for y in y_test]

    return (X_train, y_train), (X_test, y_test)


def evaluate_auc(model, dataset, device="cuda"):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda b: pad_collate(b, dataset[0][0].shape[1]))
    preds, gold = [], []
    model.eval()
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(device)
            preds.extend(torch.sigmoid(model(xb)).cpu().numpy())
            gold.extend(yb.numpy())
    return roc_auc_score(gold, preds)


