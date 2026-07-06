
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
import torch
import glob, os, math, sys, json, time, gc
import numpy as np, pandas as pd
from pathlib import Path


import random, shap
import torch.nn as nn
import torch.nn.functional as F
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

ESM_MODELS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = ESM_MODELS_DIR.parent
LOCAL_DATA_ROOT = PROTEIN_TASKS_DIR / "data" / "latest"
SOURCE_DATA_ROOT = Path(os.environ.get("BIGTB_SOURCE_DATA_ROOT", LOCAL_DATA_ROOT))

SPECIAL_DIR_NAMES = {
    ("gyrA", "levofloxacin") : "gyrA_LEV",
    ("gyrB", "levofloxacin") : "gyrB_LEV",
    ("gyrA", "moxifloxacin") : "gyrA_MOX",
    ("gyrB", "moxifloxacin") : "gyrB_MOX",
    ("ethA", "ethionamide")  : "ethA_ETH",
    ("ethR", "ethionamide")  : "ethR_ETH",
    ("inhA", "ethionamide")  : "inhA_ETH",
}


def _existing_dir(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def results_root() -> Path:
    return LOCAL_DATA_ROOT / "results"


def cross_val_root() -> Path:
    return LOCAL_DATA_ROOT / "cross_val"


def sequence_csv_path(gene: str, drug: str) -> Path:
    name = f"{gene}_{drug.upper()}_combined_sequence_data.csv"
    return _existing_dir(
        LOCAL_DATA_ROOT / "sequence_data_csv" / name,
        SOURCE_DATA_ROOT / "sequence_data_csv" / name,
    )


def embeddings_root(gene: str, drug: str | None = None) -> Path:
    dirname = SPECIAL_DIR_NAMES.get((gene, drug), gene)
    return _existing_dir(
        LOCAL_DATA_ROOT / "embeddings" / dirname,
        SOURCE_DATA_ROOT / "embeddings" / dirname,
    )


def resolve_checkpoint_path(fold_dir: Path, drug: str, fold: int | None = None) -> Path:
    candidates = [fold_dir / f"{drug}_model.pt"]
    if fold is not None:
        candidates.append(fold_dir / f"{drug}_fold{fold}_model.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing checkpoint. Tried: " + ", ".join(str(path) for path in candidates)
    )

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
        csv = sequence_csv_path(g, drug)
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
        paths = [sequence_csv_path(gene, drug)]
    else:
        genes = multi_drugs[drug]
        paths = [sequence_csv_path(g, drug)
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

