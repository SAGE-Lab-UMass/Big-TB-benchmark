# significance_testing_transformer.py
# Transformer CV + significance + per-residue SHAP (per fold)
import os, math, random, argparse
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =========================
# Data loading (merge genes)
# =========================
DRUG2GENES = {
    "rifampicin"  : ["rpoB"],
    "pyrazinamide": ["pncA"],
    "capreomycin" : ["tlyA"],
    "amikacin"    : ["eis"],
    "moxifloxacin": ["gyrB","gyrA"],
    "levofloxacin": ["gyrB","gyrA"],
    "isoniazid"   : ["katG","inhA"],
    "streptomycin": ["rpsL","gid"],
    "ethambutol"  : ["embC","embA","embB"],
    "ethionamide" : ["ethA","ethR","inhA"],
}

DATA_DIR = Path("data/latest/sequence_data_csv")

# Mapping for 20 standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

        lengths = [len(seq) for seq in sequences]
        min_len = min(lengths)
        max_len = max(lengths)

        if max_len - min_len > 2:
            raise ValueError(f"Sequences vary too much in length! Found lengths: {set(lengths)}")

        self.seq_len = max_len  # allow minor difference (pad shorter sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        onehot = np.zeros((20, self.seq_len), dtype=np.float32)
        for i, aa in enumerate(seq):
            if i >= self.seq_len:  # (safety) but should never happen
                break
            if aa in AA_TO_INDEX:
                onehot[AA_TO_INDEX[aa], i] = 1.0

        return torch.tensor(onehot), torch.tensor(label, dtype=torch.float32)



class Wrapped(nn.Module):
    """
    Tiny adapter so SHAP sees output shape (B,1) instead of (B,)
    """
    def __init__(self, base): super().__init__(); self.base = base
    def forward(self, x):     return self.base(x).unsqueeze(1)

def shap_per_residue(model, train_ds, val_ds,
                     background_size=100, explain_samples=200,
                     per_gene_lengths=None, gene_names=None,
                     device="cuda"):
    """
    Returns a DataFrame with columns:
       sample_idx , label , importance_full , importance_<gene1> , …
    * importance_… columns hold np.ndarray vectors with per-residue |SHAP|.
    """
    model = model.to(device).eval()

    # ---- (1) pick SHAP background from *training* set ------------
    B = min(background_size, len(train_ds))
    bg_idx = random.sample(range(len(train_ds)), B)
    background = torch.stack([train_ds[i][0] for i in bg_idx]).to(device)

    explainer = shap.DeepExplainer(Wrapped(model), [background])

    # ---- (2) pick validation samples to explain ------------------
    E = min(explain_samples, len(val_ds))
    samp_idx = random.sample(range(len(val_ds)), E)
    xs = torch.stack([val_ds[i][0] for i in samp_idx]).to(device)
    ys = [val_ds[i][1] for i in samp_idx]

    # ---- (3) compute SHAP ---------------------------------------
    sv  = explainer.shap_values([xs], check_additivity=False)[0]   # (E,C,L)
    imp = np.abs(sv).sum(axis=1)                                   # (E,L)

    out = {
        "sample_idx"      : samp_idx,
        "label"           : [int(y) for y in ys],
        "importance_full" : list(imp),
    }

    # ---- (4) slice per-gene chunks if needed ---------------------
    if per_gene_lengths is not None:
        cuts = np.cumsum([0] + per_gene_lengths)                  # e.g. [0,2517,4545]
        for gi, g in enumerate(gene_names):
            out[f"importance_{g}"] = [imp[n, cuts[gi]:cuts[gi+1]] for n in range(E)]

    return pd.DataFrame(out)

# ─── 1.  FIND THE RIGHT CSV FOR (gene, drug) ───────────────────────────
def _csv_path(gene:str, drug:str)->Path:
    special = DATA_DIR / f"{gene}_{drug.upper()}_combined_sequence_data.csv"
    generic = DATA_DIR / f"{gene}_combined_sequence_data.csv"
    if special.exists(): return special
    if generic.exists(): return generic
    raise FileNotFoundError(f"Missing CSV for {gene} ({drug})")


# ─── 2.  BUILD ONE DATAFRAME PER DRUG (concatenated sequences) ─────────
def build_drug_df(drug:str)->pd.DataFrame:
    gene_dfs = []
    for g in DRUG2GENES[drug]:
        df = pd.read_csv(_csv_path(g, drug))
        df = df[(df["Frameshift_Mutation"]==0) &
                (df["Phenotype"].isin(["R","S"]))].copy()
        df = df[["Filename","Protein_Sequence","Phenotype"]]
        df = df.rename(columns={"Protein_Sequence": f"seq_{g}"})
        gene_dfs.append(df)

    # inner-join on Filename & Phenotype so we keep only isolates
    # present in *all* genes
    def _merge(a,b):
        return pd.merge(a,b,on=["Filename","Phenotype"], how="inner")
    merged = reduce(_merge, gene_dfs)

    # sanity: no conflicting phenotypes
    assert merged["Phenotype"].nunique() <= 2

    # concatenate sequences in gene order
    merged["Protein_Sequence"] = merged[[f"seq_{g}" for g in DRUG2GENES[drug]]].agg("".join, axis=1)
    # merged = merged[["Filename","Protein_Sequence","Phenotype"]]
    return merged


def bootstrap_auc_ci(y, p, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2: continue
        boots.append(roc_auc_score(yy, pp))
    if not boots: return np.nan, (np.nan, np.nan)
    boots = np.array(boots)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(np.mean(boots)), (float(lo), float(hi))

def one_sample_test_vs_half(auc_values):
    a = np.asarray(auc_values, dtype=float)
    mu = np.mean(a); sd = np.std(a, ddof=1) if len(a)>1 else np.nan
    if np.isnan(sd) or sd==0: return mu, sd, np.nan
    z = (mu - 0.5) / (sd / math.sqrt(len(a)))
    from math import erf, sqrt
    p = 2 * (1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    return mu, sd, p


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False