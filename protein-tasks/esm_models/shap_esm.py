# compute_shap_only.py
import os, glob, math, random
import torch, shap
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from data_utils import *
from CNN import *
from esm_test_dataclasses import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── gene-drug maps ────────────────────────────────────────────────
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
all_drugs = {**single_drugs, **multi_drugs}

# ─── wrapper for SHAP ──────────────────────────────────────────────
class Wrapped(nn.Module):
    """Adapter so SHAP sees (B,1) instead of (B,)"""
    def __init__(self, base): super().__init__(); self.base = base
    def forward(self, x):     return self.base(x).unsqueeze(1)

# ─── deduplication with caching ─────────────────────────────────────
def dedup_and_save_indices(ds, name, out_dir="data/latest/results/dedup"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_dedup_indices.npy"

    # if already saved, reuse
    if out_path.exists():
        uniq_indices = np.load(out_path).tolist()
        print(f"[{name}] loaded cached indices ({len(ds)} → {len(uniq_indices)})")
        return uniq_indices

    # otherwise compute fresh
    uniq_indices, seen = [], set()
    for i in range(len(ds)):
        x, y = ds[i]
        key = (x.numpy().tobytes(), int(y))
        if key not in seen:
            seen.add(key); uniq_indices.append(i)

    np.save(out_path, uniq_indices)
    reduction = len(ds) - len(uniq_indices)
    frac = 100.0 * reduction / len(ds)
    log_msg = (f"[{name}] deduplicated {len(ds)} → {len(uniq_indices)} "
               f"({reduction} removed, {frac:.1f}% reduction)")
    print(log_msg)
    with open(out_dir / "dedup_log.txt", "a") as f:
        f.write(log_msg + "\n")

    return uniq_indices

# ─── SHAP core function (no dedup inside) ───────────────────────────
def shap_per_residue(model, train_ds, val_ds,
                     background_size, explain_samples,
                     per_gene_lengths=None, gene_names=None,
                     device="cuda"):
    """
    Compute SHAP values for a trained model.
    Assumes datasets are already deduplicated.
    """
    model = model.to(device).eval()

    # background from training set
    B = min(background_size, len(train_ds))
    bg_idx = random.sample(range(len(train_ds)), B)
    background = torch.stack([train_ds[i][0] for i in bg_idx]).to(device)

    explainer = shap.DeepExplainer(Wrapped(model), [background])

    # explanation from validation set
    E = min(explain_samples, len(val_ds))
    samp_idx = random.sample(range(len(val_ds)), E)
    xs = torch.stack([val_ds[i][0] for i in samp_idx]).to(device)
    ys = [val_ds[i][1] for i in samp_idx]

    # compute SHAP
    sv  = explainer.shap_values([xs], check_additivity=False)[0]  # (E,C,L)
    imp = np.abs(sv).sum(axis=1)                                  # (E,L)

    out = {
        "sample_idx": samp_idx,
        "label": [int(y) for y in ys],
        "importance_full": list(imp)
    }

    if per_gene_lengths is not None:
        cuts = np.cumsum([0] + per_gene_lengths)
        for gi, g in enumerate(gene_names):
            out[f"importance_{g}"] = [imp[n, cuts[gi]:cuts[gi+1]] for n in range(E)]

    return pd.DataFrame(out)

# ─── reload trained model ───────────────────────────────────────────
def load_model(drug, gene, mode, in_dim, run_dir):
    if drug in multi_drugs:
        genes = multi_drugs[drug]
        per_gene_len, gene_names = [], []
        for g in genes:
            mp = next(Path(embeddings_root(g, drug) / "token").glob("*_meta.npz"))
            Lg = int(np.load(mp, allow_pickle=True)["shape"][1])
            per_gene_len.append(Lg); gene_names.append(g)
        L_PAD = sum(per_gene_len)
    else:
        mp = next(Path(embeddings_root(gene, drug) / "token").glob("*_meta.npz"))
        L_PAD = int(np.load(mp, allow_pickle=True)["shape"][1])
        per_gene_len = [L_PAD]; gene_names = [gene]

    stem_out = 64 if in_dim == 320 else 32
    model = ProteinCNN1x1(seq_len=L_PAD, in_dim=in_dim, stem_out=stem_out).to(device)

    model_path = Path(run_dir) / f"{drug}_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, per_gene_len, gene_names, L_PAD

# ─── driver for SHAP only ───────────────────────────────────────────
def compute_shap_for_drug(drug, mode="full", in_dim=320,
                          background_frac=1.0, explain_frac=1.0,
                          out_root="data/latest/results"):

    gene = single_drugs.get(drug, [None])[0] if drug in single_drugs else None
    run_dir = f"{out_root}/prediction/esm/{drug}_dim{in_dim}"
    model, per_gene_len, gene_names, L_PAD = load_model(drug, gene, mode, in_dim, run_dir)

    # --- datasets ---
    (train_files, y_train), (test_files, y_test) = build_train_test_split(drug)
    train_label_map = dict(zip(train_files, y_train))
    test_label_map  = dict(zip(test_files,  y_test))
    full_label_map  = {**train_label_map, **test_label_map}

    def make_dataset(files, labels):
        label_map = dict(zip(files, labels))
        if drug in multi_drugs:
            genes = multi_drugs[drug]
            if mode == "full":
                return MultiGeneConcatDataset(genes, drug, label_map)
            elif mode == "mean":
                metas = []
                for g in genes:
                    data_path = embeddings_root(g, drug)
                    metas += [Path(p) for p in glob.glob(f"{data_path}/token/MEAN/*_pcmean_meta.npz")]
                return MeanMultiGeneConcatDataset(genes, metas, label_map)
            elif mode == "pca":
                return PcaMultiGeneConcatDataset(genes, drug, label_map, k=in_dim)
        else:
            data_path = embeddings_root(gene, drug)
            if mode == "full":
                metas = [p for p in glob.glob(f"{data_path}/token/*_meta.npz") if "_pc" not in p]
                return TokenMemmapMap(metas, label_map)
            elif mode == "mean":
                metas = [p for p in glob.glob(f"{data_path}/token/MEAN/*_pcmean_meta.npz")]
                return MeanMemmapMap(metas, label_map)
            elif mode == "pca":
                metas = glob.glob(f"{data_path}/token/PCA/*_pc{in_dim}_meta.npz")
                return PcaMemmapMap(metas, label_map, k=in_dim)

    train_ds = make_dataset(train_files, y_train)
    test_ds  = make_dataset(test_files,  y_test)
    full_ds  = make_dataset(list(full_label_map.keys()), list(full_label_map.values()))

    # --- deduplication ---
    train_idx = dedup_and_save_indices(train_ds, f"{drug}_train")
    test_idx  = dedup_and_save_indices(test_ds,  f"{drug}_test")
    full_idx  = dedup_and_save_indices(full_ds,  f"{drug}_full")
    train_ds  = torch.utils.data.Subset(train_ds, train_idx)
    test_ds   = torch.utils.data.Subset(test_ds, test_idx)
    full_ds   = torch.utils.data.Subset(full_ds, full_idx)

    out_path = Path(f"{out_root}/interpretability/{mode}_{in_dim}")
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- exploratory (train+test) ----
    bg_size_all = max(1, int(len(train_ds) * background_frac))
    ex_size_all = max(1, int(len(full_ds) * explain_frac))
    shap_df_all = shap_per_residue(model, train_ds, full_ds,
                                   background_size=bg_size_all, explain_samples=ex_size_all,
                                   per_gene_lengths=per_gene_len, gene_names=gene_names, device=device)
    shap_df_all.to_pickle(out_path / f"{drug}_dim{in_dim}_shap_all.pkl", protocol=4)

    # ---- deployment (test only) ----
    bg_size_test = max(1, int(len(train_ds) * background_frac))
    ex_size_test = max(1, int(len(test_ds) * explain_frac))
    shap_df_test = shap_per_residue(model, train_ds, test_ds,
                                    background_size=bg_size_test, explain_samples=ex_size_test,
                                    per_gene_lengths=per_gene_len, gene_names=gene_names, device=device)
    shap_df_test.to_pickle(out_path / f"{drug}_dim{in_dim}_shap_test.pkl", protocol=4)

    print(f"[done] {drug}: exploratory={len(full_ds)} | test={len(test_ds)} isolates")
