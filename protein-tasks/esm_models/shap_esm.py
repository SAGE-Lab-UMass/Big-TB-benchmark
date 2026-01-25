# compute_shap_only.py
import os, glob, math, random, json
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
def dedup_and_save_indices(ds, name, out_dir="/project/pi_annagreen_umass_edu/mahbuba/Data-Curation-for-MTB/protein-tasks/data/latest/results/dedup"):
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




# ---- helper: disjoint split on a single dataset ----
import numpy as np

from sklearn.model_selection import train_test_split
import numpy as np

def stratified_bg_explain_indices_from_ds(ds, bg_frac=0.10, seed=42, max_bg=None):
    """
    Disjoint BG/EX split for a torch-style dataset ds (getitem -> (x, y)).

    Guarantees:
      • If BOTH classes (R=1, S=0) exist in the pool, BG is stratified and contains both.
      • If only one class exists, BG is a random subset.
      • BG and EX are disjoint; EX is non-empty.
      • If max_bg is provided, |BG| ≤ max_bg (label balance preserved when possible).

    Returns: (bg_idx, ex_idx, y_pool)
      bg_idx, ex_idx: np.int64 arrays of indices into ds
      y_pool:         np.int64 array of 0/1 labels for ds
    """
    N = len(ds)
    assert N >= 2, "Need at least 2 samples to split."
    y_pool = np.array([int(ds[i][1]) for i in range(N)], dtype=int)
    idx = np.arange(N)

    # 1) initial split (stratified if both labels present)
    if np.unique(y_pool).size > 1: # both classes present
        bg_idx, ex_idx = train_test_split(
            idx, train_size=bg_frac, stratify=y_pool, random_state=seed
        ) #stratified split
    else:  # single-class pool → random split
        rng = np.random.default_rng(seed)
        cut = max(1, int(round(bg_frac * N)))
        perm = rng.permutation(idx)
        bg_idx, ex_idx = perm[:cut], perm[cut:] #random split

     # 2) optional cap on BG size, preserving class presence if possible
    if max_bg is not None and len(bg_idx) > max_bg: 
        if np.unique(y_pool[bg_idx]).size > 1:
            bg_idx, _ = train_test_split(
                bg_idx, train_size=max_bg, stratify=y_pool[bg_idx], random_state=seed
            )
        else:
            rng = np.random.default_rng(seed)
            bg_idx = rng.choice(bg_idx, size=max_bg, replace=False)

        mask = np.ones(N, dtype=bool); mask[bg_idx] = False
        ex_idx = np.where(mask)[0] # recompute ex_idx as complement

    # ensure at least one isolate exists to explain
    if len(ex_idx) == 0:
        ex_idx = np.array([bg_idx[-1]]) #move one from bg to ex
        bg_idx = bg_idx[:-1] #remove last from bg

    return bg_idx.astype(np.int64), ex_idx.astype(np.int64), y_pool


# def disjoint_bg_explain_indices(n_samples, bg_frac=0.10, seed=42, min_bg=1, min_explain=1):
#     range_seed = np.random.default_rng(seed)
#     perm = range_seed.permutation(n_samples)
#     bg_num = max(min_bg, int(round(bg_frac * n_samples)))
#     bg_num = min(bg_num, n_samples - min_explain)  # ensure at least one to explain
#     bg_idx = perm[:bg_num]
#     explain_idx = perm[bg_num:]
#     return bg_idx, explain_idx, bg_num

def shap_per_residue_single_pool(model, pool_ds,
                                 bg_frac=0.10, seed=42, max_bg=None,
                                 per_gene_lengths=None, gene_names=None,
                                 device="cuda", run_name=""):
    model = model.to(device).eval()
    N = len(pool_ds); assert N > 1

    # NEW: stratified, disjoint BG/EX (on unique pool labels)
    bg_idx, explain_idx, y_pool = stratified_bg_explain_indices_from_ds(
        pool_ds, bg_frac=bg_frac, seed=seed, max_bg=max_bg
    )

    meta = {
        "run": run_name,
        "N": int(N),
        "bg_frac": float(bg_frac),
        "max_bg": None if max_bg is None else int(max_bg),
        "actual_bg": int(len(bg_idx)),
        "actual_explain": int(len(explain_idx)),
        "seed": int(seed),
        # split sanity:
        "bg_pos": int((y_pool[bg_idx] == 1).sum()),
        "bg_neg": int((y_pool[bg_idx] == 0).sum()),
        "ex_pos": int((y_pool[explain_idx] == 1).sum()),
        "ex_neg": int((y_pool[explain_idx] == 0).sum()),
    }
    print(f"[SHAP] {run_name} | BG={meta['actual_bg']} (R={meta['bg_pos']}, S={meta['bg_neg']}) "
          f"| EX={meta['actual_explain']} (R={meta['ex_pos']}, S={meta['ex_neg']})")

    # tensors
    background = torch.stack([pool_ds[int(i)][0] for i in bg_idx]).to(device)
    xs         = torch.stack([pool_ds[int(i)][0] for i in explain_idx]).to(device)
    ys         = [int(pool_ds[int(i)][1]) for i in explain_idx]

    explainer = shap.DeepExplainer(Wrapped(model), [background])
    sv  = explainer.shap_values([xs], check_additivity=False)[0]  # (E, C, L)
    imp = np.abs(sv).sum(axis=1)                                  # (E, L)

    out = {
        "sample_idx": list(map(int, explain_idx)),
        "label": ys,
        "importance_full": list(imp)
    }
    if per_gene_lengths is not None:
        cuts = np.cumsum([0] + per_gene_lengths)
        for gi, g in enumerate(gene_names):
            out[f"importance_{g}"] = [imp[n, cuts[gi]:cuts[gi+1]] for n in range(imp.shape[0])]

    return pd.DataFrame(out), meta


# # ---- SHAP core over a single pool (no duplication) ----
# def shap_per_residue_single_pool(model, pool_ds,
#                                  bg_frac=0.10, seed=42, max_bg=None,
#                                  per_gene_lengths=None, gene_names=None,
#                                  device="cuda", run_name=""):
#     """
#     Compute SHAP with a single deduplicated dataset:
#       - background := first bg_frac of a permutation (optionally capped by max_bg)
#       - explained  := the complement
#     Ensures disjoint sets and full coverage.
#     """
#     model = model.to(device).eval()

#     N = len(pool_ds)
#     assert N > 1, "Need at least 2 samples to split background/explain."
#     bg_idx, explain_idx, bg_target = disjoint_bg_explain_indices(N, bg_frac=bg_frac, seed=seed)
#     if max_bg is not None:
#         bg_idx = np.asarray(bg_idx)[:max_bg]


#     # --- visibility: print & meta dict ---
#     meta = {
#         "run": run_name,
#         "N": int(N),
#         "bg_frac": float(bg_frac),
#         "max_bg": None if max_bg is None else int(max_bg),
#         "bg_target": int(bg_target),
#         "actual_bg": int(len(bg_idx)),
#         "actual_explain": int(len(explain_idx)),
#         "seed": int(seed),
#     }
#     print(f"[SHAP] {run_name} N={N} | bg_frac={bg_frac:.2f}, max_bg={max_bg} "
#           f" background={len(bg_idx)}, explain={len(explain_idx)}")
    
#     # tensors
#     background = torch.stack([pool_ds[int(i)][0] for i in bg_idx]).to(device)
#     xs         = torch.stack([pool_ds[int(i)][0] for i in explain_idx]).to(device)
#     ys         = [int(pool_ds[int(i)][1]) for i in explain_idx]

#     # SHAP
#     explainer = shap.DeepExplainer(Wrapped(model), [background])
#     sv  = explainer.shap_values([xs], check_additivity=False)[0]  # (E, C, L)
#     imp = np.abs(sv).sum(axis=1)                                  # (E, L)

#     # pack results
#     out = {
#         "sample_idx": list(map(int, explain_idx)),
#         "label": ys,
#         "importance_full": list(imp)
#     }
#     if per_gene_lengths is not None:
#         cuts = np.cumsum([0] + per_gene_lengths)
#         for gi, g in enumerate(gene_names):
#             out[f"importance_{g}"] = [imp[n, cuts[gi]:cuts[gi+1]] for n in range(imp.shape[0])]

#     return pd.DataFrame(out), meta

# ---- driver:the single-pool SHAP for the exploratory run ----
def compute_shap_for_drug(drug, mode="full", in_dim=320,
                          background_frac=0.10, explain_frac=None,   # explain_frac unused now
                          seed=42, max_bg=160,
                          out_root="/project/pi_annagreen_umass_edu/mahbuba/Data-Curation-for-MTB/protein-tasks/data/latest/results"):

    gene = single_drugs.get(drug, [None])[0] if drug in single_drugs else None
    run_dir = f"{out_root}/prediction/esm/{drug}_dim{in_dim}"
    model, per_gene_len, gene_names, L_PAD = load_model(drug, gene, mode, in_dim, run_dir)

    # datasets (build + dedup)
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

    # deduplicate each, then use full_ds for the exploratory SHAP
    train_ds = torch.utils.data.Subset(train_ds, dedup_and_save_indices(train_ds, f"{drug}_train"))
    test_ds  = torch.utils.data.Subset(test_ds,  dedup_and_save_indices(test_ds,  f"{drug}_test"))
    full_ds  = torch.utils.data.Subset(full_ds,  dedup_and_save_indices(full_ds,  f"{drug}_full"))

    out_path = Path(f"{out_root}/interpretability/{mode}_{in_dim}"); out_path.mkdir(parents=True, exist_ok=True)

    # ---- Exploratory (explain all unique sequences exactly once; bg=10%, explain=90%) ----
    run_name = f"{drug}_dim{in_dim}_{mode}"
    shap_df_all, meta = shap_per_residue_single_pool(
        model, full_ds, bg_frac=background_frac, seed=seed, max_bg=max_bg,
        per_gene_lengths=per_gene_len, gene_names=gene_names, device=device,
        run_name=run_name
    )

    shap_df_all.to_pickle(out_path / f"{drug}_dim{in_dim}_shap_all.pkl", protocol=4)
    print(f"[done] {drug}: exploratory={len(full_ds)} unique isolates (bg~{background_frac:.0%}, max_bg={max_bg})")

    with open(out_path / f"{drug}_dim{in_dim}_shap_all_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# # ─── SHAP core function (no dedup inside) ───────────────────────────
# def shap_per_residue(model, train_ds, val_ds,
#                      background_size, explain_samples,
#                      per_gene_lengths=None, gene_names=None,
#                      device="cuda"):
#     """
#     Compute SHAP values for a trained model.
#     Assumes datasets are already deduplicated.
#     """
#     model = model.to(device).eval()

#     # background from training set
#     B = min(background_size, len(train_ds))
#     bg_idx, explain_idx = split_bg_explain_indices(len(X), bg_frac=0.10, seed=42)
#     bg_idx = random.sample(range(len(train_ds)), B)
#     background = torch.stack([train_ds[i][0] for i in bg_idx]).to(device)

#     explainer = shap.DeepExplainer(Wrapped(model), [background])

#     # explanation from validation set
#     E = min(explain_samples, len(val_ds))
#     samp_idx = random.sample(range(len(val_ds)), E)
#     xs = torch.stack([val_ds[i][0] for i in samp_idx]).to(device)
#     ys = [val_ds[i][1] for i in samp_idx]

#     # compute SHAP
#     sv  = explainer.shap_values([xs], check_additivity=False)[0]  # (E,C,L)
#     imp = np.abs(sv).sum(axis=1)                                  # (E,L)

#     out = {
#         "sample_idx": samp_idx,
#         "label": [int(y) for y in ys],
#         "importance_full": list(imp)
#     }

#     if per_gene_lengths is not None:
#         cuts = np.cumsum([0] + per_gene_lengths)
#         for gi, g in enumerate(gene_names):
#             out[f"importance_{g}"] = [imp[n, cuts[gi]:cuts[gi+1]] for n in range(E)]

#     return pd.DataFrame(out)

# ─── reload trained model ───────────────────────────────────────────
def load_model(drug, gene, mode, in_dim, run_dir):
    if drug in multi_drugs:
        genes = multi_drugs[drug]
        per_gene_len, gene_names = [], []
        for g in genes:
            print(g, embeddings_root(g, drug))
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

# # ─── driver for SHAP only ───────────────────────────────────────────
# def compute_shap_for_drug(drug, mode="full", in_dim=320,
#                           background_frac=1.0, explain_frac=1.0,
#                           out_root="/project/pi_annagreen_umass_edu/mahbuba/Data-Curation-for-MTB/protein-tasks/data/latest/results"):

#     gene = single_drugs.get(drug, [None])[0] if drug in single_drugs else None
#     run_dir = f"{out_root}/prediction/esm/{drug}_dim{in_dim}"
#     model, per_gene_len, gene_names, L_PAD = load_model(drug, gene, mode, in_dim, run_dir)

#     # --- datasets ---
#     (train_files, y_train), (test_files, y_test) = build_train_test_split(drug)
#     train_label_map = dict(zip(train_files, y_train))
#     test_label_map  = dict(zip(test_files,  y_test))
#     full_label_map  = {**train_label_map, **test_label_map}

#     def make_dataset(files, labels):
#         label_map = dict(zip(files, labels))
#         if drug in multi_drugs:
#             genes = multi_drugs[drug]
#             if mode == "full":
#                 return MultiGeneConcatDataset(genes, drug, label_map)
#             elif mode == "mean":
#                 metas = []
#                 for g in genes:
#                     data_path = embeddings_root(g, drug)
#                     metas += [Path(p) for p in glob.glob(f"{data_path}/token/MEAN/*_pcmean_meta.npz")]
#                 return MeanMultiGeneConcatDataset(genes, metas, label_map)
#             elif mode == "pca":
#                 return PcaMultiGeneConcatDataset(genes, drug, label_map, k=in_dim)
#         else:
#             data_path = embeddings_root(gene, drug)
#             if mode == "full":
#                 metas = [p for p in glob.glob(f"{data_path}/token/*_meta.npz") if "_pc" not in p]
#                 return TokenMemmapMap(metas, label_map)
#             elif mode == "mean":
#                 metas = [p for p in glob.glob(f"{data_path}/token/MEAN/*_pcmean_meta.npz")]
#                 return MeanMemmapMap(metas, label_map)
#             elif mode == "pca":
#                 metas = glob.glob(f"{data_path}/token/PCA/*_pc{in_dim}_meta.npz")
#                 return PcaMemmapMap(metas, label_map, k=in_dim)

#     train_ds = make_dataset(train_files, y_train)
#     test_ds  = make_dataset(test_files,  y_test)
#     full_ds  = make_dataset(list(full_label_map.keys()), list(full_label_map.values()))

#     # --- deduplication ---
#     train_idx = dedup_and_save_indices(train_ds, f"{drug}_train")
#     test_idx  = dedup_and_save_indices(test_ds,  f"{drug}_test")
#     full_idx  = dedup_and_save_indices(full_ds,  f"{drug}_full")
#     train_ds  = torch.utils.data.Subset(train_ds, train_idx)
#     test_ds   = torch.utils.data.Subset(test_ds, test_idx)
#     full_ds   = torch.utils.data.Subset(full_ds, full_idx)

#     out_path = Path(f"{out_root}/interpretability/{mode}_{in_dim}")
#     out_path.mkdir(parents=True, exist_ok=True)

#     # ---- exploratory (train+test) ----
#     bg_size_all = max(1, int(len(train_ds) * background_frac))
#     ex_size_all = max(1, int(len(full_ds) * explain_frac))
#     shap_df_all = shap_per_residue(model, train_ds, full_ds,
#                                    background_size=bg_size_all, explain_samples=ex_size_all,
#                                    per_gene_lengths=per_gene_len, gene_names=gene_names, device=device)
#     shap_df_all.to_pickle(out_path / f"{drug}_dim{in_dim}_shap_all.pkl", protocol=4)

#     # # ---- deployment (test only) ----
#     # bg_size_test = max(1, int(len(train_ds) * background_frac))
#     # ex_size_test = max(1, int(len(test_ds) * explain_frac))
#     # shap_df_test = shap_per_residue(model, train_ds, test_ds,
#     #                                 background_size=bg_size_test, explain_samples=ex_size_test,
#     #                                 per_gene_lengths=per_gene_len, gene_names=gene_names, device=device)
#     # shap_df_test.to_pickle(out_path / f"{drug}_dim{in_dim}_shap_test.pkl", protocol=4)

#     print(f"[done] {drug}: exploratory={len(full_ds)} | test={len(test_ds)} isolates")
