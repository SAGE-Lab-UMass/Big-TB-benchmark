
'''
notebook for per-token embedding and pca-10 embedding training and causal variant discovery experiment. 
We are training on (N,L, 320) or (N, L, 10) to improve model performance. 
The notebook is organized in the following way:
1. Memory efficient loading of per token embedding files
2. PCA code for single gene drugs and multi-gene drugs. 
3. Mutli-gene data processing (concatenating token embeddings)
4. CNN model (inspired by MD-CNN)
5. Model training (per-token and pca)
6. SHAP residue score
7. Precision and Recall

'''
from CNN import *
from data_utils import *
from utility.esm_sig_test_dataclasses import *

#required imports
from sklearn.metrics import roc_auc_score
import glob, os, math, sys,json,time,gc
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split



import random, shap
import torch.nn as nn

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


all_drugs = {**single_drugs, **multi_drugs}   # merge dicts


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.backends.cudnn.benchmark = False      # disable the exhaustive search
torch.backends.cudnn.deterministic = True   # pick a deterministic kernel

 
# ## Step 1: convert to float16 memory mapping and meta files for efficient memory


# from tqdm.auto import tqdm  
import tqdm

ESM_MODELS_DIR = Path(__file__).resolve().parent
PROTEIN_TASKS_DIR = ESM_MODELS_DIR.parent

# -------------------------------------------------------------------
SPECIAL_DIRS = {
    ("gyrA", "levofloxacin") : PROTEIN_TASKS_DIR / "data/latest/embeddings/gyrA_LEV",
    ("gyrB", "levofloxacin") : PROTEIN_TASKS_DIR / "data/latest/embeddings/gyrB_LEV",
    ("gyrA", "moxifloxacin") : PROTEIN_TASKS_DIR / "data/latest/embeddings/gyrA_MOX",
    ("gyrB", "moxifloxacin") : PROTEIN_TASKS_DIR / "data/latest/embeddings/gyrB_MOX",
    ("ethA", "ethionamide")  : PROTEIN_TASKS_DIR / "data/latest/embeddings/ethA_ETH",
    ("ethR", "ethionamide")  : PROTEIN_TASKS_DIR / "data/latest/embeddings/ethR_ETH",
    ("inhA", "ethionamide")  : PROTEIN_TASKS_DIR / "data/latest/embeddings/inhA_ETH",
    
}

# -------------------------------------------------------------------
# 2) one helper – return the directory that actually holds the .npz
# -------------------------------------------------------------------
def embeddings_root(gene: str, drug: str | None = None) -> Path:
    return Path(
        SPECIAL_DIRS.get(
            (gene, drug),
            PROTEIN_TASKS_DIR / f"data/latest/embeddings/{gene}",
        )
    )


 
# ## Significance testing


def load_dataset_for_cv(gene, drug, mode, in_dim):
    per_gene_len = []
    gene_names = []

    if drug in multi_drugs:
        genes = multi_drugs[drug]
        label_map = build_label_map(genes, drug)

        if mode == "full":
            full_ds = MultiGeneConcatDataset(genes, drug, label_map)
        elif mode == "mean":
            metas = []
            for g in genes:
                data_path = embeddings_root(g, drug)
                metas += [Path(p) for p in glob.glob(f"{data_path}/token/MEAN/*_pcmean_meta.npz")]
            full_ds = MeanMultiGeneConcatDataset(genes, metas, label_map)
        elif mode == "pca":
            full_ds = PcaMultiGeneConcatDataset(genes, drug, label_map, k=in_dim)

        for g in genes:
            data_path = embeddings_root(g, drug)
            if mode == "mean":
                mp = next(Path(f"{data_path}/token/MEAN").glob("*_pcmean_meta.npz"))
            elif mode == "full":
                mp = next(Path(f"{data_path}/token").glob("*_meta.npz"))
            elif mode == "pca":
                mp = next(Path(f"{data_path}/token/PCA").glob(f"*_pc{in_dim}_meta.npz"))
            Lg = int(np.load(mp, allow_pickle=True)["shape"][1])
            per_gene_len.append(Lg)
            gene_names.append(g)

    else:
        gene = all_drugs[drug][0]
        ph = pd.read_csv(
            sequence_csv_path(gene, drug),
            usecols=["Filename", "Phenotype", "Frameshift_Mutation"],
        )
        # Match the model-ready cohort used by all other protein runners.
        ph = ph[(ph["Frameshift_Mutation"] == 0) &
                (ph["Phenotype"].isin(["R", "S"]))].copy()
        label_map = dict(zip(ph.Filename.astype(str), (ph.Phenotype == "R").astype(int)))
        del ph
        gc.collect()

        data_path = embeddings_root(gene, drug)
        if mode == "full":
            metas = [p for p in glob.glob(f"{data_path}/token/*_meta.npz") if "_pc" not in p]
            full_ds = TokenMemmapMap(metas, label_map)
        elif mode == "mean":
            metas = [p for p in glob.glob(f"{data_path}/token/MEAN/*_pcmean_meta.npz")]
            full_ds = MeanMemmapMap(metas, label_map)
        elif mode == "pca":
            metas = glob.glob(f"{data_path}/token/PCA/*_pc{in_dim}_meta.npz")
            full_ds = PcaMemmapMap(metas, label_map, k=in_dim)
        mp = next(Path(f"{data_path}/token").glob("*_meta.npz"))
        Lg = int(np.load(mp, allow_pickle=True)["shape"][1])
        per_gene_len.append(Lg)
        gene_names.append(gene)
    
    # assert len(full_ds) == len([k for k in label_map if k in full_ds.ids]), "label_map and dataset size mismatch"

    print("label_map size:", len(label_map))
    print("full_ds size:", len(full_ds))

    return full_ds, label_map, gene_names, per_gene_len



def train_token_split(
    gene, drug, mode, in_dim, batch_size, n_epochs, lr, freeze_bias_frac,
    out_root, train_ds, val_ds, per_gene_len, gene_names, compute_shap=False
):
    labels_arr = np.array([train_ds[i][1] for i in range(len(train_ds))])
    nR = labels_arr.sum()
    nS = len(labels_arr) - nR
    pos_weight = torch.tensor(nS / (nR + 1e-8), dtype=torch.float32).to(device)
    freeze_ep = max(1, int(n_epochs * freeze_bias_frac))

    probe_n = min(100, len(train_ds))
    L_PAD = max(train_ds[i][0].shape[1] for i in range(probe_n))

    tr_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                       collate_fn=lambda b: pad_collate(b, L_PAD))
    va_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                       collate_fn=lambda b: pad_collate(b, L_PAD))

    stem_out = 64 if in_dim == 320 else 32
    model = ProteinCNN1x1(seq_len=L_PAD, in_dim=in_dim, stem_out=stem_out).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    with torch.no_grad():
        pR = nR / (nR + nS + 1e-8)
        model.fc_out.bias.fill_(math.log(pR / (1 - pR)))
    model.fc_out.bias.requires_grad = False

    hist = []
    for ep in range(1, n_epochs + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
            del xb, yb
            torch.cuda.empty_cache()
            gc.collect()

        if ep == freeze_ep:
            model.fc_out.bias.requires_grad = True

        model.eval()
        prob, yt = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                prob.append(torch.sigmoid(model(xb.to(device))).cpu())
                yt.append(yb)
                del xb, yb
                torch.cuda.empty_cache()
                gc.collect()

        prob = np.concatenate([p.numpy() for p in prob])
        yt = np.concatenate([y.numpy() for y in yt])
        auc = roc_auc_score(yt, prob)
        acc = ((prob > 0.5) == yt).mean()
        print(f"ep {ep:02d}/{n_epochs}  val_auc={auc:.3f}  val_acc={acc:.3f}")
        hist.append({"epoch": ep, "val_auc": float(auc), "val_acc": float(acc)})

    Path(out_root).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{out_root}/{drug}_model.pt")
    pd.DataFrame(hist).to_csv(f"{out_root}/{drug}_auc_history.csv", index=False)

    if compute_shap:
        shap_df = shap_per_residue(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            background_size=100,
            explain_samples=200,
            per_gene_lengths=per_gene_len,
            gene_names=gene_names,
            device=device,
        )
        # out_path = out_root/"interpretability"
        # out_root.mkdir(exist_ok=True)
        shap_df.to_pickle(Path(out_root) / f"{drug}_dim{in_dim}_shap_per_residue.pkl", protocol=4)
        print("SHAP saved:", Path(out_root) / f"{drug}_dim{in_dim}_shap_per_residue.pkl")

    return model, train_ds, val_ds, pd.DataFrame(hist)



from sklearn.model_selection import StratifiedKFold

def train_token_cv(
    gene, drug, mode="full", in_dim=320, batch_size=32, n_epochs=20,
    lr=5e-4, freeze_bias_frac=0.25, out_root="data/latest/cross_val",
    n_folds=5, seed=42, compute_shap=False
):
    full_ds, label_map, gene_names, per_gene_len = load_dataset_for_cv(gene, drug, mode, in_dim)
    # labels_arr = np.fromiter(label_map.values(), dtype=np.int32)
    labels_arr = np.array([full_ds[i][1] for i in range(len(full_ds))])

    all_histories = []
    all_fold_aucs = []
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # indices = np.arange(len(labels_arr))
    indices = np.arange(len(full_ds))

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, labels_arr), 1):
        print(f"\n===== Fold {fold}/{n_folds} =====")
        tr_subset = torch.utils.data.Subset(full_ds, train_idx)
        va_subset = torch.utils.data.Subset(full_ds, val_idx)
        print("len(full_ds):", len(full_ds))
        print("len(train_idx):", len(train_idx))
        print("max index in train_idx:", max(train_idx))


        model, tr_ds, va_ds, hist = train_token_split(
            gene=gene,
            drug=drug,
            mode=mode,
            in_dim=in_dim,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            freeze_bias_frac=freeze_bias_frac,
            out_root=f"{out_root}/{drug}/{mode}/fold_{fold}",
            train_ds=tr_subset,
            val_ds=va_subset,
            per_gene_len=per_gene_len,
            gene_names=gene_names,
            compute_shap=compute_shap,
        )

        auc_last = hist["val_auc"].values[-1]
        all_fold_aucs.append(auc_last)
        all_histories.append(hist.assign(fold=fold))

    all_hist_df = pd.concat(all_histories)
    cv_summary = pd.DataFrame({
        "fold": list(range(1, n_folds + 1)),
        "val_auc": all_fold_aucs
    })
    print("\n=== Cross-validation results ===")
    print(cv_summary)
    print(f"Mean AUC: {cv_summary.val_auc.mean():.3f} ± {cv_summary.val_auc.std():.3f}")
    return all_hist_df, cv_summary



# ------------------------------------------------------------------
# 0. Which experiments do you want to run?
# ------------------------------------------------------------------
TODO = [
    # ("streptomycin", 1, "mean"),
    # ("streptomycin", 1, "pca"),
    # ("streptomycin", 10, "pca"),
    # ("streptomycin", 320, "full"),
    # ("amikacin", 10, "pca"),
    # ("amikacin", 320, "full"),
    # ("capreomycin", 10, "pca"),
    # ("capreomycin", 320, "full"),
    # ("rifampicin", 10, "pca"),
    # ("rifampicin", 320, "full"),
    # ("pyrazinamide", 10, "pca"),
    # ("pyrazinamide", 320, "full"),
    # ("ethambutol", 10, "pca"),
    # ("ethambutol", 320, "full"),
    # ("isoniazid", 10, "pca"),
    # ("isoniazid", 320, "full"),
    # ("ethionamide", 10, "pca"),
    # ("ethionamide", 320, "full"),
    # ("moxifloxacin", 10, "pca"),
    # ("moxifloxacin", 320, "full"),
    ("levofloxacin", 10, "pca"),
    ("levofloxacin", 320, "full")
    # add more (drug, in_dim, mode) here
]

# # ------------------------------------------------------------------
# # 1. Run 5-fold cross-validation
# # ------------------------------------------------------------------
# results = {}  # store results
# for drug, in_dim, tag in TODO:
#     print(f"\n=== {drug}  |  {tag}  ===============================")
#     all_hist_df, cv_summary = train_token_cv(
#         gene=None,                 # ← kept for compatibility
#         drug=drug,
#         mode=tag,
#         in_dim=in_dim,
#         batch_size=32,
#         n_epochs=20,
#         lr=5e-4,
#         freeze_bias_frac=0.25,
#         out_root="data/latest/cross_val",  # separate from non-CV runs
#         n_folds=5,
#         compute_shap=True,                # optionally set to True per fold
#     )
#     results[(drug, in_dim, tag)] = {
#         "history": all_hist_df,
#         "summary": cv_summary
#     }


 
# all_summaries = []
# for k, v in results.items():
#     drug, in_dim, tag = k
#     summary_df = v["summary"].copy()
#     summary_df["drug"] = drug
#     summary_df["in_dim"] = in_dim
#     summary_df["mode"] = tag
#     all_summaries.append(summary_df)

# df_summary = pd.concat(all_summaries, ignore_index=True)
# df_summary.to_csv("data/latest/cross_val_summary.csv", index=False)
# print("\n=== Cross-validation summary saved to data/latest/cross_val_summary.csv ===")
# print(df_summary)





## eval preds

import numpy as np, pandas as pd, torch, gc, os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# --- simple fixed-length collate (pads/truncates to L_PAD) ---
def _pad_to_len(x: torch.Tensor, L_PAD: int):
    # x: (C, Lx) -> (C, L_PAD)
    C, Lx = x.shape
    if Lx >= L_PAD:
        return x[:, :L_PAD]
    pad = torch.zeros((C, L_PAD - Lx), dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

def _collate_fixed(L_PAD: int):
    def _fn(batch):
        xs, ys = zip(*batch)
        xs = [ _pad_to_len(x, L_PAD) for x in xs ]
        return torch.stack(xs, 0), torch.stack(ys, 0)
    return _fn

# --- evaluate a subset with a fixed L_PAD ---
def _eval_subset(model, subset, batch_size, device, L_PAD):
    ld = DataLoader(subset, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=False,
                    collate_fn=_collate_fixed(L_PAD))
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in ld:
            probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel())
            labels.append(yb.numpy().ravel())
    prob = np.concatenate(probs) if probs else np.array([])
    y    = np.concatenate(labels).astype(int) if labels else np.array([], dtype=int)
    return prob, y

# --- main: regenerate per-fold predictions from saved checkpoints ---
def regenerate_esm_fold_preds(
    gene, drug, mode, in_dim,
    out_root="data/latest/cross_val/esm_cv_sig",   # base folder that contains <drug>/<mode>/fold_k
    n_folds=5, seed=42, batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Rebuilds the dataset, replays the same StratifiedKFold, loads each fold's saved
    checkpoint from: <out_root>/<drug>/<mode>/fold_<k>/{drug}_model.pt,
    runs inference on the fold's validation subset, and writes:
       <out_root>/<drug>/<mode>/fold_<k>/val_preds.csv  (prob,label)
    """
    # 1) rebuild dataset exactly as during training (you already have this)
    full_ds, label_map, gene_names, per_gene_len = load_dataset_for_cv(gene, drug, mode, in_dim)

    # labels must come from the dataset to ensure alignment
    labels_arr = np.array([full_ds[i][1] for i in range(len(full_ds))], dtype=np.int32)
    # assert len(labels_arr) == len(full_ds), "dataset/labels mismatch"

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(full_ds))

    all_probs, all_labels = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, labels_arr), 1):
        print(f"\n===== {drug} | {mode} | fold {fold}/{n_folds} =====")
        tr_subset = torch.utils.data.Subset(full_ds, train_idx)
        va_subset = torch.utils.data.Subset(full_ds, val_idx)

        # L_PAD = max length in the TRAIN subset for this fold (matches training)
        L_PAD = max(tr_subset[i][0].shape[1] for i in range(len(tr_subset)))

        # recreate model exactly
        stem_out = 64 if in_dim == 320 else 32
        model = ProteinCNN1x1(seq_len=L_PAD, in_dim=in_dim, stem_out=stem_out).to(device)

        # checkpoint path for this fold
        fold_dir = Path(out_root) / drug / mode / f"fold_{fold}"
        ckpt = resolve_checkpoint_path(fold_dir, drug, fold)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        # inference on held-out val
        prob, y = _eval_subset(model, va_subset, batch_size, device, L_PAD)

        # save
        preds_df = pd.DataFrame({"prob": prob, "label": y})
        out_csv = fold_dir / "val_preds.csv"
        named_out_csv = fold_dir / f"{drug}_fold{fold}_preds.csv"
        preds_df.to_csv(out_csv, index=False)
        preds_df.to_csv(named_out_csv, index=False)
        print(f"saved {out_csv}")
        print(f"saved {named_out_csv}")

        # (optional) show fold AUC
        if len(np.unique(y)) == 2 and len(y) == len(prob) and len(y) > 0:
            auc = roc_auc_score(y, prob)
            print(f"AUC = {auc:.3f}")
        else:
            print("AUC = NA (single-class or empty fold)")

        all_probs.append(prob); all_labels.append(y)

        del model; gc.collect(); torch.cuda.empty_cache()

    # pooled AUC
    if all_probs and all_labels:
        P = np.concatenate(all_probs); Y = np.concatenate(all_labels)
        if len(P) and len(np.unique(Y)) == 2:
            pooled_auc = roc_auc_score(Y, P)
            print(f"\nPooled val AUC across folds: {pooled_auc:.3f}")
        else:
            print("\nPooled val AUC: NA")


# # FULL-320
# regenerate_esm_fold_preds(
#     gene=None, drug="capreomycin", mode="full", in_dim=320,
#     out_root="data/latest/cross_val/esm_cv_sig", n_folds=5, seed=42, batch_size=32
# )

# # PCA-10
# regenerate_esm_fold_preds(
#     gene=None, drug="capreomycin", mode="pca", in_dim=10,
#     out_root="data/latest/cross_val/esm_cv_sig", n_folds=5, seed=42, batch_size=32
# )




DRUG_LIST = ['rifampicin','streptomycin','isoniazid','pyrazinamide',
             'ethionamide','amikacin','capreomycin','moxifloxacin', 'levofloxacin','ethambutol']


def main():
    for drug_name in DRUG_LIST:
        if drug_name not in DRUG2GENES:
            print(f"[skip] unknown drug: {drug_name}")
            continue
        regenerate_esm_fold_preds(
            gene=None, drug=drug_name, mode="full", in_dim=320,
            out_root="data/latest/cross_val/esm_cv_sig", n_folds=5, seed=42, batch_size=32
        )

        regenerate_esm_fold_preds(
            gene=None, drug=drug_name, mode="pca", in_dim=10,
            out_root="data/latest/cross_val/esm_cv_sig", n_folds=5, seed=42, batch_size=32
        )


if __name__ == "__main__":
    main()
