# significance_testing_cnn_cv.py
# Stratified K-fold CV for CNN with optional per-fold SHAP,


import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import shap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score



from transformer import *
from transformer_utils import *

# ────────────────────────────────────────────────────────────────────
# Config (kept parallel to original file)
# ────────────────────────────────────────────────────────────────────
SEED            = 42
N_SPLITS        = 5
N_EPOCHS        = 20
BATCH_SIZE      = 32
LR              = 5e-4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# SHAP (per-fold)
COMPUTE_SHAP    = True
BG_SIZE         = 100
EXPL_SAMPLES    = 200

# Keep the DRUG2GENES mapping  already use elsewhere
DRUG2GENES = {
    "rifampicin"  : ["rpoB"],
    "pyrazinamide": ["pncA"],
    "capreomycin" : ["tlyA"],
    "amikacin"    : ["eis"],
    "moxifloxacin": ["gyrA","gyrB"],
    "levofloxacin": ["gyrA","gyrB"],
    "isoniazid"   : ["katG","inhA"],
    "streptomycin": ["rpsL","gid"],
    "ethambutol"  : ["embC","embA","embB"],
    "ethionamide" : ["ethA","ethR","inhA"],
}

# Output roots aligned with  current CV consumers
OUT_ROOT = Path("data/latest/cross_val/transformer_cv_sig")  
CURVE_DIR = OUT_ROOT                                  # per-fold curves/preds live here



# ────────────────────────────────────────────────────────────────────
# One fold of training/eval — mirrors  original training loop
# ────────────────────────────────────────────────────────────────────
def _train_one_fold(
    model: nn.Module,
    train_ds: Subset,
    val_ds: Subset,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, float]]:
    """Train on train_ds, evaluate on val_ds; returns (curve_df, probs, gold, metrics)"""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    train_labels = [y for _, y in train_ds]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    curve_rows = []
    for ep in range(1, n_epochs + 1):
        # train
        model.train(); running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X), y)
            loss.backward(); optimiser.step()
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / max(1, len(train_ds))

        # val
        model.eval(); preds, gold = [], []
        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X.to(device))
                preds.extend(torch.sigmoid(logits).cpu().numpy().ravel())
                gold.extend(y.numpy().ravel())
        preds = np.asarray(preds, dtype=float)
        gold  = np.asarray(gold, dtype=int)

        # AUC can be undefined if val fold is single-class; guard it.
        auc = float("nan")
        if np.unique(gold).size == 2:
            auc = float(roc_auc_score(gold, preds))

        curve_rows.append({"Epoch": ep, "Loss": float(epoch_loss), "AUC": auc})
        print(f"epoch {ep:02d} | loss {epoch_loss:.4f} | AUC {auc:.3f}" if np.isfinite(auc) else
              f"epoch {ep:02d} | loss {epoch_loss:.4f} | AUC NaN (single-class fold)")

    # threshold 0.5 for quick point metrics (not used for significance)
    thr = 0.5
    pred = (preds >= thr).astype(int)
    tp = int(((pred == 1) & (gold == 1)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())

    acc  = (tp + tn) / max(1, gold.size)
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    metrics = {"acc": acc, "sens": sens, "spec": spec, "auc": auc}

    return pd.DataFrame(curve_rows), preds, gold, metrics


# ────────────────────────────────────────────────────────────────────
# CV driver that mirrors `train_eval_model` semantics
# ────────────────────────────────────────────────────────────────────
def train_eval_model_cv(
    tag: str,                  # e.g., "moxifloxacin"
    df: pd.DataFrame,          # must have Protein_Sequence + Phenotype
    n_splits: int = N_SPLITS,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
    compute_shap: bool = COMPUTE_SHAP,
    background_size: int = BG_SIZE,
    explain_samples: int = EXPL_SAMPLES,
    per_gene_lengths: List[int] = None,
    gene_names: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      fold_df: per-fold metrics
      summary_df: 1-row pooled summary with bootstrap CI and p_vs_0.5
    """
    # 1) Build full dataset identical to original
    full_ds = ProteinDataset(
        df["Protein_Sequence"].tolist(),
        (df["Phenotype"] == "R").astype(int).tolist()
    )
    labels = np.asarray(full_ds.labels, dtype=int)

    # 2) K-fold splitter with class-count guard
    class_counts = np.bincount(labels, minlength=2)
    max_splits = int(class_counts.min()) if class_counts.min() > 0 else 1
    if max_splits < 2:
        raise ValueError(
            f"{tag}: Not enough minority-class samples for CV "
            f"(counts={class_counts.tolist()}); need at least 2."
        )
    if n_splits > max_splits:
        print(f"[warn] {tag}: reducing n_splits from {n_splits} → {max_splits} "
              f"to satisfy per-fold class balance.")
        n_splits = max_splits

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # 3) Loop folds
    fold_rows: List[Dict] = []
    all_fold_aucs: List[float] = []
    pooled_gold, pooled_pred = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(full_ds)), labels), start=1):
        print(f"\n=== {tag} | Fold {fold}/{n_splits} ===")
        train_ds = Subset(full_ds, tr_idx)
        val_ds   = Subset(full_ds, va_idx)

        # model init mirrors original (NO L_PAD)
        model = ProteinTransformer().to(device)

        # train one fold
        curve_df, probs, gold, metrics = _train_one_fold(
            model, train_ds, val_ds,
            n_epochs=n_epochs, lr=lr, batch_size=batch_size, device=device
        )

        # 3a) save per-fold artifacts
        CURVE_DIR.mkdir(parents=True, exist_ok=True)
        curve_df.to_csv(CURVE_DIR / f"{tag}_fold{fold}_training_curve.csv", index=False)

        pd.DataFrame({"prob": probs, "label": gold})\
          .to_csv(CURVE_DIR / f"{tag}_fold{fold}_preds.csv", index=False)

        # torch.save(model.state_dict(), CURVE_DIR / f"{tag}_fold{fold}_cnn.pt")
        torch.save(model.state_dict(), CURVE_DIR / f"{tag}_fold{fold}_transformer.pt")


        # 3b) optional SHAP per fold 
        if compute_shap:
            shap_df = shap_per_residue(
                model            = model,
                train_ds         = train_ds,
                val_ds           = val_ds,
                background_size  = background_size,
                explain_samples  = explain_samples,
                per_gene_lengths = per_gene_lengths,
                gene_names       = gene_names,
                device           = device,
            )
            # shap_df.to_pickle(CURVE_DIR / f"{tag}_fold{fold}_shap.pkl", protocol=4)
            shap_df.to_pickle(CURVE_DIR / f"{tag}_fold{fold}_shap_per_residue.pkl", protocol=4)


        # record fold metrics
        fold_rows.append({"drug": tag, "fold": fold, **metrics})
        if np.isfinite(metrics["auc"]):
            all_fold_aucs.append(float(metrics["auc"]))
        pooled_gold.append(gold)
        pooled_pred.append(probs)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(CURVE_DIR / f"{tag}_cv_metrics.csv", index=False)

    # 4) pooled bootstrap CI and simple significance vs 0.5
    concat_gold = np.concatenate(pooled_gold)
    concat_prob = np.concatenate(pooled_pred)

    # Bootstrap CI only if pooled labels contain both classes
    if np.unique(concat_gold).size == 2:
        boot_mean, (ci_lo, ci_hi) = bootstrap_auc_ci(
            concat_gold, concat_prob, n_boot=5000, alpha=0.05, seed=SEED
        )
    else:
        boot_mean, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")

    # Per-fold significance (one-sample test on AUCs) only if we got valid AUCs
    if len(all_fold_aucs) >= 2:
        mean_auc, std_auc, pval = one_sample_test_vs_half(all_fold_aucs)
    else:
        mean_auc, std_auc, pval = float("nan"), float("nan"), float("nan")

    summary_df = pd.DataFrame([{
        "drug": tag,
        "folds": n_splits,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "pooled_boot_auc_mean": boot_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_vs_0.5": pval,
        "n_val_total": int(concat_gold.size)
    }])
    summary_df.to_csv(CURVE_DIR / f"{tag}_cv_summary.csv", index=False)

    print(f"[OK] {tag}: mean AUC={mean_auc:.3f}  pooled CI=({ci_lo:.3f}, {ci_hi:.3f})")
    return fold_df, summary_df


# ────────────────────────────────────────────────────────────────────
# Wrapper matching the shape of  original `run_cnn_for_drug`
# ────────────────────────────────────────────────────────────────────
def run_transformer_cv_for_drug(drug: str, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """
    Mirrors `run_transformer_for_drug`: builds df & per-gene lengths, then calls `train_eval_model_cv`.
    Returns the 1-row summary for aggregation 
    """
    if drug not in DRUG2GENES:
        raise KeyError(f"Unknown drug: {drug}")

    genes = DRUG2GENES[drug]
    print(f"\n=== {drug.upper()} ({', '.join(genes)}) ===")


    df = build_drug_df(drug)
    per_gene_lengths = [len(df[f"seq_{g}"].iloc[0]) for g in genes]
    print(f"{drug.upper()} | isolates={len(df)} | lengths={per_gene_lengths}")

    df = df[["Filename", "Protein_Sequence", "Phenotype"]]  # drop helpers like in original

    # Run CV with identical SHAP semantics
    _, summary_df = train_eval_model_cv(
        tag               = drug,
        df                = df,
        n_splits          = n_splits,
        n_epochs          = N_EPOCHS,
        lr                = LR,
        batch_size        = BATCH_SIZE,
        device            = DEVICE,
        compute_shap      = COMPUTE_SHAP,
        background_size   = BG_SIZE,
        explain_samples   = EXPL_SAMPLES,
        per_gene_lengths  = per_gene_lengths if len(genes) > 1 else None,
        gene_names        = genes if len(genes) > 1 else None,
    )
    return summary_df


# ────────────────────────────────────────────────────────────────────
# Main: run the full panel
# ────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    drug_list = [
        'isoniazid', 'rifampicin', 'pyrazinamide', 'capreomycin', 'amikacin',
         'ethionamide', 'streptomycin',
        'ethambutol', 'moxifloxacin', 'levofloxacin'
    ]

    summaries = []
    for d in drug_list:
        try:
            summaries.append(run_transformer_cv_for_drug(d, n_splits=N_SPLITS))
        except Exception as e:
            print(f"[skip] {d} → {e}")

    if summaries:
        pd.concat(summaries, ignore_index=True)\
          .to_csv(OUT_ROOT / "all_drugs_cv_summary.csv", index=False)
        print(f"[DONE] Wrote {OUT_ROOT / 'all_drugs_cv_summary.csv'}")


if __name__ == "__main__":
    main()
