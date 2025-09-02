from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import norm

from regression_utils import *

# ────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ────────────────────────────────────────────────────────────
CONFIG = {
    "FEATURE_DIR" : Path("./data/latest/feature_matrix_labels"),
    "WHO_CATALOG" : Path("./data/filtered_variants_output.csv"),
    "SEQ_META"    : Path("./data/catalog/protein_sequences.csv"),
    "PR_OUT_DIR"  : Path("./data/latest/results/interpretability/regression"),
}

DRUG2GENES: Dict[str, list] = {
    # ── single-gene drugs ───────────────────────────────────
    "rifampicin"   : ["rpoB"],
    "pyrazinamide" : ["pncA"],
    "capreomycin"  : ["tlyA"],
    "amikacin"     : ["eis"],
    # ── multi-gene drugs (pre-merged matrices already saved)─
    "moxifloxacin" : ["gyrA", "gyrB"],
    "streptomycin" : ["rpsL", "gid"],
    "isoniazid"    : ["katG", "inhA"],
    "ethionamide"  : ["ethA", "ethR", "inhA"],
    "ethambutol"   : ["embA", "embB", "embC"],
    "levofloxacin" : ["gyrA", "gyrB"],
}

# ------------ utilities ------------
def _auc_safe(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def _bootstrap_auc_ci(y, p, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:  # skip single-class bootstrap sample
            continue
        boots.append(roc_auc_score(yy, pp))
    if not boots:
        return np.nan, (np.nan, np.nan)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(np.mean(boots)), (float(lo), float(hi))

def _extract_raw_coefs(pipeline):
    """
    Convert linear coefficients from standardized space back to raw feature space.
    Works for Pipeline([('scaler', StandardScaler), ('est', ...)]).

    Returns 1D numpy array of shape (n_features,)
    """
    scaler = pipeline.named_steps["scaler"]
    est    = pipeline.named_steps["est"]
    # LogisticRegressionCV: coef_.shape = (1, n_features)
    coef_std = est.coef_.ravel() if hasattr(est, "coef_") else est.coef_
    # Avoid division by zero: StandardScaler.scale_ can have zeros if a column is constant
    scale = np.where(scaler.scale_ != 0, scaler.scale_, 1.0)
    coef_raw = coef_std / scale
    return coef_raw

# ------------ main CV runner ------------
def run_models_cv(
    drug_name: str,
    k_vals=(10,),
    n_splits=5,
    seed=42,
    out_root=Path("data/latest/cross_val/regression_cv")
):
    """
    Outer 5-fold CV across LassoCV, RidgeCV, LogisticRegressionCV.
    Saves per-fold predictions & coefficients, per-gene residue scores,
    per-fold PR rows, and summary CSVs.
    """
    out_root = Path(out_root)
    out_drug = out_root / drug_name
    out_drug.mkdir(parents=True, exist_ok=True)

    # ---- load data & WHO ----
    X, y = load_feature_matrix_and_labels(drug_name)
    y_enc = encode_labels(y)
    slices = gene_slices(drug_name, X.shape[1])

    ALLOWED_CONF = ['1) Assoc w R', '2) Assoc w R - Interim']
    who_df = load_catalog(CONFIG["WHO_CATALOG"], ALLOWED_CONF)

    # ---- define models (with inner CV) ----
    models = {
        "lasso": Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("est", LassoCV(max_iter=10000, cv=5, random_state=seed,
                            alphas=[0.001, 0.01, 0.1, 1, 10, 100]))
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("est", RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10]))
        ]),
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("est", LogisticRegressionCV(
                cv=5, scoring="roc_auc", max_iter=5000,
                Cs=[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                class_weight="balanced",
                solver="liblinear",  # or 'saga' for large problems
                refit=True
            ))
        ]),
    }

    # ---- CV loop ----
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    per_model_fold_aucs = {m: [] for m in models}
    per_model_fold_rows = {m: [] for m in models}  # metrics rows
    per_model_pr_rows   = {m: [] for m in models}  # PR evaluation rows
    pooled_preds        = {m: [] for m in models}  # (y, p) concatenated

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)

            # Scores for AUC
            if name == "logreg":
                y_score = pipe.predict_proba(X_va)[:, 1]
            else:
                y_score = pipe.predict(X_va)  # continuous score OK for AUC

            auc = _auc_safe(y_va, y_score)
            acc = accuracy_score(y_va, (y_score >= 0.5).astype(int)) if not np.isnan(auc) else np.nan

            # save per-fold preds
            preds_df = pd.DataFrame({"prob": y_score, "label": y_va})
            preds_path = out_drug / f"{name}_fold{fold_idx}_preds.csv"
            preds_df.to_csv(preds_path, index=False)

            # extract raw-space coefs (for linear models)
            try:
                coef_raw = _extract_raw_coefs(pipe)
                np.save(out_drug / f"{name}_fold{fold_idx}_coefs.npy", coef_raw)

                # per-gene residue scores & PR@k
                for gene, (start, end) in slices.items():
                    scores = compute_residue_scores(coef_raw[start:end])
                    # per-gene full score table (optional; per fold)
                    full_scores = pd.DataFrame({
                        "Residue_Position": np.arange(start, end),
                        "Importance": scores,
                        "Model": name,
                        "Gene": gene,
                        "Fold": fold_idx
                    })
                    full_scores.to_csv(out_drug / f"full_residue_scores_{gene}_{drug_name}_{name}_fold{fold_idx}.csv",
                                       index=False)
                    per_model_pr_rows[name].extend(
                        evaluate_topk_precision_recall(drug_name, gene, scores, who_df, k_vals=k_vals, model=name)
                    )

            except Exception as e:
                # Some models might fail coef extraction; skip PR in that case
                print(f"[warn] coef extraction failed for {name}, fold {fold_idx}: {e}")

            per_model_fold_aucs[name].append(auc)
            per_model_fold_rows[name].append({
                "drug": drug_name, "model": name, "fold": fold_idx,
                "auc": auc, "acc": acc, "n_val": int(len(y_va))
            })
            pooled_preds[name].append((y_va, y_score))

    # ---- aggregate & save summaries ----
    summary_rows = []
    for name in models.keys():
        fold_df = pd.DataFrame(per_model_fold_rows[name])
        fold_df.to_csv(out_drug / f"{name}_cv_metrics.csv", index=False)

        aucs = np.array([a for a in per_model_fold_aucs[name] if not np.isnan(a)], dtype=float)
        mean_auc = float(np.nanmean(aucs)) if aucs.size else np.nan
        std_auc  = float(np.nanstd(aucs, ddof=1)) if aucs.size > 1 else np.nan

        # pooled concatenation for CI
        if pooled_preds[name]:
            y_all = np.concatenate([y for (y, _) in pooled_preds[name]])
            p_all = np.concatenate([p for (_, p) in pooled_preds[name]])
            pooled_mean, (ci_lo, ci_hi) = _bootstrap_auc_ci(y_all, p_all, n_boot=5000, alpha=0.05, seed=seed)
        else:
            pooled_mean, ci_lo, ci_hi = np.nan, np.nan, np.nan

        summary_rows.append({
            "drug": drug_name,
            "model": name,
            "folds": n_splits,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "pooled_auc": pooled_mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi
        })

        # save PR@k across folds for this model
        if per_model_pr_rows[name]:
            pr_df = pd.DataFrame(per_model_pr_rows[name])
            pr_df.to_csv(out_drug / f"PR_{drug_name}_{name}.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(out_drug / f"{drug_name}_cv_summary.csv", index=False)
    print(f"[OK] wrote {out_drug / f'{drug_name}_cv_summary.csv'}")

    return summary_rows


drug_list = ['rifampicin','pyrazinamide','capreomycin','amikacin',
             'isoniazid','ethionamide','streptomycin','ethambutol',
             'moxifloxacin','levofloxacin']

kvals = (1, 5, 10)
all_summaries = []
for drug in drug_list:
    if drug not in DRUG2GENES:
        print(f"[skip] {drug}: not in DRUG2GENES")
        continue
    rows = run_models_cv(drug_name=drug, k_vals=kvals, n_splits=5, seed=42,
                         out_root=Path("data/latest/cross_val/regression_cv"))
    all_summaries.extend(rows)

pd.DataFrame(all_summaries).to_csv(
    "data/latest/cross_val/regression_cv/all_drugs_cv_summary.csv", index=False
)
print("[DONE] wrote all_drugs_cv_summary.csv")
