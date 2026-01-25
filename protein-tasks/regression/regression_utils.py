from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import sys, argparse, os, numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, Dict

# ────────────────────────────────────────────────────────────
# 1)  COMMON HELPERS
# ────────────────────────────────────────────────────────────
CONFIG = {
    "FEATURE_DIR" : Path("../data/latest/feature_matrix_labels"),
    "WHO_CATALOG" : Path("../data/filtered_variants_output.csv"),
    "SEQ_META"    : Path("../data/catalog/protein_sequences.csv"),
    "PR_OUT_DIR"  : Path("../data/latest/results/interpretability/regression"),
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


# --- PR Helper Functions ---

def compute_residue_scores(coef: np.ndarray) -> np.ndarray:
    return np.abs(coef)
    
def load_catalog(catalog_path, allowed_confidences):
    catalog = pd.read_csv(catalog_path)
    catalog = catalog[
        (catalog["confidence"].isin(allowed_confidences)) &
        (catalog["intersectional"] == True)
    ].copy()
    catalog["aa_pos_0idx"] = catalog["aa_pos"].astype(int) - 1
 
    return catalog

def evaluate_topk_precision_recall(drug:str, gene_name:str, scores:np.ndarray, catalog_df:pd.DataFrame, k_vals=(10,), model:str="") -> list:
        
    variants_df = catalog_df[catalog_df["gene"].str.lower() == gene_name.lower()].copy()
    if variants_df.empty:
        print(f"Skipping {gene_name}: no intersectional variants found.")
        return []

    total_actual_positives = len(np.unique(variants_df["aa_pos_0idx"]))
    print(f"Total confirmed resistance positions for {gene_name}: {total_actual_positives}")

    imp_df = pd.DataFrame({"Residue_Position": np.arange(len(scores)), "Importance": scores})
    imp_df_sorted = imp_df.sort_values("Importance", ascending=False)
    

    rows = []
    for k in k_vals:
        top_k = imp_df_sorted.head(k)
        # top_k = imp_df.nlargest(int(np.ceil(len(imp_df) * (k / 100))), "Importance")

        important_positions = set(top_k["Residue_Position"])

        true_positions = set(variants_df["aa_pos_0idx"])
        
        true_positives = len(true_positions.intersection(important_positions))
        total_predictions = len(important_positions) #k

        precision = true_positives / total_predictions if total_predictions > 0 else 0
        recall = true_positives / total_actual_positives if total_actual_positives > 0 else 0
        # f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        assert precision <= 1.0, f"Precision > 1 for {gene_name} at k={k}"

        matched_df = variants_df[variants_df["aa_pos_0idx"].isin(important_positions)]

        identified_variants = matched_df.drop_duplicates("aa_pos_0idx")["variant"].tolist()
        
        # print(f"Top residues from model {model} for top-k {k}")
        # print(top_k)

        # print("Known resistance positions from WHO (1-indexed):")
        # print(sorted(set(variants_df["aa_pos_0idx"])))


        rows.append({
            "drug": drug, "gene": gene_name, "model": model, "k": k,
            "Total_Resistance_Positions": total_actual_positives,
            "TP": true_positives,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "identified_variants": ", ".join(identified_variants) if identified_variants else "None"
        })
    return rows



def encode_labels(y):
    le = LabelEncoder()
    return le.fit_transform(y)


def load_feature_matrix_and_labels(drug_name: str):
    """
    Load a design matrix and label vector for *either* a single-gene or
    a pre-merged multi-gene drug.

    • For single-gene drugs it automatically strips the mutation-flag
      column ( column-0 and full of 0/1).
    • For multi-gene drugs it leaves the matrix untouched.
    """
    mat_f = CONFIG["FEATURE_DIR"] / f"{drug_name.upper()}_feature_matrix.npy"
    lab_f = CONFIG["FEATURE_DIR"]/ f"{drug_name.upper()}_labels.npy"

    if not (mat_f.exists() and lab_f.exists()):
        raise FileNotFoundError(
            f"Expected files not found:\n  {mat_f}\n  {lab_f}"
        )

    X = np.load(mat_f)
    y = np.load(lab_f, allow_pickle=True)

    # ── drop mutation flag IFF this is a single-gene drug ──────────
    if len(DRUG2GENES[drug_name]) == 1:
        # extra guard: be sure the first column really *is* a flag
        X = X[:, 1:]
    print(f"{drug_name}: X shape = {X.shape}")
    return X, y

def gene_slices(drug:str, n_cols:int):
    """Return {gene:(start,end)} based on reference lengths."""
    ref = pd.read_csv(CONFIG["SEQ_META"])
    lens={ g:len(ref.loc[ref["gene"]==g,"protein_sequence"].values[0])
           for g in DRUG2GENES[drug]}
    gene_slices,cursor={},0
    for g in DRUG2GENES[drug]:
        L=lens[g]; gene_slices[g]=(cursor,cursor+L); cursor+=L
    assert cursor==n_cols, f"{drug}: expected {cursor}, got {n_cols}"
    return gene_slices