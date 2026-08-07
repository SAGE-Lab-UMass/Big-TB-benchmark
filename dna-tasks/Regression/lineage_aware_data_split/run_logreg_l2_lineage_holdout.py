import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
REGRESSION_DIR = THIS_DIR.parent
MODEL_TRAINING_DIR = REGRESSION_DIR / "model_training"
DATA_ROOT = REGRESSION_DIR.parents[1]
LINEAGE_CSV = DATA_ROOT / "BIG_TB_isolates_with_lineages.csv"

sys.path.insert(0, str(MODEL_TRAINING_DIR))

from parameters.locus_order import DRUG_TO_LOCI  # noqa: E402

MAJOR_LINEAGES = ("1", "2", "3", "4")
DEFAULT_MIN_CLASS_COUNT = 50
LOGREG_CS = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]


def _to_binary_labels(y: pd.Series) -> pd.Series:
    """Normalize labels to {0,1} where 0=R and 1=S."""
    if y.dtype.kind in {"i", "u", "b", "f"}:
        vals = pd.to_numeric(y, errors="coerce")
        if vals.isna().any():
            raise ValueError("Numeric phenotype column has non-numeric values")
        unique_vals = set(vals.unique().tolist())
        if unique_vals.issubset({0, 1}):
            return vals.astype(int)

    y_str = y.astype(str).str.strip().str.upper()
    mapped = y_str.map({"R": 0, "S": 1})
    if mapped.isna().any():
        bad = sorted(y_str[mapped.isna()].unique().tolist())
        raise ValueError(f"Unsupported phenotype labels: {bad}. Expected 0/1 or R/S")
    return mapped.astype(int)


def _normalize_lineage_series(lineage: pd.Series) -> pd.Series:
    """Normalize lineage labels so exact held-out lineage matching is stable."""
    s = lineage.astype("string")
    s = s.str.strip()
    s = s.mask(s.str.lower().isin(["nan", "none", ""]))

    def _normalize_one(x):
        if pd.isna(x):
            return pd.NA
        txt = str(x).strip()
        if txt == "":
            return pd.NA
        # Keep ambiguous labels (e.g. "1,4") intact, but canonicalize simple
        # numeric values so comparisons against "1"/"2"/"3"/"4" are reliable.
        if "," not in txt:
            try:
                val = float(txt)
                if float(val).is_integer():
                    return str(int(val))
            except Exception:
                pass
        return txt

    return s.apply(_normalize_one).astype("string")


def _resolve_lineages(df: pd.DataFrame) -> pd.Series:
    """Resolve isolate lineage from existing column or the canonical lineage CSV."""
    if "Lineage" in df.columns:
        return _normalize_lineage_series(df["Lineage"])

    if not LINEAGE_CSV.exists():
        raise FileNotFoundError(
            f"Lineage information missing: no 'Lineage' column in input and lineage CSV not found at {LINEAGE_CSV}"
        )

    lineage_df = pd.read_csv(LINEAGE_CSV, usecols=["ROLLINGDB_ID", "Lineage"])
    lineage_df["ROLLINGDB_ID"] = lineage_df["ROLLINGDB_ID"].astype(str)
    lineage_df["Lineage"] = lineage_df["Lineage"].astype("string")

    id_candidates = [
        "ROLLINGDB_ID",
        "rollingdb_id",
        "isolate",
        "Isolate",
        "Filename",
        "filename",
        "isolate_id",
        "Isolate_ID",
    ]
    join_col = None
    for col in id_candidates:
        if col in df.columns:
            join_col = col
            break

    if join_col is not None:
        merged = df[[join_col]].astype(str).merge(
            lineage_df,
            left_on=join_col,
            right_on="ROLLINGDB_ID",
            how="left",
        )
        return _normalize_lineage_series(merged["Lineage"])

    # Fallback: many prepared tables keep isolate IDs as the index.
    idx_df = pd.DataFrame({"ROW_ID": df.index.astype(str)})
    merged = idx_df.merge(lineage_df, left_on="ROW_ID", right_on="ROLLINGDB_ID", how="left")
    return _normalize_lineage_series(merged["Lineage"])


def _split_stats(y: pd.Series) -> tuple[int, int]:
    return int((y == 0).sum()), int((y == 1).sum())


def _write_aggregate_summary(drug_output_dir: Path) -> None:
    rows = []
    for heldout in MAJOR_LINEAGES:
        summary_path = drug_output_dir / f"heldout_lineage_{heldout}" / "test_set_accuracy.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            rows.append(df)
    if not rows:
        return
    all_df = pd.concat(rows, ignore_index=True)
    sort_cols = ["heldout_lineage"]
    if "model" in all_df.columns:
        sort_cols.append("model")
    all_df = all_df.sort_values(sort_cols).reset_index(drop=True)
    all_df.to_csv(drug_output_dir / "all_lineage_summary.csv", index=False)


def _validate_splits(drug_output_dir: Path) -> None:
    """Validate that split counts in summary files match split_manifest files."""
    for heldout in MAJOR_LINEAGES:
        lineage_dir = drug_output_dir / f"heldout_lineage_{heldout}"
        summary_path = lineage_dir / "test_set_accuracy.csv"
        manifest_path = lineage_dir / "split_manifest.csv"

        if not summary_path.exists() or not manifest_path.exists():
            continue

        summary_df = pd.read_csv(summary_path)
        manifest_df = pd.read_csv(manifest_path)

        if summary_df.empty or manifest_df.empty:
            continue

        summary_row = summary_df.iloc[0]
        manifest_test_count = int((manifest_df["split"] == "test").sum())
        manifest_train_count = int((manifest_df["split"] == "train").sum())

        summary_n = int(summary_row.get("N", 0))
        summary_train_n = int(summary_row.get("train_N", 0))

        # Warn if there's a mismatch
        if summary_n != manifest_test_count or summary_train_n != manifest_train_count:
            print(
                f"[warn] {drug_output_dir.name} heldout_lineage_{heldout}: "
                f"summary N={summary_n} (should be {manifest_test_count}), "
                f"train_N={summary_train_n} (should be {manifest_train_count})"
            )


def run_for_lineage(
    drug: str,
    heldout_lineage: str,
    kwargs: dict,
    input_data_df: pd.DataFrame,
    genotype_columns: list[str],
    min_class_count: int,
) -> None:
    lineage = input_data_df["_resolved_lineage"].astype("string")
    phenotype = _to_binary_labels(input_data_df[drug])

    annotated_mask = lineage.notna()
    lineage_annotated_df = input_data_df.loc[annotated_mask].copy()
    lineage_annotated_df["_resolved_lineage"] = lineage.loc[annotated_mask].astype(str)
    lineage_annotated_df["_binary_pheno"] = phenotype.loc[annotated_mask].astype(int)

    test_mask = lineage_annotated_df["_resolved_lineage"] == str(heldout_lineage)
    train_mask = ~test_mask

    train_df = lineage_annotated_df.loc[train_mask].copy()
    test_df = lineage_annotated_df.loc[test_mask].copy()

    train_r, train_s = _split_stats(train_df["_binary_pheno"])
    test_r, test_s = _split_stats(test_df["_binary_pheno"])

    lineage_output_dir = Path(kwargs["output_dir"]) / drug / f"heldout_lineage_{heldout_lineage}"
    lineage_output_dir.mkdir(parents=True, exist_ok=True)
    saved_models_dir = lineage_output_dir / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    legacy_xval_path = lineage_output_dir / "XVal_accuracy.csv"
    if legacy_xval_path.exists():
        legacy_xval_path.unlink()
    for legacy_model in ["GridSearchCV.model", "LogisticRegression_bestC.model"]:
        legacy_model_path = saved_models_dir / legacy_model
        if legacy_model_path.exists():
            legacy_model_path.unlink()

    split_manifest = pd.DataFrame(
        {
            "row_id": lineage_annotated_df.index.astype(str),
            "Lineage": lineage_annotated_df["_resolved_lineage"].astype(str),
            "label": lineage_annotated_df["_binary_pheno"].astype(int),
            "split": np.where(test_mask.to_numpy(), "test", "train"),
        }
    )
    split_manifest.to_csv(lineage_output_dir / "split_manifest.csv", index=False)

    base_summary = {
        "drug": drug,
        "heldout_lineage": str(heldout_lineage),
        "model": "logreg",
        "N": int(test_df.shape[0]),
        "N_S": int(test_s),
        "N_R": int(test_r),
        "AUC": np.nan,
        "acc": np.nan,
        "best_C": np.nan,
        "train_N": int(train_df.shape[0]),
        "train_N_S": int(train_s),
        "train_N_R": int(train_r),
        "feasible": bool(min(train_r, train_s, test_r, test_s) >= min_class_count),
        "min_class_count": int(min_class_count),
    }

    # Persist split counts IMMEDIATELY before model fitting so summary N values are never stale zeros.
    # This is critical: write summary ALWAYS, even if split is empty or infeasible, so the summary
    # reflects the actual data split and never shows stale N=0 values from a previous failed run.
    pd.DataFrame([base_summary]).to_csv(lineage_output_dir / "test_set_accuracy.csv", index=False)

    if min(train_r, train_s, test_r, test_s) < min_class_count:
        print(
            f"[skip] {drug} held-out lineage {heldout_lineage}: "
            f"train(R/S)=({train_r}/{train_s}), test(R/S)=({test_r}/{test_s})"
        )
        return

    print(
        f"[split] {drug} held-out lineage {heldout_lineage}: "
        f"train={train_df.shape[0]} test={test_df.shape[0]} "
        f"train(R/S)=({train_r}/{train_s}) test(R/S)=({test_r}/{test_s})"
    )

    X_train = train_df[genotype_columns]
    y_train = train_df["_binary_pheno"]

    clf = LogisticRegressionCV(
        Cs=LOGREG_CS,
        cv=5,
        scoring="roc_auc",
        max_iter=int(kwargs["max_iterations"]),
        penalty=kwargs["regularization"],
        class_weight="balanced",
        solver="liblinear",
        refit=True,
    )
    print(f"[fit] {drug} held-out lineage {heldout_lineage}: fitting LogisticRegressionCV")
    clf.fit(X_train, y_train)

    joblib.dump(clf, saved_models_dir / "LogisticRegressionCV.model")

    X_test = test_df[genotype_columns]
    y_test = test_df["_binary_pheno"]

    y_pred = clf.predict_proba(X_test.values)[:, 1]
    test_auc = np.nan
    if len(np.unique(y_test.values)) > 1:
        test_auc = roc_auc_score(y_test.values, y_pred)
    test_acc = accuracy_score(y_test.values, (y_pred >= 0.5).astype(int))

    test_summary = pd.DataFrame(
        [
            {
                **base_summary,
                "AUC": test_auc,
                "acc": test_acc,
                "best_C": float(clf.C_[0]),
                "feasible": True,
            }
        ]
    )
    test_summary.to_csv(lineage_output_dir / "test_set_accuracy.csv", index=False)

    preds_df = pd.DataFrame(
        {
            "row_id": test_df.index.astype(str),
            "Lineage": test_df["_resolved_lineage"].astype(str),
            "y_true": y_test.astype(int).to_numpy(),
            "y_pred": y_pred,
        }
    )
    preds_df.to_csv(lineage_output_dir / "test_predictions.csv", index=False)

    print(f"[ok] {drug} held-out lineage {heldout_lineage}: results in {lineage_output_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Usage: python run_logreg_l2_lineage_holdout.py <parameter_file>")

    _, input_file, *rest = sys.argv

    kwargs = yaml.safe_load(open(input_file, "r"))
    drug = kwargs["drug"]

    heldout_lineage = None
    min_class_count = DEFAULT_MIN_CLASS_COUNT
    for token in rest:
        if token.startswith("--heldout-lineage="):
            heldout_lineage = token.split("=", 1)[1]
        elif token.startswith("--min-class-count="):
            min_class_count = int(token.split("=", 1)[1])

    if heldout_lineage is not None and heldout_lineage not in MAJOR_LINEAGES:
        raise ValueError(f"heldout lineage must be one of {MAJOR_LINEAGES}")

    genotypes = pd.read_csv(kwargs["genotype_sites_file"], index_col=0)
    selected_loci = [f"/{gene}" for gene in DRUG_TO_LOCI[drug]]
    drug_genotypes = genotypes[genotypes["locus"].isin(selected_loci)]
    genotype_columns = [
        f"{locus}_{site}" for locus, site in zip(drug_genotypes.locus, drug_genotypes.sites)
    ]

    input_data_df_old = pd.read_csv(kwargs["input_data_file"], index_col=0, low_memory=False)
    input_data_df = input_data_df_old[input_data_df_old[drug].notna()].copy()

    missing_columns = sorted(set(genotype_columns) - set(input_data_df.columns))
    if missing_columns:
        raise ValueError(
            f"Input data is missing {len(missing_columns)} genotype feature columns for {drug}. "
            f"First missing columns: {missing_columns[:10]}"
        )

    input_data_df["_resolved_lineage"] = _resolve_lineages(input_data_df)

    heldouts = [heldout_lineage] if heldout_lineage else list(MAJOR_LINEAGES)
    for heldout in heldouts:
        run_for_lineage(
            drug=drug,
            heldout_lineage=heldout,
            kwargs=kwargs,
            input_data_df=input_data_df,
            genotype_columns=genotype_columns,
            min_class_count=min_class_count,
        )

    drug_output_dir = Path(kwargs["output_dir"]) / drug
    _validate_splits(drug_output_dir)
    _write_aggregate_summary(drug_output_dir)


if __name__ == "__main__":
    main()
