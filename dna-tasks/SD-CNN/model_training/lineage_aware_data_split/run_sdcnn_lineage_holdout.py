#!/usr/bin/env python
# coding: utf-8
"""Lineage-aware SD-CNN training with leave-one-major-lineage-out splits.

This is a minimal, self-contained adaptation of SD-CNN single-drug training:
- keeps SD-CNN architecture and masked loss behavior
- replaces random split with held-out lineage split
- writes split manifests and training/eval outputs per held-out lineage
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sparse
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = THIS_DIR.parent
PARAMETERS_DIR = MODEL_TRAINING_DIR / "parameters"
DNA_TASKS_DIR = MODEL_TRAINING_DIR.parents[1]
DATA_CURATION_DIR = DNA_TASKS_DIR.parent
LINEAGE_CSV = DATA_CURATION_DIR / "BIG_TB_isolates_with_lineages.csv"

sys.path.insert(0, str(PARAMETERS_DIR))

from tb_cnn_codebase import (  # noqa: E402
    alpha_mat,
    create_X,
    get_threshold_val,
    load_combined_geno_pheno,
    make_geno_pheno_dataset,
    masked_multi_weighted_bce,
    masked_weighted_accuracy,
    rs_encoding_to_numeric,
)


MAJOR_LINEAGES = ("1", "2", "3", "4")


def _normalize_lineage_series(lineage: pd.Series) -> pd.Series:
    s = lineage.astype("string").str.strip()
    s = s.mask(s.str.lower().isin(["nan", "none", ""]))

    def _normalize_one(x):
        if pd.isna(x):
            return pd.NA
        txt = str(x).strip()
        if txt == "":
            return pd.NA
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
    if "Lineage" in df.columns:
        return _normalize_lineage_series(df["Lineage"])

    if not LINEAGE_CSV.exists():
        raise FileNotFoundError(
            f"Lineage information missing: no Lineage column and CSV not found at {LINEAGE_CSV}"
        )

    lineage_df = pd.read_csv(LINEAGE_CSV, usecols=["ROLLINGDB_ID", "Lineage"])
    lineage_df["ROLLINGDB_ID"] = lineage_df["ROLLINGDB_ID"].astype(str)

    id_candidates = [
        "ROLLINGDB_ID",
        "rollingdb_id",
        "accessions",
        "biosample",
        "isolate",
        "Isolate",
        "Filename",
        "filename",
        "isolate_id",
        "Isolate_ID",
        "New_ID",
    ]

    resolved = pd.Series(pd.NA, index=df.index, dtype="string")
    matched_any = False
    for col in id_candidates:
        if col not in df.columns:
            continue

        merged = df[[col]].astype(str).merge(
            lineage_df,
            left_on=col,
            right_on="ROLLINGDB_ID",
            how="left",
        )
        candidate = pd.Series(
            _normalize_lineage_series(merged["Lineage"]).to_numpy(),
            index=df.index,
            dtype="string",
        )
        fill_mask = resolved.isna() & candidate.notna()
        if fill_mask.any():
            resolved.loc[fill_mask] = candidate.loc[fill_mask]
            matched_any = True

    if matched_any:
        return resolved

    idx_df = pd.DataFrame({"ROW_ID": df.index.astype(str)})
    merged = idx_df.merge(lineage_df, left_on="ROW_ID", right_on="ROLLINGDB_ID", how="left")
    return pd.Series(
        _normalize_lineage_series(merged["Lineage"]).to_numpy(),
        index=df.index,
        dtype="string",
    )


def _get_model(x_shape: tuple[int, ...], filter_size: int) -> models.Model:
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            64,
            (5, filter_size),
            activation="relu",
            data_format="channels_last",
            input_shape=x_shape,
        )
    )
    model.add(layers.Conv2D(64, (1, 12), activation="relu"))
    model.add(layers.MaxPooling2D((1, 3)))
    model.add(layers.Conv2D(32, (1, 3), activation="relu"))
    model.add(layers.Conv2D(32, (1, 3), activation="relu"))
    model.add(layers.MaxPooling2D((1, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = Adam(learning_rate=np.exp(-9.0))
    model.compile(optimizer=opt, loss=masked_multi_weighted_bce, metrics=[masked_weighted_accuracy])
    return model


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, drug: str) -> pd.DataFrame:
    col_names = ["Algorithm", "Drug", "num_sensitive", "num_resistant", "AUC", "AUC_PR", "threshold", "spec", "sens"]
    non_missing = np.where(y_true != -1)[0]
    if len(non_missing) == 0:
        return pd.DataFrame([["SD-CNN", drug, 0, 0, np.nan, np.nan, threshold, np.nan, np.nan]], columns=col_names)

    yv = y_true[non_missing].astype(int)
    pv = y_pred[non_missing]
    num_sensitive = int(np.sum(yv == 1))
    num_resistant = int(np.sum(yv == 0))

    if num_sensitive == 0 or num_resistant == 0:
        return pd.DataFrame(
            [["SD-CNN", drug, num_sensitive, num_resistant, np.nan, np.nan, threshold, np.nan, np.nan]],
            columns=col_names,
        )

    auc = roc_auc_score(yv, pv)
    auc_pr = average_precision_score(1 - yv, 1 - pv)
    binary_pred = (pv > threshold).astype(int)
    spec = np.sum(np.logical_and(binary_pred == 1, yv == 1)) / num_sensitive
    sens = np.sum(np.logical_and(binary_pred == 0, yv == 0)) / num_resistant

    return pd.DataFrame([["SD-CNN", drug, num_sensitive, num_resistant, auc, auc_pr, threshold, spec, sens]], columns=col_names)


def _run_one_lineage(kwargs: dict, heldout_lineage: str, dry_run: bool, epochs_override: int | None) -> None:
    drug = kwargs["drug"]
    out_root = Path(kwargs["output_path"]) / f"heldout_lineage_{heldout_lineage}"
    out_root.mkdir(parents=True, exist_ok=True)

    parquet_file = kwargs["metadata_path"]
    h5_file = kwargs["h5_path"]
    if os.path.isfile(parquet_file) and os.path.isfile(h5_file):
        print("Found existing parquet/HDF5 files.")
    else:
        print("Creating genotype-phenotype dataset ...")
        make_geno_pheno_dataset(**kwargs)

    print("Loading combined geno+pheno DataFrame ...")
    df = load_combined_geno_pheno(**kwargs).copy()
    df["_row_id"] = df.index.astype(str)
    df["_resolved_lineage"] = _resolve_lineages(df)

    y_df, _ = rs_encoding_to_numeric(df, drug)
    y_all = y_df.values.astype(int).reshape(-1)

    valid_mask = (y_all != -1) & df["_resolved_lineage"].notna().to_numpy()
    df = df.loc[valid_mask].copy().reset_index(drop=True)
    y_all = y_all[valid_mask]

    lineage_vals = df["_resolved_lineage"].astype(str)
    test_mask = lineage_vals == str(heldout_lineage)
    train_mask = ~test_mask

    train_idx = np.where(train_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    train_sensitive = int(np.sum(y_train == 1)) if len(y_train) > 0 else 0
    train_resistant = int(np.sum(y_train == 0)) if len(y_train) > 0 else 0
    test_sensitive = int(np.sum(y_test == 1)) if len(y_test) > 0 else 0
    test_resistant = int(np.sum(y_test == 0)) if len(y_test) > 0 else 0

    split_summary = {
        "drug": drug,
        "heldout_lineage": str(heldout_lineage),
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "train_sensitive": train_sensitive,
        "train_resistant": train_resistant,
        "test_sensitive": test_sensitive,
        "test_resistant": test_resistant,
        "nonzero_train_and_test": bool(len(train_idx) > 0 and len(test_idx) > 0),
    }

    split_manifest = pd.DataFrame(
        {
            "row_id": df["_row_id"].astype(str),
            "lineage": lineage_vals.values,
            "y": y_all,
            "split": np.where(test_mask.values, "test", "train"),
        }
    )
    split_manifest.to_csv(out_root / "split_manifest.csv", index=False)
    pd.DataFrame([split_summary]).to_csv(out_root / "split_summary.csv", index=False)

    print(
        f"[split] {drug} heldout_lineage={heldout_lineage} "
        f"train={len(train_idx)} test={len(test_idx)} "
        f"train(S/R)=({train_sensitive}/{train_resistant}) test(S/R)=({test_sensitive}/{test_resistant})"
    )

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("[skip] split does not satisfy non-zero train/test requirement")
        return

    if dry_run:
        print("[dry-run] wrote split manifest/summary only")
        return

    x_sparse_path = kwargs.get("X_sparse_path", None)
    x_sparse = None
    if x_sparse_path and os.path.isfile(x_sparse_path):
        try:
            x_sparse = sparse.load_npz(x_sparse_path)
            if x_sparse.shape[0] != len(df):
                print("Cached X_sparse row count mismatch for this filtered frame; rebuilding X.")
                x_sparse = None
        except Exception:
            x_sparse = None

    if x_sparse is None:
        print("Creating X array ...")
        x_all = create_X(df, drug)
        x_sparse = sparse.COO(x_all)
        del x_all

    x_train = x_sparse[train_idx, :].todense()
    x_test = x_sparse[test_idx, :].todense()

    y_train_2d = y_train.reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)

    alpha_train = alpha_mat(y_train_2d, df.iloc[train_idx], weight=kwargs.get("weight_of_sensitive_class", 1.0), drug_name=drug)
    alpha_test = alpha_mat(y_test_2d, df.iloc[test_idx], weight=kwargs.get("weight_of_sensitive_class", 1.0), drug_name=drug)

    n_epochs = int(epochs_override if epochs_override is not None else kwargs["N_epochs"])
    model = _get_model(x_train.shape[1:], int(kwargs["filter_size"]))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        min_delta=1e-4,
        verbose=1,
    )

    print(f"Training SD-CNN: epochs={n_epochs}")
    history = model.fit(
        x_train,
        alpha_train,
        validation_data=(x_test, alpha_test),
        epochs=n_epochs,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(out_root / "history.csv", index=False)

    loss_delta = np.nan
    val_loss_delta = np.nan
    if "loss" in hist_df.columns and len(hist_df) > 1:
        loss_delta = float(hist_df["loss"].iloc[-1] - hist_df["loss"].iloc[0])
    if "val_loss" in hist_df.columns and len(hist_df) > 1:
        val_loss_delta = float(hist_df["val_loss"].iloc[-1] - hist_df["val_loss"].iloc[0])

    pd.DataFrame(
        [
            {
                "first_loss": float(hist_df["loss"].iloc[0]) if "loss" in hist_df.columns else np.nan,
                "last_loss": float(hist_df["loss"].iloc[-1]) if "loss" in hist_df.columns else np.nan,
                "loss_delta_last_minus_first": loss_delta,
                "first_val_loss": float(hist_df["val_loss"].iloc[0]) if "val_loss" in hist_df.columns else np.nan,
                "last_val_loss": float(hist_df["val_loss"].iloc[-1]) if "val_loss" in hist_df.columns else np.nan,
                "val_loss_delta_last_minus_first": val_loss_delta,
            }
        ]
    ).to_csv(out_root / "loss_trend_summary.csv", index=False)

    model_dir = out_root / "saved_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "sd-cnn_model_lineage_holdout.h5", include_optimizer=True)

    train_pred = np.squeeze(model.predict(x_train, verbose=0))
    test_pred = np.squeeze(model.predict(x_test, verbose=0))

    thresh = get_threshold_val(y_train.reshape(-1, 1), train_pred.reshape(-1, 1))
    threshold = float(thresh["threshold"])
    pd.DataFrame([thresh]).to_csv(out_root / "threshold_file.csv", index=False)

    train_metrics = _compute_metrics(y_train, train_pred, threshold, drug)
    test_metrics = _compute_metrics(y_test, test_pred, threshold, drug)
    train_metrics.to_csv(out_root / "training_set_drug_auc.csv", index=False)
    test_metrics.to_csv(out_root / "test_set_drug_auc.csv", index=False)

    pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": test_pred,
            "lineage": lineage_vals.iloc[test_idx].values,
            "row_id": df["_row_id"].iloc[test_idx].values,
        }
    ).to_csv(out_root / "test_predictions.csv", index=False)

    print(f"[ok] completed lineage holdout {heldout_lineage}. Outputs: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to SD-CNN YAML/TXT config")
    parser.add_argument("--heldout-lineage", default=None, choices=list(MAJOR_LINEAGES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs-override", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        kwargs = yaml.safe_load(f)

    seed = int(kwargs.get("random_seed", 42))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)
    for heldout in heldouts:
        _run_one_lineage(kwargs, str(heldout), args.dry_run, args.epochs_override)

    K.clear_session()


if __name__ == "__main__":
    main()
