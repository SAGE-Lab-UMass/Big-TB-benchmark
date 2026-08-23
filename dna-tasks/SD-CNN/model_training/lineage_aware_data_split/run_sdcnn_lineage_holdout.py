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
from tensorflow.keras.callbacks import Callback
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
DEFAULT_MIN_CLASS_COUNT = 50
DEFAULT_BATCH_SIZE = 100
DEFAULT_EARLY_STOPPING_MIN_EPOCHS = 20
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = 1e-3
DEFAULT_EARLY_STOPPING_SMOOTHING_WINDOW = 3


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

    # Prefer the canonical New_ID -> ROLLINGDB_ID join for SD-CNN tables.
    if "New_ID" in df.columns:
        merged = df[["New_ID"]].astype(str).merge(
            lineage_df,
            left_on="New_ID",
            right_on="ROLLINGDB_ID",
            how="left",
        )
        return pd.Series(
            _normalize_lineage_series(merged["Lineage"]).to_numpy(),
            index=df.index,
            dtype="string",
        )

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


class TrainingLossConvergenceCallback(Callback):
    """Stop on smoothed epoch-mean training loss without using validation/test data."""

    def __init__(
        self,
        checkpoint_path: Path,
        min_epochs: int,
        patience: int,
        min_relative_improvement: float,
        smoothing_window: int,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.min_relative_improvement = float(min_relative_improvement)
        self.smoothing_window = int(smoothing_window)
        if self.min_epochs < 1:
            raise ValueError("early_stopping_min_epochs must be >= 1")
        if self.patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.smoothing_window < 1:
            raise ValueError("early_stopping_smoothing_window must be >= 1")

        self.losses: list[float] = []
        self.records: list[dict[str, float | int]] = []
        self.best_smoothed_loss = np.inf
        self.best_epoch = 0
        self.patience_counter = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        if "loss" not in logs:
            raise RuntimeError("Training loss is missing from Keras epoch logs")

        epoch_num = epoch + 1
        mean_train_loss = float(logs["loss"])
        self.losses.append(mean_train_loss)

        window_losses = self.losses[-self.smoothing_window :]
        smoothed_train_loss = float(np.mean(window_losses))

        if np.isfinite(self.best_smoothed_loss):
            relative_improvement = (
                self.best_smoothed_loss - smoothed_train_loss
            ) / max(abs(self.best_smoothed_loss), 1e-12)
            meaningful_improvement = relative_improvement >= self.min_relative_improvement
        else:
            relative_improvement = np.nan
            meaningful_improvement = True

        if meaningful_improvement:
            self.best_smoothed_loss = smoothed_train_loss
            self.best_epoch = epoch_num
            self.patience_counter = 0
            self.model.save_weights(str(self.checkpoint_path))
        elif epoch_num >= self.min_epochs:
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        self.records.append(
            {
                "epoch": epoch_num,
                "mean_train_loss": mean_train_loss,
                "smoothed_train_loss": smoothed_train_loss,
                "best_smoothed_train_loss": self.best_smoothed_loss,
                "relative_improvement": relative_improvement,
                "patience_counter": self.patience_counter,
            }
        )
        print(
            "[early-stop] "
            f"epoch={epoch_num} "
            f"mean_train_loss={mean_train_loss:.8f} "
            f"smoothed_train_loss={smoothed_train_loss:.8f} "
            f"best_smoothed_train_loss={self.best_smoothed_loss:.8f} "
            f"relative_improvement={relative_improvement:.8g} "
            f"patience_counter={self.patience_counter}"
        )

        if epoch_num >= self.min_epochs and self.patience_counter >= self.patience:
            self.stopped_epoch = epoch_num
            self.model.stop_training = True
            print(
                f"Training converged at epoch {epoch_num}.\n"
                f"Smoothed training loss did not improve by >= {self.min_relative_improvement * 100:.3g}%\n"
                f"for {self.patience} consecutive epochs.\n"
                f"Best checkpoint was from epoch {self.best_epoch}."
            )


def _run_one_lineage(
    kwargs: dict,
    heldout_lineage: str,
    dry_run: bool,
    epochs_override: int | None,
    min_class_count: int,
    batch_size_override: int | None,
    early_stopping_min_epochs_override: int | None,
    early_stopping_patience_override: int | None,
    early_stopping_min_relative_improvement_override: float | None,
    early_stopping_smoothing_window_override: int | None,
) -> None:
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
    if "New_ID" not in df.columns:
        df["New_ID"] = df.index.astype(str)
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
        "feasible": bool(min(train_sensitive, train_resistant, test_sensitive, test_resistant) >= min_class_count),
        "min_class_count": int(min_class_count),
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

    if min(train_sensitive, train_resistant, test_sensitive, test_resistant) < min_class_count:
        print(
            f"[skip] split does not satisfy min_class_count={min_class_count}: "
            f"train(S/R)=({train_sensitive}/{train_resistant}) "
            f"test(S/R)=({test_sensitive}/{test_resistant})"
        )
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
    y_train_2d = y_train.reshape(-1, 1)

    alpha_train = alpha_mat(y_train_2d, df.iloc[train_idx], weight=kwargs.get("weight_of_sensitive_class", 1.0), drug_name=drug)

    n_epochs = int(epochs_override if epochs_override is not None else kwargs["N_epochs"])
    batch_size = int(batch_size_override if batch_size_override is not None else kwargs.get("batch_size", DEFAULT_BATCH_SIZE))
    early_stopping_min_epochs = int(
        early_stopping_min_epochs_override
        if early_stopping_min_epochs_override is not None
        else kwargs.get("early_stopping_min_epochs", DEFAULT_EARLY_STOPPING_MIN_EPOCHS)
    )
    early_stopping_patience = int(
        early_stopping_patience_override
        if early_stopping_patience_override is not None
        else kwargs.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
    )
    early_stopping_min_relative_improvement = float(
        early_stopping_min_relative_improvement_override
        if early_stopping_min_relative_improvement_override is not None
        else kwargs.get(
            "early_stopping_min_relative_improvement",
            DEFAULT_EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT,
        )
    )
    early_stopping_smoothing_window = int(
        early_stopping_smoothing_window_override
        if early_stopping_smoothing_window_override is not None
        else kwargs.get(
            "early_stopping_smoothing_window",
            DEFAULT_EARLY_STOPPING_SMOOTHING_WINDOW,
        )
    )
    model = _get_model(x_train.shape[1:], int(kwargs["filter_size"]))

    model_dir = out_root / "saved_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = model_dir / "best_training_loss.weights.h5"
    early_stop = TrainingLossConvergenceCallback(
        checkpoint_path=best_checkpoint_path,
        min_epochs=early_stopping_min_epochs,
        patience=early_stopping_patience,
        min_relative_improvement=early_stopping_min_relative_improvement,
        smoothing_window=early_stopping_smoothing_window,
    )

    print(
        f"Training SD-CNN: epochs={n_epochs} batch_size={batch_size} "
        f"early_stopping_min_epochs={early_stopping_min_epochs} "
        f"early_stopping_patience={early_stopping_patience} "
        f"early_stopping_min_relative_improvement={early_stopping_min_relative_improvement} "
        f"early_stopping_smoothing_window={early_stopping_smoothing_window}"
    )
    history = model.fit(
        x_train,
        alpha_train,
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(out_root / "history.csv", index=False)
    pd.DataFrame(early_stop.records).to_csv(out_root / "training_loss_early_stopping_log.csv", index=False)

    loss_delta = np.nan
    if "loss" in hist_df.columns and len(hist_df) > 1:
        loss_delta = float(hist_df["loss"].iloc[-1] - hist_df["loss"].iloc[0])

    pd.DataFrame(
        [
            {
                "first_loss": float(hist_df["loss"].iloc[0]) if "loss" in hist_df.columns else np.nan,
                "last_loss": float(hist_df["loss"].iloc[-1]) if "loss" in hist_df.columns else np.nan,
                "loss_delta_last_minus_first": loss_delta,
                "best_smoothed_train_loss": early_stop.best_smoothed_loss,
                "best_smoothed_train_loss_epoch": early_stop.best_epoch,
                "early_stopped_epoch": early_stop.stopped_epoch if early_stop.stopped_epoch else np.nan,
                "early_stopping_min_epochs": early_stopping_min_epochs,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_relative_improvement": early_stopping_min_relative_improvement,
                "early_stopping_smoothing_window": early_stopping_smoothing_window,
            }
        ]
    ).to_csv(out_root / "loss_trend_summary.csv", index=False)

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint was not written: {best_checkpoint_path}")
    model.load_weights(str(best_checkpoint_path))
    model.save(str(model_dir / "sd-cnn_model_lineage_holdout.h5"), include_optimizer=True)

    x_test = x_sparse[test_idx, :].todense()
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
    parser.add_argument("--min-class-count", type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-relative-improvement", type=float, default=None)
    parser.add_argument("--early-stopping-smoothing-window", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        kwargs = yaml.safe_load(f)

    seed = int(kwargs.get("random_seed", 42))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    heldouts = [args.heldout_lineage] if args.heldout_lineage else list(MAJOR_LINEAGES)
    for heldout in heldouts:
        _run_one_lineage(
            kwargs,
            str(heldout),
            args.dry_run,
            args.epochs_override,
            args.min_class_count,
            args.batch_size,
            args.early_stopping_min_epochs,
            args.early_stopping_patience,
            args.early_stopping_min_relative_improvement,
            args.early_stopping_smoothing_window,
        )

    K.clear_session()


if __name__ == "__main__":
    main()
