#!/usr/bin/env python
"""Summarize SD-CNN lineage-holdout split counts without training.

Writes N, N_S, N_R for requested held-out lineages using the same
New_ID -> ROLLINGDB_ID lineage join as the SD-CNN lineage-holdout runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = THIS_DIR.parent
DNA_TASKS_DIR = MODEL_TRAINING_DIR.parents[1]
DATA_CURATION_DIR = DNA_TASKS_DIR.parent
LINEAGE_CSV = DATA_CURATION_DIR / "BIG_TB_isolates_with_lineages.csv"


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


def _to_binary_labels(y: pd.Series) -> pd.Series:
    y_str = y.fillna("-1").astype(str).str.strip().str.upper()
    mapped = y_str.map({"R": 0, "S": 1, "-1": -1, "-1.0": -1})
    mapped = mapped.fillna(-1).astype(int)
    return mapped


def summarize_counts(
    config_path: Path,
    heldout_lineages: list[str],
    output_csv: Path,
    min_class_count: int = 50,
) -> pd.DataFrame:
    with open(config_path, "r") as f:
        kwargs = yaml.safe_load(f)

    drug = kwargs["drug"]
    metadata_path = Path(kwargs["metadata_path"])

    df = pd.read_parquet(metadata_path).copy()
    if "New_ID" not in df.columns:
        df["New_ID"] = df.index.astype(str)

    lineage_df = pd.read_csv(LINEAGE_CSV, usecols=["ROLLINGDB_ID", "Lineage"])
    lineage_df["ROLLINGDB_ID"] = lineage_df["ROLLINGDB_ID"].astype(str)

    merged = df.merge(
        lineage_df,
        left_on="New_ID",
        right_on="ROLLINGDB_ID",
        how="left",
    )
    merged["Lineage"] = _normalize_lineage_series(merged["Lineage"])

    y_all = _to_binary_labels(merged[drug])
    valid_mask = (y_all != -1) & merged["Lineage"].notna()
    valid_df = merged.loc[valid_mask].copy()
    valid_y = y_all.loc[valid_mask]

    rows = []
    for heldout in heldout_lineages:
        test_mask = valid_df["Lineage"].astype(str) == str(heldout)
        train_mask = ~test_mask

        y_test = valid_y.loc[test_mask]
        y_train = valid_y.loc[train_mask]

        n = int(test_mask.sum())
        n_s = int((y_test == 1).sum())
        n_r = int((y_test == 0).sum())
        train_n_s = int((y_train == 1).sum())
        train_n_r = int((y_train == 0).sum())
        feasible = str(
            bool(min(train_n_s, train_n_r, n_s, n_r) >= min_class_count)
        ).lower()

        rows.append(
            {
                "drug": drug,
                "heldout_lineage": str(heldout),
                "N": n,
                "N_S": n_s,
                "N_R": n_r,
                "feasible": feasible,
            }
        )

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to lineage-holdout YAML config")
    parser.add_argument(
        "--heldout-lineages",
        nargs="+",
        default=["1", "2", "3", "4"],
        help="Held-out lineage labels to summarize (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. Defaults next to config as split_counts_lineage_1_2_3_4.csv",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    heldouts = [str(x) for x in args.heldout_lineages]

    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        suffix = "_".join(heldouts)
        output_csv = config_path.parent / f"split_counts_lineage_{suffix}.csv"

    out_df = summarize_counts(config_path, heldouts, output_csv, min_class_count=50)
    print(out_df.to_string(index=False))
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
