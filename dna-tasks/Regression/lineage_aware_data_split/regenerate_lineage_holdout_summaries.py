#!/usr/bin/env python3
"""
Regenerate test_set_accuracy.csv and all_lineage_summary.csv from split_manifest.csv files.
This fixes the issue where summary files contain stale zero values after input data or
lineage resolution is fixed.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

MAJOR_LINEAGES = ("1", "2", "3", "4")


def regenerate_lineage_holdout_summaries_for_drug(drug_output_dir: Path) -> None:
    """Regenerate all lineage summary files for a given drug."""
    drug_output_dir = Path(drug_output_dir)
    drug_name = drug_output_dir.name

    rows = []
    for heldout in MAJOR_LINEAGES:
        lineage_dir = drug_output_dir / f"heldout_lineage_{heldout}"
        split_manifest_path = lineage_dir / "split_manifest.csv"

        if not split_manifest_path.exists():
            print(f"[skip] {drug_name} lineage {heldout}: no split_manifest.csv found")
            continue

        # Read split manifest
        split_df = pd.read_csv(split_manifest_path)

        if split_df.empty:
            print(f"[skip] {drug_name} lineage {heldout}: split_manifest is empty")
            continue

        # Calculate split statistics from the manifest
        test_mask = split_df["split"] == "test"
        train_mask = split_df["split"] == "train"

        test_df = split_df.loc[test_mask]
        train_df = split_df.loc[train_mask]

        test_s = int((test_df["label"] == 1).sum())
        test_r = int((test_df["label"] == 0).sum())
        train_s = int((train_df["label"] == 1).sum())
        train_r = int((train_df["label"] == 0).sum())

        summary = {
            "drug": drug_name,
            "heldout_lineage": str(heldout),
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
            "feasible": bool(min(train_r, train_s, test_r, test_s) >= 50),
            "min_class_count": 50,
        }

        # Check if there are existing results to preserve
        accuracy_path = lineage_dir / "test_set_accuracy.csv"
        if accuracy_path.exists():
            existing_df = pd.read_csv(accuracy_path)
            if not existing_df.empty:
                existing_row = existing_df.iloc[0]
                # Preserve model metrics if they exist
                if pd.notna(existing_row.get("AUC")):
                    summary["AUC"] = existing_row["AUC"]
                if pd.notna(existing_row.get("acc")):
                    summary["acc"] = existing_row["acc"]
                if pd.notna(existing_row.get("best_C")):
                    summary["best_C"] = existing_row["best_C"]
                if pd.notna(existing_row.get("feasible")):
                    # Only override feasible if it was True (meaning training completed)
                    if existing_row["feasible"]:
                        summary["feasible"] = True

        # Write updated test_set_accuracy.csv
        pd.DataFrame([summary]).to_csv(accuracy_path, index=False)
        print(
            f"[ok] {drug_name} lineage {heldout}: "
            f"N={summary['N']} train_N={summary['train_N']} "
            f"test(S/R)=({test_s}/{test_r}) train(S/R)=({train_s}/{train_r})"
        )
        rows.append(summary)

    if not rows:
        print(f"[warn] {drug_name}: no valid summaries generated")
        return

    # Regenerate all_lineage_summary.csv
    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values(["heldout_lineage"]).reset_index(drop=True)
    summary_path = drug_output_dir / "all_lineage_summary.csv"
    all_df.to_csv(summary_path, index=False)
    print(f"[ok] {drug_name}: wrote all_lineage_summary.csv with {len(all_df)} lineages")


def main():
    if len(sys.argv) < 2:
        print("Usage: python regenerate_lineage_holdout_summaries.py <output_root_dir> [drug1 drug2 ...]")
        print("  If no drugs specified, regenerate for all subdirectories")
        sys.exit(1)

    output_root = Path(sys.argv[1])
    drugs = sys.argv[2:] if len(sys.argv) > 2 else None

    if not output_root.exists():
        raise FileNotFoundError(f"Output root directory not found: {output_root}")

    if drugs:
        for drug in drugs:
            drug_dir = output_root / drug
            if drug_dir.is_dir():
                regenerate_lineage_holdout_summaries_for_drug(drug_dir)
            else:
                print(f"[skip] {drug}: directory not found at {drug_dir}")
    else:
        # Regenerate for all subdirectories
        for drug_dir in sorted(output_root.iterdir()):
            if drug_dir.is_dir() and (drug_dir / "heldout_lineage_1").exists():
                regenerate_lineage_holdout_summaries_for_drug(drug_dir)


if __name__ == "__main__":
    main()
