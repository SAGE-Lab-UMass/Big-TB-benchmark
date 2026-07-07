"""Utilities for lineage-aware protein-task train/test splits.

The lineage robustness analysis is intentionally isolate-level. For each drug,
we first build the same model-ready protein table used by the one-hot and ESM
runners, then merge in lineage calls by isolate ID. Test groups are
restricted to the four major M. tuberculosis lineages (1-4). Training groups
include every other isolate with a lineage annotation, including minor or
ambiguous lineage labels, matching the final reviewer-response design.
"""
from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

PROTEIN_TASKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROTEIN_TASKS_DIR.parent
DATA_ROOT = PROTEIN_TASKS_DIR / "data" / "latest"
SEQUENCE_DATA_DIR = DATA_ROOT / "sequence_data_csv"
LINEAGE_CSV = REPO_ROOT / "BIG_TB_isolates_with_lineages.csv"
LINEAGE_SPLIT_DIR = DATA_ROOT / "lineage_splits_all_train"
MAJOR_LINEAGES = ("1", "2", "3", "4")
DEFAULT_MIN_CLASS_COUNT = 50


def load_lineage_table(lineage_csv: Path = LINEAGE_CSV) -> pd.DataFrame:
    """Load the isolate-level top-lineage calls used for split assignment."""
    df = pd.read_csv(lineage_csv)
    df = df[["ROLLINGDB_ID", "Lineage", "F2"]].copy()
    df["ROLLINGDB_ID"] = df["ROLLINGDB_ID"].astype(str)
    df["Lineage"] = df["Lineage"].astype("string")
    return df


def build_protein_drug_df(
    drug: str,
    drug2genes: Dict[str, list[str]],
    sequence_dir: Path = SEQUENCE_DATA_DIR,
) -> pd.DataFrame:
    """Build the per-drug, model-ready protein table before lineage assignment.

    Multi-gene drugs are merged at the isolate level. This keeps the split unit
    as the clinical isolate rather than an individual gene row, which prevents
    one locus from the same isolate entering train while another enters test.
    """
    gene_dfs = []
    for gene in drug2genes[drug]:
        csv_path = sequence_dir / f"{gene}_{drug.upper()}_combined_sequence_data.csv"
        # Match the protein-task preprocessing used by the model runners:
        # binary phenotypes only and no frameshift-flagged rows.
        df = pd.read_csv(csv_path)
        df = df[(df["Frameshift_Mutation"] == 0) & (df["Phenotype"].isin(["R", "S"]))].copy()
        df = df[["Filename", "Protein_Sequence", "Phenotype"]]
        df = df.rename(columns={"Protein_Sequence": f"seq_{gene}"})
        gene_dfs.append(df)

    merged = reduce(
        lambda left, right: pd.merge(left, right, on=["Filename", "Phenotype"], how="inner"),
        gene_dfs,
    )
    merged["Filename"] = merged["Filename"].astype(str)
    merged["Protein_Sequence"] = merged[[f"seq_{g}" for g in drug2genes[drug]]].agg("".join, axis=1)
    return merged


def attach_lineages(df: pd.DataFrame, id_col: str = "Filename") -> pd.DataFrame:
    """Attach lineage labels to a model-ready table without filtering rows."""
    lineage_df = load_lineage_table()
    merged = df.merge(lineage_df, left_on=id_col, right_on="ROLLINGDB_ID", how="left")
    merged["Lineage"] = merged["Lineage"].astype("string")
    return merged


def filter_major_lineages(
    df: pd.DataFrame,
    major_lineages: Iterable[str] = MAJOR_LINEAGES,
) -> pd.DataFrame:
    """Return only major-lineage rows.

    This helper is kept for diagnostics. The final robustness split does not
    call it before training, because minor/ambiguous lineages should remain in
    the training partition when they are not the held-out major lineage.
    """
    major = set(map(str, major_lineages))
    out = df[df["Lineage"].isin(major)].copy()
    out["Lineage"] = out["Lineage"].astype(str)
    return out.reset_index(drop=True)


def filter_lineage_annotated(df: pd.DataFrame) -> pd.DataFrame:
    """Keep all isolates with any lineage call; test eligibility is handled later."""
    out = df[df["Lineage"].notna()].copy()
    out["Lineage"] = out["Lineage"].astype(str)
    return out.reset_index(drop=True)


def split_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count total, resistant, and susceptible rows for split feasibility."""
    resistant = int((df["Phenotype"] == "R").sum())
    susceptible = int((df["Phenotype"] == "S").sum())
    return {"n": int(len(df)), "r": resistant, "s": susceptible}


def build_leave_one_lineage_out_splits(
    df: pd.DataFrame,
    id_col: str = "Filename",
    heldout_lineages: Iterable[str] = MAJOR_LINEAGES,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
) -> dict[str, dict]:
    """Construct leave-one-major-lineage-out splits for a drug.

    For each held-out major lineage, test contains only isolates exactly assigned
    to that top-level lineage. Train contains all remaining lineage-annotated
    isolates, including other major lineages and minor/ambiguous labels. A split
    is reportable only if both train and test have at least `min_class_count`
    resistant and susceptible isolates, so both fitting and AUC estimation are
    supported by both classes.
    """
    splits: dict[str, dict] = {}
    df = filter_lineage_annotated(df)
    for heldout in map(str, heldout_lineages):
        test_df = df[df["Lineage"] == heldout].copy()
        train_df = df[df["Lineage"] != heldout].copy()
        train_counts = split_counts(train_df)
        test_counts = split_counts(test_df)
        feasible = min(
            train_counts["r"], train_counts["s"], test_counts["r"], test_counts["s"]
        ) >= min_class_count
        splits[heldout] = {
            "heldout_lineage": heldout,
            "train_df": train_df,
            "test_df": test_df,
            "train_ids": train_df[id_col].astype(str).tolist(),
            "test_ids": test_df[id_col].astype(str).tolist(),
            "train_counts": train_counts,
            "test_counts": test_counts,
            "feasible": feasible,
            "min_class_count": min_class_count,
        }
    return splits


def write_split_manifest(
    drug: str,
    split: dict,
    out_root: Path = LINEAGE_SPLIT_DIR,
    id_col: str = "Filename",
) -> Path:
    """Write the canonical isolate-level split manifest for one held-out lineage."""
    out_dir = out_root / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout = split["heldout_lineage"]
    train_df = split["train_df"].copy()
    test_df = split["test_df"].copy()
    train_df["split"] = "train"
    test_df["split"] = "test"
    cols = [id_col, "Phenotype", "Lineage", "F2", "split"]
    manifest = pd.concat([train_df[cols], test_df[cols]], ignore_index=True)
    out_path = out_dir / f"heldout_lineage_{heldout}.csv"
    manifest.to_csv(out_path, index=False)
    return out_path


def build_and_save_drug_splits(
    drug: str,
    drug2genes: Dict[str, list[str]],
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Build lineage splits and persist manifests for downstream model runners."""
    df = build_protein_drug_df(drug, drug2genes)
    df = attach_lineages(df)
    df = filter_lineage_annotated(df)
    splits = build_leave_one_lineage_out_splits(df, min_class_count=min_class_count)
    for split in splits.values():
        write_split_manifest(drug, split)
    return df, splits
