"""Recover isolate-level row manifests for saved regression feature matrices.

The regression `.npy` matrices were produced before companion row manifests were
saved. Lineage-aware splitting requires a trustworthy mapping from matrix row to
isolate ID, phenotype, and lineage. This script reconstructs that mapping from
the original sequence CSVs and validates it against the saved labels/features
before writing `*_row_manifest.csv` files.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROTEIN_TASKS_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = PROTEIN_TASKS_DIR / "data" / "latest"
SEQ_DIR = DATA_ROOT / "sequence_data_csv"
FEATURE_DIR = DATA_ROOT / "feature_matrix_labels"
MANIFEST_DIR = FEATURE_DIR / "row_manifests"
REF_SEQ_FILE = PROTEIN_TASKS_DIR / "data" / "catalog" / "protein_sequences.csv"
LINEAGE_CSV = PROTEIN_TASKS_DIR.parent / "BIG_TB_isolates_with_lineages.csv"

# Single-gene matrices preserve CSV row order after length/frame filtering.
SINGLE_GENE_DRUGS: Dict[str, list[str]] = {
    "rifampicin": ["rpoB"],
    "pyrazinamide": ["pncA"],
    "capreomycin": ["tlyA"],
    "amikacin": ["eis"],
}

# Multi-gene matrices need hash-based recovery because the historical notebook
# used set-derived shared isolate IDs, so row order cannot be recreated safely
# from filenames alone.
MULTI_GENE_DRUGS: Dict[str, list[str]] = {
    "moxifloxacin": ["gyrA", "gyrB"],
    "streptomycin": ["rpsL", "gid"],
    "isoniazid": ["katG", "inhA"],
    "ethionamide": ["ethA", "ethR", "inhA"],
    "ethambutol": ["embC", "embB", "embA"],
    "levofloxacin": ["gyrA", "gyrB"],
}

ALL_DRUGS = {**SINGLE_GENE_DRUGS, **MULTI_GENE_DRUGS}


def load_lineage_table() -> pd.DataFrame:
    df = pd.read_csv(LINEAGE_CSV, usecols=["ROLLINGDB_ID", "Lineage", "F2"])
    df["ROLLINGDB_ID"] = df["ROLLINGDB_ID"].astype(str)
    return df


def load_reference_sequences() -> dict[str, str]:
    ref = pd.read_csv(REF_SEQ_FILE)
    return {row.gene: str(row.protein_sequence) for row in ref.itertuples(index=False)}


def normalize_saved_labels(y_saved: np.ndarray) -> pd.Series:
    """Normalize historical label arrays to the R/S strings used in manifests."""
    ys = pd.Series(y_saved)
    if ys.dtype.kind in {"U", "S", "O"}:
        return ys.astype(str)
    if ys.dtype.kind in {"i", "u", "b"}:
        mapper = {0: "R", 1: "S"}
        uniq = set(ys.astype(int).unique().tolist())
        if uniq.issubset({0, 1}):
            return ys.astype(int).map(mapper)
    raise ValueError(f"Unsupported label dtype for manifest recovery: {ys.dtype}")


def _row_hash(arr: np.ndarray) -> str:
    """Hash an encoded feature row for order-independent matrix matching."""
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _encode_sequence(sequence: str, reference: str) -> np.ndarray | None:
    """Recreate the binary reference/alternate encoding used by regression."""
    sequence = str(sequence)
    reference = str(reference)
    if len(sequence) == len(reference):
        use = sequence
    elif len(sequence) == len(reference) - 1:
        use = sequence + reference[-1]
    else:
        return None
    return np.fromiter((0 if aa == ref else 1 for aa, ref in zip(use, reference)), dtype=np.int8)


def _load_saved_matrix_and_labels(drug: str) -> tuple[np.ndarray, pd.Series]:
    feat_path = FEATURE_DIR / f"{drug.upper()}_feature_matrix.npy"
    label_path = FEATURE_DIR / f"{drug.upper()}_labels.npy"
    X = np.load(feat_path, mmap_mode="r")
    y = normalize_saved_labels(np.load(label_path, allow_pickle=True))
    return X, y


def recover_single_gene_manifest(drug: str, gene: str, ref_sequences: dict[str, str], lineage_df: pd.DataFrame) -> pd.DataFrame:
    """Recover row IDs for single-gene regression matrices by CSV order."""
    csv_path = SEQ_DIR / f"{gene}_{drug.upper()}_combined_sequence_data.csv"
    df = pd.read_csv(csv_path)
    df["Filename"] = df["Filename"].astype(str)
    df["Phenotype"] = df["Phenotype"].astype(str)
    df["protein_len"] = df["Protein_Sequence"].astype(str).str.len()
    ref_len = len(ref_sequences[gene])

    # The original single-gene feature matrices were built from non-frameshift
    # rows with reference-length protein sequences, preserving this CSV order.
    candidate = df[(df["Frameshift_Mutation"] == 0) & (df["protein_len"] == ref_len)].copy()
    candidate = candidate.reset_index(drop=False).rename(columns={"index": "sequence_csv_row_idx"})

    X, y_saved = _load_saved_matrix_and_labels(drug)
    if len(candidate) != len(y_saved):
        raise ValueError(f"{drug}: candidate rows {len(candidate)} do not match saved labels {len(y_saved)}")
    if X.shape[0] != len(candidate):
        raise ValueError(f"{drug}: feature rows {X.shape[0]} do not match candidate rows {len(candidate)}")

    # Label-order equality is the strongest check that the reconstructed row
    # order matches the saved matrix and label arrays.
    observed = candidate["Phenotype"].reset_index(drop=True)
    if not observed.equals(y_saved.reset_index(drop=True)):
        mismatch = (observed != y_saved.reset_index(drop=True))
        mismatch_n = int(mismatch.sum())
        example_idx = int(np.flatnonzero(mismatch.to_numpy())[0]) if mismatch_n else -1
        raise ValueError(
            f"{drug}: saved labels do not match candidate phenotype order; mismatches={mismatch_n}, first_idx={example_idx}"
        )

    manifest = candidate[[
        "sequence_csv_row_idx",
        "Filename",
        "Phenotype",
        "Frameshift_Mutation",
        "protein_len",
    ]].copy()
    manifest["drug"] = drug
    manifest["genes"] = gene
    manifest["reference_length"] = ref_len
    manifest["matrix_row_idx"] = np.arange(len(manifest), dtype=int)
    manifest["saved_label"] = y_saved.to_numpy()
    manifest["vector_hash"] = [
        _row_hash(np.asarray(X[i], dtype=np.int8)) for i in range(X.shape[0])
    ]
    manifest["duplicate_bucket_size"] = 1
    manifest["duplicate_bucket_lineage_count"] = 1
    manifest["assignment_ambiguous"] = False
    manifest["cross_lineage_duplicate_bucket"] = False
    manifest = manifest.merge(lineage_df, left_on="Filename", right_on="ROLLINGDB_ID", how="left").drop(columns=["ROLLINGDB_ID"])

    cols = [
        "drug", "genes", "matrix_row_idx", "sequence_csv_row_idx", "Filename", "Phenotype", "saved_label",
        "Lineage", "F2", "Frameshift_Mutation", "protein_len", "reference_length", "vector_hash",
        "duplicate_bucket_size", "duplicate_bucket_lineage_count", "assignment_ambiguous", "cross_lineage_duplicate_bucket",
    ]
    return manifest[cols]


def recover_multi_gene_manifest(drug: str, genes: list[str], ref_sequences: dict[str, str], lineage_df: pd.DataFrame) -> pd.DataFrame:
    """Recover row IDs for multi-gene matrices by feature-vector matching."""
    gene_tables = []
    shared_ids = None
    for gene in genes:
        csv_path = SEQ_DIR / f"{gene}_{drug.upper()}_combined_sequence_data.csv"
        df = pd.read_csv(csv_path)
        df["Filename"] = df["Filename"].astype(str)
        df["Phenotype"] = df["Phenotype"].astype(str)
        df = df[df["Frameshift_Mutation"] == 0].copy()
        df = df.reset_index(drop=False).rename(columns={"index": f"{gene}_sequence_csv_row_idx"})
        keep_cols = ["Filename", "Phenotype", "Protein_Sequence", f"{gene}_sequence_csv_row_idx"]
        df = df[keep_cols].set_index("Filename", drop=False)
        gene_tables.append((gene, df))
        ids = set(df.index)
        shared_ids = ids if shared_ids is None else shared_ids & ids

    # Reconstruct one concatenated binary feature vector per shared isolate.
    # Any isolate with inconsistent phenotypes or incompatible sequence lengths
    # is skipped, matching the constraints of the saved matrices.
    records = []
    skipped = []
    for fid in sorted(shared_ids):
        vectors = []
        labels = []
        rec = {"Filename": fid}
        ok = True
        for gene, df in gene_tables:
            row = df.loc[fid]
            rec[f"{gene}_sequence_csv_row_idx"] = int(row[f"{gene}_sequence_csv_row_idx"])
            rec[f"{gene}_protein_len"] = len(str(row["Protein_Sequence"]))
            encoded = _encode_sequence(row["Protein_Sequence"], ref_sequences[gene])
            if encoded is None:
                skipped.append({"Filename": fid, "Reason": f"{gene}: length mismatch"})
                ok = False
                break
            vectors.append(encoded)
            labels.append(str(row["Phenotype"]))
        if not ok:
            continue
        if len(set(labels)) != 1:
            skipped.append({"Filename": fid, "Reason": "phenotype mismatch across genes"})
            continue
        full_vector = np.concatenate(vectors).astype(np.int8)
        rec["Phenotype"] = labels[0]
        rec["saved_label"] = labels[0]
        rec["vector_hash"] = _row_hash(full_vector)
        records.append(rec)

    candidate = pd.DataFrame(records)
    X, y_saved = _load_saved_matrix_and_labels(drug)
    if len(candidate) != X.shape[0] or len(candidate) != len(y_saved):
        raise ValueError(
            f"{drug}: candidate rows {len(candidate)}, saved rows {X.shape[0]}, saved labels {len(y_saved)}"
        )

    saved_rows = []
    for row_idx in range(X.shape[0]):
        saved_rows.append({
            "matrix_row_idx": row_idx,
            "saved_label": y_saved.iloc[row_idx],
            "vector_hash": _row_hash(np.asarray(X[row_idx], dtype=np.int8)),
        })
    saved_df = pd.DataFrame(saved_rows)

    # Compare hash/label multiplicities before assignment. This catches any
    # drift in preprocessing without relying on row order.
    saved_counts = saved_df.groupby(["vector_hash", "saved_label"]).size().to_dict()
    cand_counts = candidate.groupby(["vector_hash", "saved_label"]).size().to_dict()
    if saved_counts != cand_counts:
        raise ValueError(f"{drug}: reconstructed hash/label counts do not match saved matrix counts")

    # Duplicate feature vectors are common after reference/alternate encoding.
    # Within each hash/label bucket, assignment to exact matrix rows is not
    # biologically identifiable, so we mark ambiguous and cross-lineage buckets
    # explicitly for downstream reporting.
    assigned = []
    for (vec_hash, label), bucket_df in candidate.groupby(["vector_hash", "saved_label"], sort=False):
        saved_bucket = saved_df[(saved_df["vector_hash"] == vec_hash) & (saved_df["saved_label"] == label)].sort_values("matrix_row_idx")
        cand_bucket = bucket_df.sort_values(["Filename"]).copy()
        if len(saved_bucket) != len(cand_bucket):
            raise ValueError(f"{drug}: bucket size mismatch for {(vec_hash, label)}")
        lineage_bucket = cand_bucket.merge(lineage_df, left_on="Filename", right_on="ROLLINGDB_ID", how="left")
        lineage_count = int(lineage_bucket["Lineage"].dropna().astype(str).nunique())
        bucket_size = len(cand_bucket)
        cand_bucket["matrix_row_idx"] = saved_bucket["matrix_row_idx"].to_numpy()
        cand_bucket["duplicate_bucket_size"] = bucket_size
        cand_bucket["duplicate_bucket_lineage_count"] = lineage_count
        cand_bucket["assignment_ambiguous"] = bucket_size > 1
        cand_bucket["cross_lineage_duplicate_bucket"] = lineage_count > 1
        assigned.append(cand_bucket)

    manifest = pd.concat(assigned, ignore_index=True)
    manifest = manifest.merge(lineage_df, left_on="Filename", right_on="ROLLINGDB_ID", how="left").drop(columns=["ROLLINGDB_ID"])
    manifest["drug"] = drug
    manifest["genes"] = ",".join(genes)
    manifest = manifest.sort_values("matrix_row_idx").reset_index(drop=True)
    if not manifest["saved_label"].reset_index(drop=True).equals(y_saved.reset_index(drop=True)):
        raise ValueError(f"{drug}: manifest saved_label order does not match saved label array after assignment")

    ordered_cols = [
        "drug", "genes", "matrix_row_idx", "Filename", "Phenotype", "saved_label", "Lineage", "F2",
        "vector_hash", "duplicate_bucket_size", "duplicate_bucket_lineage_count", "assignment_ambiguous",
        "cross_lineage_duplicate_bucket",
    ]
    for gene in genes:
        ordered_cols.extend([f"{gene}_sequence_csv_row_idx", f"{gene}_protein_len"])
    return manifest[ordered_cols]


def write_manifest(drug: str, manifest: pd.DataFrame) -> Path:
    """Persist one recovered row manifest next to the regression matrices."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANIFEST_DIR / f"{drug.upper()}_row_manifest.csv"
    manifest.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", choices=sorted(ALL_DRUGS), default=None)
    parser.add_argument("--only", choices=["single", "multi", "all"], default="all")
    args = parser.parse_args()

    lineage_df = load_lineage_table()
    ref_sequences = load_reference_sequences()

    items = []
    if args.only in {"single", "all"}:
        items.extend(SINGLE_GENE_DRUGS.items())
    if args.only in {"multi", "all"}:
        items.extend(MULTI_GENE_DRUGS.items())
    if args.drug:
        items = [(args.drug, ALL_DRUGS[args.drug])]

    rows = []
    for drug, genes in items:
        if len(genes) == 1:
            manifest = recover_single_gene_manifest(drug, genes[0], ref_sequences, lineage_df)
        else:
            manifest = recover_multi_gene_manifest(drug, genes, ref_sequences, lineage_df)
        out_path = write_manifest(drug, manifest)
        rows.append({
            "drug": drug,
            "genes": ",".join(genes),
            "rows": len(manifest),
            "lineage_annotated": int(manifest["Lineage"].notna().sum()),
            "ambiguous_rows": int(manifest.get("assignment_ambiguous", pd.Series(dtype=bool)).sum()),
            "cross_lineage_ambiguous_rows": int(manifest.get("cross_lineage_duplicate_bucket", pd.Series(dtype=bool)).sum()),
            "out_path": str(out_path),
        })
        print(f"[ok] {drug}: recovered {len(manifest)} rows -> {out_path}")

    pd.DataFrame(rows).to_csv(MANIFEST_DIR / "manifest_summary.csv", index=False)


if __name__ == "__main__":
    main()
