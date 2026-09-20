"""MAP/MAR helpers for random-split Evo2 interpretability.

Copied unchanged from ``interpretability/lineage_aware_split/map_mar.py``
(itself copied from ``Regression/interpretability/lineage_aware_split/map_mar.py``).
The metric definitions are model-agnostic (they only operate on ranked
``"<gene>_<position>"`` feature strings and a WHO/VCF ``"R"``-labelled
relevant set), so this module is reused as-is here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


K_VALUES = (1, 5, 10)


def find_drug_column(df: pd.DataFrame, drug: str) -> str:
    """Return the case-insensitive phenotype column matching ``drug``."""
    for column in df.columns:
        if column.lower() == drug.lower():
            return column
    raise ValueError(f"Drug column '{drug}' not found")


def create_mutations_column(
    mapped_df: pd.DataFrame,
    has_neg_strand: bool,
) -> pd.DataFrame:
    """Create feature names using the same mapping as the random-split run."""
    mapped_df = mapped_df.copy()

    if has_neg_strand:
        position = mapped_df["rel_gapped_mutation_position_neg_strand"].where(
            mapped_df["rel_gapped_mutation_position_neg_strand"].notna(),
            mapped_df["rel_gapped_mutation_position_pos_strand"],
        )
    else:
        position = mapped_df["rel_gapped_mutation_position_pos_strand"]

    # Positive positions in the WHO-mapped files are one-indexed. Upstream
    # (negative) positions are already zero-indexed.
    adjusted_position = position.where(position < 0, position - 1)
    mask = position.notna()
    mapped_df.loc[mask, "WHO_mutation_feature"] = (
        mapped_df.loc[mask, "gene"].astype(str)
        + "_"
        + adjusted_position.loc[mask].astype(int).astype(str)
    )
    return mapped_df


def load_relevant_features(
    vcf_who_map_directory: str | Path,
    drug: str,
    has_neg_strand: bool,
    output_csv: str | Path | None = None,
) -> set[str]:
    """Collect all mapped features labelled resistant for ``drug``."""
    directory = Path(vcf_who_map_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"WHO/VCF mapping directory not found: {directory}")

    relevant: set[str] = set()
    csv_paths = sorted(directory.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    for csv_path in csv_paths:
        try:
            mapped_df = pd.read_csv(csv_path)
            mapped_df = create_mutations_column(mapped_df, has_neg_strand)
            drug_column = find_drug_column(mapped_df, drug)
        except Exception as exc:
            print(f"[warn] Skipping {csv_path.name}: {exc}")
            continue

        relevant.update(
            mapped_df.loc[
                mapped_df[drug_column].astype(str).str.upper() == "R",
                "WHO_mutation_feature",
            ].dropna().astype(str)
        )

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sorted(relevant), columns=["WHO_R_features"]).to_csv(
            output_path, index=False
        )

    if not relevant:
        raise ValueError(f"No resistant WHO features found for {drug} in {directory}")
    return relevant


def _hits_at_k(predicted: Sequence[str], relevant: set[str], k: int) -> int:
    return sum(feature in relevant for feature in predicted[:k])


def _average_precision_at_k(
    predicted: Sequence[str], relevant: set[str], k: int
) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, feature in enumerate(predicted[:k], start=1):
        if feature in relevant:
            hits += 1
            precision_sum += hits / rank
    # This denominator matches the existing random-split implementation.
    return precision_sum / len(relevant) if relevant else 0.0


def compute_map_mar_rows(
    important_features_ranked: Sequence[str],
    relevant_features: set[str],
    k_values: Iterable[int] = K_VALUES,
) -> list[dict[str, float | int]]:
    """Compute the same P@k, R@k, MAP@k and Hits@k columns as random split."""
    rows: list[dict[str, float | int]] = []
    for k_value in k_values:
        k = int(k_value)
        if k <= 0:
            raise ValueError(f"k values must be positive; received {k}")
        hits = _hits_at_k(important_features_ranked, relevant_features, k)
        rows.append(
            {
                "k": k,
                "P@k": hits / k if relevant_features else 0.0,
                "R@k": hits / len(relevant_features) if relevant_features else 0.0,
                "MAP@k": _average_precision_at_k(
                    important_features_ranked, relevant_features, k
                ),
                "Hits@k": hits,
            }
        )
    return rows
