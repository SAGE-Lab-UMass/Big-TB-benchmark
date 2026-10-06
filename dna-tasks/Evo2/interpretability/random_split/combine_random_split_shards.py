"""Merge per-shard SHAP rankings written by parallel array jobs.

When ``run_interpret_evo2_random_split.py`` is invoked with
``--explainer-shard-index``/``--explainer-shard-count`` (one process per
explainer shard, e.g. from a SLURM array), each task writes its own
``shap_values/<DRUG>/fold_<N>/shards/shard_XXX_of_YYY/ranked_shap.csv``
instead of the combined ``fold_<N>/ranked_shap.csv``. Run this script once
every shard has finished to merge them (taking the per-position maximum
across shards, equivalent to ranking over the complete explainer set) and
compute the drug's MAP/MAR metrics.

Usage::

    python combine_random_split_shards.py parameter_files/shap_interpret_random_split.yaml --drug AMIKACIN
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from map_mar import compute_map_mar_rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_file", help="YAML configuration file")
    parser.add_argument("--drug", required=True, help="Drug whose shard files should be combined")
    parser.add_argument(
        "--shard-count",
        type=int,
        required=True,
        help="Number of explainer shards that were submitted for this drug",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    drug = args.drug.upper()
    shard_count = args.shard_count
    with Path(args.parameter_file).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_dir = Path(config["output_dir"])
    drug_shap_dir = output_dir / "shap_values" / drug
    selection_path = drug_shap_dir / "fold_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Missing fold selection: {selection_path}. Run --prep-only first."
        )
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    fold = selection["fold"]
    fold_shap_dir = drug_shap_dir / f"fold_{fold}"

    fixed_explainer_path = fold_shap_dir / "explainer_samples.csv"
    if not fixed_explainer_path.is_file():
        raise FileNotFoundError(f"Missing fixed explainer: {fixed_explainer_path}")
    fixed_ids = set(
        pd.read_csv(fixed_explainer_path, dtype={"sample_id": str})["sample_id"]
    )

    shard_root = fold_shap_dir / "shards"
    shard_dirs = [
        shard_root / f"shard_{index:03d}_of_{shard_count:03d}" for index in range(shard_count)
    ]
    missing = [
        str(path / "ranked_shap.csv") for path in shard_dirs if not (path / "ranked_shap.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} SHAP shard(s) for {drug} fold {fold}: " + ", ".join(missing)
        )

    ranked_parts = []
    background_parts = []
    explainer_ids: list[str] = []
    selection_parts = []
    for shard_dir in shard_dirs:
        ranked = pd.read_csv(shard_dir / "ranked_shap.csv")
        required = {"gene", "position", "max_abs_shap"}
        if not required.issubset(ranked.columns):
            raise ValueError(f"Invalid shard ranking: {shard_dir / 'ranked_shap.csv'}")
        ranked_parts.append(ranked[["gene", "position", "max_abs_shap"]])
        background_parts.append(
            pd.read_csv(
                shard_dir / "background_samples.csv", dtype={"row_id": str, "sample_id": str}
            )
        )
        explainer_ids.extend(
            pd.read_csv(shard_dir / "explainer_samples.csv", dtype={"sample_id": str})[
                "sample_id"
            ].tolist()
        )
        selection_parts.append(pd.read_csv(shard_dir / "sample_selection_summary.csv"))

    reference_background_ids = background_parts[0]["sample_id"].tolist()
    if any(
        frame["sample_id"].tolist() != reference_background_ids for frame in background_parts[1:]
    ):
        raise ValueError(f"Background mismatch among shards for {drug} fold {fold}")
    if len(explainer_ids) != len(set(explainer_ids)) or set(explainer_ids) != fixed_ids:
        raise ValueError(
            f"Explainer shards do not form the fixed explainer for {drug} fold {fold}"
        )

    ranked = (
        pd.concat(ranked_parts, ignore_index=True)
        .groupby(["gene", "position"], as_index=False)["max_abs_shap"]
        .max()
        .sort_values(["max_abs_shap", "gene", "position"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    ranked.to_csv(fold_shap_dir / "ranked_shap.csv", index=False)

    top_n_positions = int(config.get("top_n_positions", 100))
    ranked_features = [
        f"{row.gene}_{int(row.position)}" for row in ranked.head(top_n_positions).itertuples()
    ]
    pd.DataFrame({"feature": ranked_features}).to_csv(
        fold_shap_dir / "selected_features.csv", index=False
    )
    background_parts[0].to_csv(fold_shap_dir / "background_samples.csv", index=False)

    selection_row = selection_parts[0].iloc[0].to_dict()
    selection_row["explainer_samples_in_shard"] = sum(
        int(frame.iloc[0]["explainer_samples_in_shard"]) for frame in selection_parts
    )
    selection_row["explainer_shard_index"] = "all"
    selection_row["explainer_shard_total"] = shard_count
    pd.DataFrame([selection_row]).to_csv(
        fold_shap_dir / "sample_selection_summary.csv", index=False
    )
    print(f"Combined {shard_count} SHAP shards for {drug} fold {fold}")

    relevant_path = output_dir / "map_mar" / f"relevant_features_{drug}.csv"
    relevant = set(pd.read_csv(relevant_path)["WHO_R_features"].dropna().astype(str))
    k_values = config.get("k_values", [1, 5, 10])

    rows = [
        {"drug": drug, "fold": fold, **row}
        for row in compute_map_mar_rows(ranked_features, relevant, k_values)
    ]
    map_mar_dir = output_dir / "map_mar"
    map_mar_dir.mkdir(parents=True, exist_ok=True)
    map_mar_path = map_mar_dir / f"map_mar_{drug}.csv"
    pd.DataFrame(rows).sort_values("k").to_csv(map_mar_path, index=False)
    print(f"Saved MAP/MAR metrics to {map_mar_path}")


if __name__ == "__main__":
    main()
