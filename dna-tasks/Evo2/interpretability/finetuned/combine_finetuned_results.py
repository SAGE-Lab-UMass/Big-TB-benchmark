"""Merge finetuned lineage/shard SHAP outputs and compute MAP/MAR summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from map_mar import compute_map_mar_rows, mean_map_mar_rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_file", help="YAML configuration file")
    parser.add_argument("--drug", required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def combine_explainer_shards(config: dict, drug: str, shard_count: int) -> None:
    if shard_count < 2:
        return

    output_dir = Path(config["output_dir"])
    drug_shap_dir = output_dir / "shap_values" / drug
    fixed_explainer_path = drug_shap_dir / "explainer_samples.csv"
    if not fixed_explainer_path.is_file():
        raise FileNotFoundError(f"Missing fixed explainer: {fixed_explainer_path}")
    fixed_ids = set(pd.read_csv(fixed_explainer_path, dtype={"sample_id": str})["sample_id"])

    relevant_path = output_dir / "map_mar" / f"relevant_features_{drug}.csv"
    relevant = set(pd.read_csv(relevant_path)["WHO_R_features"].dropna().astype(str))
    top_n_positions = int(config.get("top_n_positions", 100))
    k_values = config.get("k_values", [1, 5, 10])

    lineage_dirs = sorted(drug_shap_dir.glob("heldout_lineage_*"))
    if not lineage_dirs:
        raise FileNotFoundError(f"No heldout lineage directories found below {drug_shap_dir}")

    for lineage_dir in lineage_dirs:
        lineage = lineage_dir.name.removeprefix("heldout_lineage_")
        shard_root = lineage_dir / "shards"
        shard_dirs = [shard_root / f"shard_{index:03d}_of_{shard_count:03d}" for index in range(shard_count)]
        missing = [str(path / "ranked_shap.csv") for path in shard_dirs if not (path / "ranked_shap.csv").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing SHAP shards for {drug} lineage {lineage}: " + ", ".join(missing))

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
            background_parts.append(pd.read_csv(shard_dir / "background_samples.csv", dtype={"sample_id": str}))
            explainer_ids.extend(pd.read_csv(shard_dir / "explainer_samples.csv", dtype={"sample_id": str})["sample_id"].tolist())
            selection_parts.append(pd.read_csv(shard_dir / "sample_selection_summary.csv"))

        reference_background_ids = background_parts[0]["sample_id"].tolist()
        if any(frame["sample_id"].tolist() != reference_background_ids for frame in background_parts[1:]):
            raise ValueError(f"Background mismatch among shards for {drug} lineage {lineage}")
        if len(explainer_ids) != len(set(explainer_ids)) or set(explainer_ids) != fixed_ids:
            raise ValueError(f"Explainer shards do not form the fixed explainer for {drug} lineage {lineage}")

        ranked = (
            pd.concat(ranked_parts, ignore_index=True)
            .groupby(["gene", "position"], as_index=False)["max_abs_shap"]
            .max()
            .sort_values(["max_abs_shap", "gene", "position"], ascending=[False, True, True])
            .reset_index(drop=True)
        )
        ranked.to_csv(lineage_dir / "ranked_shap.csv", index=False)
        ranked_features = [f"{row.gene}_{int(row.position)}" for row in ranked.head(top_n_positions).itertuples()]
        pd.DataFrame({"feature": ranked_features}).to_csv(lineage_dir / "selected_features.csv", index=False)
        background_parts[0].to_csv(lineage_dir / "background_samples.csv", index=False)

        selection = selection_parts[0].iloc[0].to_dict()
        selection["explainer_samples_in_shard"] = sum(int(frame.iloc[0]["explainer_samples_in_shard"]) for frame in selection_parts)
        selection["explainer_shard_index"] = "all"
        selection["completed_shards"] = shard_count
        pd.DataFrame([selection]).to_csv(lineage_dir / "sample_selection_summary.csv", index=False)

        metric_rows = [{"drug": drug, "heldout_lineage": lineage, **row} for row in compute_map_mar_rows(ranked_features, relevant, k_values)]
        pd.DataFrame(metric_rows).to_csv(output_dir / "map_mar" / f"map_mar_{drug}_heldout_lineage_{lineage}.csv", index=False)
        print(f"Combined {shard_count} SHAP shards for {drug} lineage {lineage}")


def main() -> None:
    args = parse_arguments()
    drug = args.drug.upper()
    with Path(args.parameter_file).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    combine_explainer_shards(config, drug, args.shard_count)

    map_mar_dir = Path(config["output_dir"]) / "map_mar"
    per_lineage_paths = sorted(map_mar_dir.glob(f"map_mar_{drug}_heldout_lineage_*.csv"))
    if not per_lineage_paths:
        raise FileNotFoundError(f"No per-lineage MAP/MAR CSVs found for {drug} under {map_mar_dir}")

    by_lineage = pd.concat((pd.read_csv(path) for path in per_lineage_paths), ignore_index=True)
    by_lineage = by_lineage.sort_values(["heldout_lineage", "k"])
    mean_rows = mean_map_mar_rows(by_lineage)
    mean_rows.insert(0, "drug", drug)

    by_lineage_path = map_mar_dir / f"map_mar_{drug}_by_lineage.csv"
    mean_path = map_mar_dir / f"map_mar_{drug}_mean.csv"
    by_lineage.to_csv(by_lineage_path, index=False)
    mean_rows.to_csv(mean_path, index=False)

    drug_shap_dir = Path(config["output_dir"]) / "shap_values" / drug
    selection_paths = sorted(drug_shap_dir.glob("heldout_lineage_*/sample_selection_summary.csv"))
    if selection_paths:
        selection_summary = pd.concat((pd.read_csv(path) for path in selection_paths), ignore_index=True)
        selection_summary.sort_values("heldout_lineage").to_csv(drug_shap_dir / "sample_selection_summary.csv", index=False)

    print(f"Saved lineage-level metrics to {by_lineage_path}")
    print(f"Saved mean metrics to {mean_path}")


if __name__ == "__main__":
    main()
