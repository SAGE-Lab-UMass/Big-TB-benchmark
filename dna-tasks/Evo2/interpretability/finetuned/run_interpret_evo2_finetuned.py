"""Run lineage-aware SHAP interpretability for finetuned Evo2 models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

THIS_DIR = Path(__file__).resolve().parent
EVO2_DIR = THIS_DIR.parents[1]
if str(EVO2_DIR) not in sys.path:
    sys.path.insert(0, str(EVO2_DIR))

from map_mar import K_VALUES, compute_map_mar_rows, load_relevant_features, mean_map_mar_rows
from shap_utils import (
    build_sequence_table,
    compute_shap_for_finetuned_model,
    dedup_and_save_indices,
    discover_eligible_lineages,
    load_finetuned_model,
    load_or_create_fingerprints,
    rank_features_by_shap,
    reconstruct_model_training_indices,
    sample_manifest,
    select_explainer_shard,
    select_fixed_explainer_indices,
    select_lineage_background_indices,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_file", help="YAML configuration file")
    parser.add_argument("--drug", help="Override the drug in the YAML file")
    parser.add_argument("--heldout-lineage", dest="heldout_lineage", default=None)
    parser.add_argument("--explainer-shard-index", type=int, default=None)
    parser.add_argument("--explainer-shard-count", type=int, default=None)
    parser.add_argument("--prep-only", action="store_true")
    return parser.parse_args()


def load_config(
    parameter_file: str | Path,
    drug_override: str | None,
    heldout_lineage_override: str | None,
    prep_only: bool,
    explainer_shard_index: int | None,
    explainer_shard_count: int | None,
) -> dict[str, Any]:
    with Path(parameter_file).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {parameter_file}")
    if drug_override:
        config["drug"] = drug_override
    config["drug"] = str(config["drug"]).upper()
    if heldout_lineage_override is not None:
        config["heldout_lineage"] = str(heldout_lineage_override)
    if explainer_shard_index is not None:
        config["explainer_shard_index"] = explainer_shard_index
    if explainer_shard_count is not None:
        config["explainer_shard_count"] = explainer_shard_count
    config["prep_only"] = bool(prep_only)
    return config


def _read_or_create_explainer(
    *,
    dataset,
    dedup_indices: list[int],
    drug_shap_dir: Path,
    background_frac: float,
    max_background: int,
    max_explain: int,
    seed: int,
) -> list[int]:
    explainer_path = drug_shap_dir / "explainer_samples.csv"
    if explainer_path.exists():
        existing = pd.read_csv(explainer_path, dtype={"sample_id": str})
        id_to_index = {sample_id: index for index, sample_id in enumerate(dataset.isolate_ids)}
        return [id_to_index[sample_id] for sample_id in existing["sample_id"].astype(str)]

    explanation_indices = select_fixed_explainer_indices(
        dataset,
        dedup_indices,
        background_fraction=background_frac,
        max_background=max_background,
        max_explain=max_explain,
        seed=seed,
    )
    sample_manifest(dataset, explanation_indices).to_csv(explainer_path, index=False)
    return explanation_indices


def run(config: dict[str, Any]) -> tuple[Path | None, Path | None]:
    drug = config["drug"]
    heldout_lineage_filter = config.get("heldout_lineage")
    heldout_lineage_filter = str(heldout_lineage_filter) if heldout_lineage_filter is not None else None
    shard_count = int(config.get("explainer_shard_count", 1))
    shard_index_value = config.get("explainer_shard_index")
    shard_index = int(shard_index_value) if shard_index_value is not None else None
    sharded = shard_count > 1
    if sharded and (heldout_lineage_filter is None or shard_index is None):
        raise ValueError("Sharded runs require --heldout-lineage, --explainer-shard-index, and --explainer-shard-count")
    if not sharded and shard_index not in (None, 0):
        raise ValueError("A nonzero explainer shard index requires shard count > 1")

    eligible = discover_eligible_lineages(
        config["model_dir"],
        drug,
        require_test_metrics=bool(config.get("require_test_metrics", True)),
    )
    if heldout_lineage_filter is not None:
        eligible = [item for item in eligible if item[0] == heldout_lineage_filter]
        if not eligible:
            raise ValueError(f"Requested heldout lineage {heldout_lineage_filter} is not eligible for {drug}")
    print(f"{drug}: eligible held-out lineages = {[lineage for lineage, _ in eligible]}")

    output_dir = Path(config["output_dir"])
    map_mar_dir = output_dir / "map_mar"
    dedup_dir = output_dir / "dedup_sequence_data"
    drug_shap_dir = output_dir / "shap_values" / drug
    drug_shap_dir.mkdir(parents=True, exist_ok=True)

    dataset, per_gene_lengths, gene_names = build_sequence_table(
        drug,
        config["geno_pheno_csv"],
        config["fasta_dir"],
    )
    fingerprints = load_or_create_fingerprints(dataset, f"{drug}_full", dedup_dir)
    dedup_indices = dedup_and_save_indices(dataset, f"{drug}_full", dedup_dir, fingerprints=fingerprints)

    background_frac = float(config.get("background_frac", 0.2))
    max_background = int(config.get("max_background", 160))
    max_explain = int(config.get("max_explain", 360))
    seed = int(config.get("random_seed", 42))
    explanation_indices = _read_or_create_explainer(
        dataset=dataset,
        dedup_indices=dedup_indices,
        drug_shap_dir=drug_shap_dir,
        background_frac=background_frac,
        max_background=max_background,
        max_explain=max_explain,
        seed=seed,
    )
    print(f"{drug}: fixed explainer samples = {len(explanation_indices)}")

    relevant = load_relevant_features(
        config["WHO_VCF_mapped_dir"],
        drug,
        bool(config.get("has_neg_strand", False)),
        map_mar_dir / f"relevant_features_{drug}.csv",
    )
    print(f"{drug}: found {len(relevant)} resistant WHO features")

    if config.get("prep_only"):
        print(f"{drug}: prep-only run complete")
        return None, None

    k_values: Iterable[int] = config.get("k_values", K_VALUES)
    top_n_positions = int(config.get("top_n_positions", 100))
    hidden_batch_size = int(config.get("hidden_batch_size", 1))
    shap_batch_size = int(config.get("shap_batch_size", 1))
    shap_background_batch_size = int(config.get("shap_background_batch_size", 8))
    hidden_dtype = str(config.get("hidden_dtype", "float16"))
    save_shap_values = bool(config.get("save_shap_values", False))
    val_frac = float(config.get("val_frac", 0.2))

    lineage_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for lineage, checkpoint_dir in eligible:
        lineage_shap_dir = drug_shap_dir / f"heldout_lineage_{lineage}"
        lineage_shap_dir.mkdir(parents=True, exist_ok=True)
        result_dir = lineage_shap_dir
        if sharded:
            assert shard_index is not None
            result_dir = lineage_shap_dir / "shards" / f"shard_{shard_index:03d}_of_{shard_count:03d}"
            result_dir.mkdir(parents=True, exist_ok=True)

        training_indices = reconstruct_model_training_indices(
            dataset,
            drug,
            lineage,
            config["geno_pheno_csv"],
            config["lineage_csv"],
            val_frac=val_frac,
            seed=seed,
        )
        background_indices, candidate_count = select_lineage_background_indices(
            dataset,
            training_indices,
            explanation_indices,
            max_background=max_background,
            seed=seed,
            fingerprints=fingerprints,
        )
        run_explanation_indices = explanation_indices
        if sharded:
            assert shard_index is not None
            run_explanation_indices = select_explainer_shard(explanation_indices, shard_index, shard_count)
            sample_manifest(dataset, run_explanation_indices).to_csv(result_dir / "explainer_samples.csv", index=False)

        background_manifest = sample_manifest(dataset, background_indices)
        background_manifest.to_csv(result_dir / "background_samples.csv", index=False)
        background_labels = background_manifest["label"].astype(int)
        selection_row = {
            "drug": drug,
            "heldout_lineage": lineage,
            "checkpoint_dir": str(checkpoint_dir),
            "explainer_count": len(explanation_indices),
            "explainer_samples_in_shard": len(run_explanation_indices),
            "explainer_shard_index": shard_index if sharded else 0,
            "explainer_shard_total": shard_count,
            "model_training_count": len(training_indices),
            "unique_background_candidates": candidate_count,
            "background_count": len(background_indices),
            "background_label_0": int((background_labels == 0).sum()),
            "background_label_1": int((background_labels == 1).sum()),
        }
        selection_rows.append(selection_row)
        pd.DataFrame([selection_row]).to_csv(result_dir / "sample_selection_summary.csv", index=False)

        print(f"{drug} lineage {lineage}: loading finetuned checkpoint {checkpoint_dir}")
        model = load_finetuned_model(checkpoint_dir)
        shap_df = compute_shap_for_finetuned_model(
            model,
            dataset,
            background_indices,
            run_explanation_indices,
            per_gene_lengths,
            gene_names,
            hidden_batch_size=hidden_batch_size,
            shap_batch_size=shap_batch_size,
            background_batch_size=shap_background_batch_size,
            hidden_dtype=hidden_dtype,
        )
        if save_shap_values:
            shap_df.to_pickle(result_dir / "shap_values.pkl")

        ranked_df = rank_features_by_shap(shap_df, gene_names)
        ranked_path = result_dir / "ranked_shap.csv"
        ranked_df.to_csv(ranked_path, index=False)
        if sharded:
            print(f"Saved shard ranking to {ranked_path}")
            return ranked_path, None

        top_df = ranked_df.head(top_n_positions)
        ranked_features = [f"{row.gene}_{row.position}" for row in top_df.itertuples()]
        pd.DataFrame({"feature": ranked_features}).to_csv(lineage_shap_dir / "selected_features.csv", index=False)
        for metric_row in compute_map_mar_rows(ranked_features, relevant, k_values):
            lineage_rows.append({"drug": drug, "heldout_lineage": lineage, **metric_row})

    map_mar_dir.mkdir(parents=True, exist_ok=True)
    by_lineage = pd.DataFrame(lineage_rows).sort_values(["heldout_lineage", "k"])
    if heldout_lineage_filter is not None:
        lineage_path = map_mar_dir / f"map_mar_{drug}_heldout_lineage_{heldout_lineage_filter}.csv"
        by_lineage.to_csv(lineage_path, index=False)
        print(f"Saved lineage-level metrics to {lineage_path}")
        return lineage_path, None

    pd.DataFrame(selection_rows).to_csv(drug_shap_dir / "sample_selection_summary.csv", index=False)
    mean_rows = mean_map_mar_rows(by_lineage)
    mean_rows.insert(0, "drug", drug)
    by_lineage_path = map_mar_dir / f"map_mar_{drug}_by_lineage.csv"
    mean_path = map_mar_dir / f"map_mar_{drug}_mean.csv"
    by_lineage.to_csv(by_lineage_path, index=False)
    mean_rows.to_csv(mean_path, index=False)
    print(f"Saved lineage-level metrics to {by_lineage_path}")
    print(f"Saved mean metrics to {mean_path}")
    return by_lineage_path, mean_path


def main() -> None:
    args = parse_arguments()
    run(
        load_config(
            args.parameter_file,
            args.drug,
            args.heldout_lineage,
            args.prep_only,
            args.explainer_shard_index,
            args.explainer_shard_count,
        )
    )


if __name__ == "__main__":
    main()
