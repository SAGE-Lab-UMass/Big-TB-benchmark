"""Run Evo2 zero-shot random-split SHAP interpretability for one drug.

Unlike ``lineage_aware_split`` (which explains every eligible held-out
lineage), the random-split setting trains several independent folds per
drug, so a single representative fold is chosen: the fold with the highest
validation AUC recorded in its training history (falling back to the first
available fold when no fold has a usable history). SHAP importance is then
computed for that fold's exact checkpoint, ranked positions are mapped to
WHO/VCF resistant features, and MAP/MAR metrics are computed.

The background and explainer sets are selected together in one stratified
split directly from the drug's deduplicated dataset, matching
``SD-CNN/interpretability/utils.py::compute_shap_values_strat`` (20%
stratified background capped at 160 samples), with the explainer pool
additionally capped at 360 samples (matching ``lineage_aware_split``).

Usage::

    python run_interpret_evo2_random_split.py parameter_files/shap_interpret_random_split.yaml --drug STREPTOMYCIN

    ./run_all_drugs.sh
"""

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

from evo2_downstream.train import enforce_evo2_drug_index_mapping  # noqa: E402
from zero_shot.lineage_aware.lineage_split import load_isolate_id_map  # noqa: E402

from map_mar import (  # noqa: E402
    K_VALUES,
    compute_map_mar_rows,
    load_relevant_features,
)
from shap_utils import (  # noqa: E402
    build_full_dataset,
    compute_shap_for_model,
    dataset_sample_ids,
    dedup_and_save_indices,
    discover_folds,
    load_fold_model,
    load_fold_selection,
    load_or_create_fingerprints,
    rank_features_by_shap,
    sample_manifest,
    save_fold_selection,
    select_background_and_explainer_indices,
    select_best_fold,
)


DEFAULT_MODEL_SEED = "42"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_file", help="YAML configuration file")
    parser.add_argument(
        "--drug",
        help="Override the drug in the YAML file (useful for the all-drug runner)",
    )
    parser.add_argument(
        "--fold",
        default=None,
        help="Override the auto-selected fold (skips the best-val-AUC search)",
    )
    parser.add_argument(
        "--model-filename",
        default=None,
        help="Override the checkpoint filename ('auto' tries the final model, "
        "then the best-checkpoint model)",
    )
    parser.add_argument(
        "--explainer-shard-index",
        type=int,
        default=None,
        help="Zero-based explainer shard assigned to this process",
    )
    parser.add_argument(
        "--explainer-shard-count",
        type=int,
        default=None,
        help="Total number of explainer shards for this drug",
    )
    parser.add_argument(
        "--prep-only",
        dest="prep_only",
        action="store_true",
        help="Build the dataset, select the best fold, cache dedup indices and "
        "the background/explainer split, then exit (no SHAP). Run once before "
        "launching parallel explainer-shard array jobs.",
    )
    return parser.parse_args()


def load_config(
    parameter_file: str | Path,
    drug_override: str | None = None,
    fold_override: str | None = None,
    prep_only: bool = False,
    model_filename_override: str | None = None,
    explainer_shard_index: int | None = None,
    explainer_shard_count: int | None = None,
) -> dict[str, Any]:
    with Path(parameter_file).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {parameter_file}")
    if drug_override:
        config["drug"] = drug_override
    config["drug"] = str(config["drug"]).upper()
    if fold_override is not None:
        config["fold"] = fold_override
    if model_filename_override is not None:
        config["model_filename"] = model_filename_override
    if explainer_shard_index is not None:
        config["explainer_shard_index"] = explainer_shard_index
    if explainer_shard_count is not None:
        config["explainer_shard_count"] = explainer_shard_count
    config["prep_only"] = bool(prep_only)
    return config


def select_explainer_shard(
    explanation_indices: Iterable[int],
    shard_index: int,
    shard_count: int,
) -> list[int]:
    """Partition an explainer deterministically without dropping samples."""
    explanation_indices = [int(index) for index in explanation_indices]
    if shard_count < 1:
        raise ValueError("explainer shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"explainer shard index {shard_index} is outside [0, {shard_count})"
        )
    shard = explanation_indices[shard_index::shard_count]
    if not shard:
        raise ValueError(
            f"Explainer shard {shard_index} is empty for {len(explanation_indices)} samples"
        )
    return shard


def _resolve_fold_selection(config: dict[str, Any], selection_path: Path) -> dict[str, Any]:
    """Resolve which fold to explain, caching the choice for parallel shard tasks."""
    drug = config["drug"]
    embed_type = config.get("embed_type", "token")
    model_name = config.get("model_name", "DNABERTCNN")
    model_filename = config.get("model_filename", "auto")
    model_seed = str(config.get("model_seed", DEFAULT_MODEL_SEED))
    fold_override = config.get("fold")
    fold_override = str(fold_override) if fold_override is not None else None

    if fold_override is not None:
        folds = discover_folds(
            config["model_dir"], drug, embed_type, model_name, model_filename, model_seed
        )
        if fold_override not in folds:
            raise ValueError(
                f"Requested --fold {fold_override} is not available for {drug} "
                f"(found: {sorted(folds)})"
            )
        seed_used, model_path, checkpoint_name = folds[fold_override]
        selection = {
            "fold": fold_override,
            "seed": seed_used,
            "model_path": str(model_path),
            "model_filename": checkpoint_name,
            "best_val_auc": None,
            "reason": "fold explicitly requested via --fold/config",
            "folds_considered": sorted(folds),
        }
        save_fold_selection(selection_path, selection)
        return selection

    if selection_path.exists():
        return load_fold_selection(selection_path)

    selection = select_best_fold(
        config["model_dir"], drug, embed_type, model_name, model_filename, model_seed
    )
    save_fold_selection(selection_path, selection)
    return selection


def run(config: dict[str, Any]) -> tuple[Path | None, Path | None]:
    drug = config["drug"]
    embed_type = config.get("embed_type", "token")
    model_name = config.get("model_name", "DNABERTCNN")

    shard_count = int(config.get("explainer_shard_count", 1))
    shard_index_value = config.get("explainer_shard_index")
    shard_index = int(shard_index_value) if shard_index_value is not None else None
    sharded = shard_count > 1
    if sharded and shard_index is None:
        raise ValueError(
            "Explainer sharding requires --explainer-shard-index and "
            "--explainer-shard-count"
        )
    if not sharded and shard_index not in (None, 0):
        raise ValueError("A nonzero explainer shard index requires shard count > 1")

    enforce_evo2_drug_index_mapping(config["phenotype_label_path"])

    output_dir = Path(config["output_dir"])
    map_mar_dir = output_dir / "map_mar"
    dedup_dir = output_dir / "dedup_geno_data"
    drug_shap_dir = output_dir / "shap_values" / drug
    drug_shap_dir.mkdir(parents=True, exist_ok=True)

    selection_path = drug_shap_dir / "fold_selection.json"
    selection = _resolve_fold_selection(config, selection_path)
    fold = selection["fold"]
    model_path = Path(selection["model_path"])
    print(f"{drug}: selected fold {fold} -> {model_path.name} ({selection['reason']})")

    dataset, _label_map, per_gene_len, gene_names = build_full_dataset(
        drug, embed_type, config["memmap_dir"], config["phenotype_label_path"]
    )
    print(f"{drug}: dataset has {len(dataset)} labelled samples across genes {gene_names}")

    fingerprints = load_or_create_fingerprints(dataset, f"{drug}_full", dedup_dir)
    dedup_indices = dedup_and_save_indices(
        dataset, f"{drug}_full", dedup_dir, fingerprints=fingerprints
    )
    print(f"{drug}: {len(dedup_indices)} deduplicated samples available for SHAP")

    background_frac = float(config.get("background_frac", 0.2))
    max_background = int(config.get("max_background", 160))
    max_explain = int(config.get("max_explain", 360))
    seed = int(config.get("random_seed", 42))

    fold_shap_dir = drug_shap_dir / f"fold_{fold}"
    fold_shap_dir.mkdir(parents=True, exist_ok=True)
    background_path = fold_shap_dir / "background_samples.csv"
    explainer_path = fold_shap_dir / "explainer_samples.csv"

    isolate_id_map = load_isolate_id_map(config["geno_pheno_csv"])

    if background_path.exists() and explainer_path.exists():
        background_manifest = pd.read_csv(
            background_path, dtype={"row_id": str, "sample_id": str}
        )
        explainer_manifest = pd.read_csv(
            explainer_path, dtype={"row_id": str, "sample_id": str}
        )
        id_to_index = {sid: index for index, sid in enumerate(dataset_sample_ids(dataset))}
        background_indices = [id_to_index[sid] for sid in background_manifest["sample_id"]]
        explanation_indices = [id_to_index[sid] for sid in explainer_manifest["sample_id"]]
    else:
        background_indices, explanation_indices = select_background_and_explainer_indices(
            dataset,
            dedup_indices,
            background_fraction=background_frac,
            max_background=max_background,
            max_explain=max_explain,
            seed=seed,
        )
        background_manifest = sample_manifest(dataset, background_indices, isolate_id_map)
        explainer_manifest = sample_manifest(dataset, explanation_indices, isolate_id_map)
        background_manifest.to_csv(background_path, index=False)
        explainer_manifest.to_csv(explainer_path, index=False)

    background_labels = background_manifest["label"].astype(int)
    print(
        f"{drug} fold {fold}: background = {len(background_indices)} "
        f"(R={int((background_labels == 0).sum())}, S={int((background_labels == 1).sum())}), "
        f"explainer = {len(explanation_indices)}"
    )

    selection_row = {
        "drug": drug,
        "fold": fold,
        "model_filename": selection["model_filename"],
        "best_val_auc": selection["best_val_auc"],
        "fold_selection_reason": selection["reason"],
        "background_count": len(background_indices),
        "background_label_0": int((background_labels == 0).sum()),
        "background_label_1": int((background_labels == 1).sum()),
        "explainer_count": len(explanation_indices),
    }

    run_explanation_indices = explanation_indices
    result_dir = fold_shap_dir
    if sharded:
        assert shard_index is not None
        run_explanation_indices = select_explainer_shard(
            explanation_indices, shard_index, shard_count
        )
        result_dir = fold_shap_dir / "shards" / f"shard_{shard_index:03d}_of_{shard_count:03d}"
        result_dir.mkdir(parents=True, exist_ok=True)
        sample_manifest(dataset, run_explanation_indices, isolate_id_map).to_csv(
            result_dir / "explainer_samples.csv", index=False
        )
        background_manifest.to_csv(result_dir / "background_samples.csv", index=False)
        selection_row["explainer_samples_in_shard"] = len(run_explanation_indices)
        selection_row["explainer_shard_index"] = shard_index
        selection_row["explainer_shard_total"] = shard_count
        print(
            f"{drug} fold {fold}: explainer shard {shard_index + 1}/{shard_count} has "
            f"{len(run_explanation_indices)} samples"
        )

    pd.DataFrame([selection_row]).to_csv(
        result_dir / "sample_selection_summary.csv", index=False
    )

    relevant = load_relevant_features(
        config["WHO_VCF_mapped_dir"],
        drug,
        bool(config.get("has_neg_strand", False)),
        map_mar_dir / f"relevant_features_{drug}.csv",
    )
    print(f"{drug}: found {len(relevant)} resistant WHO features")

    if config.get("prep_only"):
        print(
            f"{drug}: prep-only run complete "
            "(selected fold, cached dedup indices, background/explainer split, "
            "and relevant features)"
        )
        return None, None

    embeds, _ = dataset[0]
    model_dim, model_seq_len = embeds.shape
    print(f"{drug}: Evo2 input shape (D={model_dim}, L={model_seq_len})")

    k_values: Iterable[int] = config.get("k_values", K_VALUES)
    top_n_positions = int(config.get("top_n_positions", 100))
    shap_batch_size = int(config.get("shap_batch_size", 4))
    shap_background_batch_size = int(config.get("shap_background_batch_size", 16))
    save_shap_values = bool(config.get("save_shap_values", False))

    print(f"\n{drug} fold {fold}: loading {model_path}")
    model = load_fold_model(
        model_path, in_dim=model_dim, seq_len=model_seq_len, model_name=model_name
    )

    shap_df = compute_shap_for_model(
        model,
        dataset,
        background_indices=background_indices,
        explanation_indices=run_explanation_indices,
        per_gene_lengths=per_gene_len,
        gene_names=gene_names,
        batch_size=shap_batch_size,
        background_batch_size=shap_background_batch_size,
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
    pd.DataFrame({"feature": ranked_features}).to_csv(
        fold_shap_dir / "selected_features.csv", index=False
    )

    map_mar_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"drug": drug, "fold": fold, **metric_row}
        for metric_row in compute_map_mar_rows(ranked_features, relevant, k_values)
    ]
    map_mar_df = pd.DataFrame(rows).sort_values("k")
    map_mar_path = map_mar_dir / f"map_mar_{drug}.csv"
    map_mar_df.to_csv(map_mar_path, index=False)
    print(f"\nSaved MAP/MAR metrics to {map_mar_path}")
    return map_mar_path, None


def main() -> None:
    args = parse_arguments()
    run(
        load_config(
            args.parameter_file,
            args.drug,
            args.fold,
            args.prep_only,
            args.model_filename,
            args.explainer_shard_index,
            args.explainer_shard_count,
        )
    )


if __name__ == "__main__":
    main()
