"""Run Evo2 lineage-aware SHAP interpretability for every eligible held-out lineage.

This follows the same per-lineage workflow as
``Regression/interpretability/lineage_aware_split/run_interpret_logreg_lineage_aware.py``:
for a given drug, every held-out lineage that has both a saved model and a
completed test evaluation is treated as one "eligible" model. SHAP importance
is computed for that lineage's exact ``DNABERTCNN`` checkpoint, ranked
positions are mapped to WHO/VCF resistant features, and MAP/MAR metrics are
computed per lineage and then averaged.

The only real differences from the random-split Evo2 interpretability
(besides model loading) are input-data specific: the SHAP explainer runs over
Evo2 token memmaps instead of DNABERT-2 embeddings/logistic-regression
features, and a token position already equals the gapped-alignment nucleotide
position (Evo2 tokenizes 1 nucleotide == 1 token), so no tokenizer
offset-mapping step is required.

Usage::

    python run_interpret_evo2_lineage_aware.py parameter_files/shap_interpret_lineage_aware.yaml --drug STREPTOMYCIN

    ./run_all_drugs.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
EVO2_DIR = THIS_DIR.parents[1]
if str(EVO2_DIR) not in sys.path:
    sys.path.insert(0, str(EVO2_DIR))

from evo2_downstream.train import enforce_evo2_drug_index_mapping  # noqa: E402
from zero_shot.lineage_aware.lineage_split import (  # noqa: E402
    load_isolate_id_map,
    load_lineage_map,
    make_lineage_aware_split_fn,
)

from map_mar import (  # noqa: E402
    K_VALUES,
    compute_map_mar_rows,
    load_relevant_features,
    mean_map_mar_rows,
)
from shap_utils import (  # noqa: E402
    build_full_dataset,
    compute_shap_for_lineage_model,
    dedup_and_save_indices,
    load_or_create_fingerprints,
    load_lineage_model,
    rank_features_by_shap,
    sample_manifest,
    select_fixed_explainer_indices,
    select_lineage_background_indices,
)


# This must match the checkpoint eval_lineage_holdout.py actually loads
# (--saved_model_name, default "DNABERTCNN") to produce test_set_auc_<DRUG>.csv --
# NOT an intermediate early-stopping checkpoint name.
DEFAULT_MODEL_FILENAME = "DNABERTCNN.pt"
DEFAULT_MODEL_SEED = "42"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_file", help="YAML configuration file")
    parser.add_argument(
        "--drug",
        help="Override the drug in the YAML file (useful for the all-drug runner)",
    )
    parser.add_argument(
        "--heldout-lineage",
        dest="heldout_lineage",
        default=None,
        help="Restrict processing to a single eligible held-out lineage "
        "(for running lineages as parallel SLURM array tasks)",
    )
    parser.add_argument(
        "--model-filename",
        default=None,
        help="Override the checkpoint filename in the YAML file",
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
        help="Total number of explainer shards for this lineage",
    )
    parser.add_argument(
        "--prep-only",
        dest="prep_only",
        action="store_true",
        help="Build the dataset and cache dedup indices, the fixed explainer set, "
        "and relevant features, then exit (no SHAP). Run once before launching "
        "parallel per-lineage array jobs.",
    )
    return parser.parse_args()


def load_config(
    parameter_file: str | Path,
    drug_override: str | None = None,
    heldout_lineage_override: str | None = None,
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
    if heldout_lineage_override is not None:
        config["heldout_lineage"] = heldout_lineage_override
    if model_filename_override is not None:
        config["model_filename"] = model_filename_override
    if explainer_shard_index is not None:
        config["explainer_shard_index"] = explainer_shard_index
    if explainer_shard_count is not None:
        config["explainer_shard_count"] = explainer_shard_count
    config["prep_only"] = bool(prep_only)
    return config


def discover_eligible_lineages(
    model_dir: str | Path,
    drug: str,
    embed_type: str,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    model_seed: str = DEFAULT_MODEL_SEED,
) -> list[tuple[str, str, Path]]:
    """Return ``(lineage, seed, model_path)`` for every held-out lineage that
    has both a saved model checkpoint and a completed test evaluation at
    ``model_seed`` (the exact seed used by ``eval_lineage_holdout.py`` to
    produce ``test_set_auc_<DRUG>.csv``).
    """
    drug_root = Path(model_dir) / drug
    saved_models_dir = drug_root / "saved_models" / "evo2" / embed_type
    classification_dir = drug_root / "classification_results" / "evo2" / embed_type
    if not saved_models_dir.is_dir():
        raise FileNotFoundError(f"Training output for {drug} not found: {saved_models_dir}")

    eligible: list[tuple[str, str, Path]] = []
    for lineage_dir in sorted(saved_models_dir.glob("heldout_lineage_*")):
        if not lineage_dir.is_dir():
            continue
        lineage = lineage_dir.name.removeprefix("heldout_lineage_")
        seed_dir = lineage_dir / drug / f"seed_{model_seed}"
        if not seed_dir.is_dir():
            continue

        model_path = seed_dir / model_filename
        test_auc_path = (
            classification_dir
            / f"heldout_lineage_{lineage}"
            / drug
            / f"seed_{model_seed}"
            / f"test_set_auc_{drug}.csv"
        )
        if model_path.is_file() and test_auc_path.is_file():
            eligible.append((lineage, model_seed, model_path))
        elif model_path.is_file():
            print(
                f"[warn] Ignoring lineage {lineage} seed {model_seed}: model exists but "
                f"{test_auc_path.name} is missing"
            )

    def lineage_sort_key(item: tuple[str, str, Path]) -> tuple[int, int | str]:
        lineage = item[0]
        return (0, int(lineage)) if lineage.isdigit() else (1, lineage)

    eligible.sort(key=lineage_sort_key)
    if not eligible:
        raise ValueError(
            f"No eligible held-out lineages for {drug} at seed {model_seed}; expected both "
            f"{model_filename} and test_set_auc_{drug}.csv below {saved_models_dir}"
        )
    return eligible


def reconstruct_model_training_indices(
    nonheldout_indices: Iterable[int],
    *,
    seed: int,
    validation_fraction: float = 0.2,
) -> list[int]:
    """Reproduce the inner split used by lineage-aware Evo2 training."""
    nonheldout_indices = [int(index) for index in nonheldout_indices]
    if len(nonheldout_indices) < 2:
        raise ValueError("At least two nonheldout samples are required")
    validation_count = max(1, int(len(nonheldout_indices) * validation_fraction))
    training_count = len(nonheldout_indices) - validation_count
    generator = torch.Generator().manual_seed(seed)
    shuffled_positions = torch.randperm(
        len(nonheldout_indices), generator=generator
    ).tolist()
    return [nonheldout_indices[position] for position in shuffled_positions[:training_count]]


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


def run(config: dict[str, Any]) -> tuple[Path | None, Path | None]:
    drug = config["drug"]
    embed_type = config.get("embed_type", "token")
    model_name = config.get("model_name", "DNABERTCNN")
    model_filename = config.get("model_filename", DEFAULT_MODEL_FILENAME)
    model_seed = str(config.get("model_seed", DEFAULT_MODEL_SEED))
    heldout_lineage_filter = config.get("heldout_lineage")
    if heldout_lineage_filter is not None:
        heldout_lineage_filter = str(heldout_lineage_filter)
    shard_count = int(config.get("explainer_shard_count", 1))
    shard_index_value = config.get("explainer_shard_index")
    shard_index = int(shard_index_value) if shard_index_value is not None else None
    sharded = shard_count > 1
    if sharded and (heldout_lineage_filter is None or shard_index is None):
        raise ValueError(
            "Explainer sharding requires --heldout-lineage, "
            "--explainer-shard-index, and --explainer-shard-count"
        )
    if not sharded and shard_index not in (None, 0):
        raise ValueError("A nonzero explainer shard index requires shard count > 1")

    enforce_evo2_drug_index_mapping(config["phenotype_label_path"])

    eligible = discover_eligible_lineages(
        config["model_dir"], drug, embed_type, model_filename, model_seed
    )
    print(f"{drug}: eligible held-out lineages (seed {model_seed}) = {[lineage for lineage, _, _ in eligible]}")

    if heldout_lineage_filter is not None:
        eligible = [item for item in eligible if item[0] == heldout_lineage_filter]
        if not eligible:
            raise ValueError(
                f"Requested --heldout-lineage {heldout_lineage_filter} is not eligible for {drug} "
                f"(seed {model_seed})"
            )

    output_dir = Path(config["output_dir"])
    map_mar_dir = output_dir / "map_mar"
    dedup_dir = output_dir / "dedup_geno_data"

    dataset, label_map, per_gene_len, gene_names = build_full_dataset(
        drug, embed_type, config["memmap_dir"], config["phenotype_label_path"]
    )
    print(f"{drug}: dataset has {len(dataset)} labelled samples across genes {gene_names}")

    fingerprints = load_or_create_fingerprints(
        dataset, f"{drug}_full", dedup_dir
    )
    dedup_indices = dedup_and_save_indices(
        dataset, f"{drug}_full", dedup_dir, fingerprints=fingerprints
    )
    print(f"{drug}: {len(dedup_indices)} deduplicated samples available for SHAP")

    background_frac = float(config.get("background_frac", 0.2))
    max_background = int(config.get("max_background", 160))
    max_explain = int(config.get("max_explain", 360))
    seed = int(config.get("random_seed", 42))
    explanation_indices = select_fixed_explainer_indices(
        dataset,
        dedup_indices,
        background_fraction=background_frac,
        max_background=max_background,
        max_explain=max_explain,
        seed=seed,
    )

    isolate_id_map = load_isolate_id_map(config["geno_pheno_csv"])
    lineage_map = load_lineage_map(config["lineage_csv"])
    drug_shap_dir = output_dir / "shap_values" / drug
    drug_shap_dir.mkdir(parents=True, exist_ok=True)
    explainer_manifest = sample_manifest(
        dataset, explanation_indices, isolate_id_map
    )
    explainer_path = drug_shap_dir / "explainer_samples.csv"
    if heldout_lineage_filter is None or not explainer_path.exists():
        explainer_manifest.to_csv(explainer_path, index=False)
    else:
        existing_explainer = pd.read_csv(
            explainer_path, dtype={"row_id": str, "sample_id": str}
        )
        required_columns = {"row_id", "sample_id", "label"}
        if not required_columns.issubset(existing_explainer.columns):
            raise ValueError(f"Invalid fixed explainer manifest: {explainer_path}")
        expected_pairs = list(
            zip(
                explainer_manifest["sample_id"].astype(str),
                explainer_manifest["label"].astype(int),
            )
        )
        existing_pairs = list(
            zip(
                existing_explainer["sample_id"].astype(str),
                existing_explainer["label"].astype(int),
            )
        )
        if existing_pairs != expected_pairs:
            raise ValueError(
                f"Existing fixed explainer manifest does not match seed/config: "
                f"{explainer_path}. Run --prep-only before lineage array jobs."
            )
    print(f"{drug}: fixed explainer samples = {len(explanation_indices)}")
    run_explanation_indices = explanation_indices
    if sharded:
        assert shard_index is not None
        run_explanation_indices = select_explainer_shard(
            explanation_indices, shard_index, shard_count
        )
        print(
            f"{drug}: explainer shard {shard_index + 1}/{shard_count} has "
            f"{len(run_explanation_indices)} samples"
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
            "(cached dedup indices + fixed explainer + relevant features)"
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

    lineage_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for lineage, seed_used, model_path in eligible:
        print(f"\n{drug} lineage {lineage}: loading {model_path}")
        model = load_lineage_model(
            model_path, in_dim=model_dim, seq_len=model_seq_len, model_name=model_name
        )

        lineage_shap_dir = output_dir / "shap_values" / drug / f"heldout_lineage_{lineage}"
        lineage_shap_dir.mkdir(parents=True, exist_ok=True)

        split_fn = make_lineage_aware_split_fn(
            lineage,
            isolate_id_map,
            lineage_map,
            min_class_count=0,
        )
        nonheldout_indices, _test_indices, _y_train, _y_test = split_fn(
            dataset, label_map, seed=seed
        )
        training_indices = reconstruct_model_training_indices(
            nonheldout_indices,
            seed=int(seed_used),
        )
        background_indices, candidate_count = select_lineage_background_indices(
            dataset,
            training_indices,
            explanation_indices,
            max_background=max_background,
            seed=seed,
            fingerprints=fingerprints,
        )
        result_dir = lineage_shap_dir
        if sharded:
            assert shard_index is not None
            result_dir = (
                lineage_shap_dir
                / "shards"
                / f"shard_{shard_index:03d}_of_{shard_count:03d}"
            )
            result_dir.mkdir(parents=True, exist_ok=True)
        background_manifest = sample_manifest(
            dataset, background_indices, isolate_id_map
        )
        background_manifest.to_csv(
            result_dir / "background_samples.csv", index=False
        )
        if sharded:
            sample_manifest(
                dataset, run_explanation_indices, isolate_id_map
            ).to_csv(result_dir / "explainer_samples.csv", index=False)
        background_labels = background_manifest["label"].astype(int)
        selection_row = {
            "drug": drug,
            "heldout_lineage": lineage,
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
        pd.DataFrame([selection_row]).to_csv(
            result_dir / "sample_selection_summary.csv", index=False
        )
        print(
            f"{drug} lineage {lineage}: unique training background candidates = "
            f"{candidate_count}, selected = {len(background_indices)}"
        )

        shap_df = compute_shap_for_lineage_model(
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
            lineage_shap_dir / "selected_features.csv", index=False
        )

        for metric_row in compute_map_mar_rows(ranked_features, relevant, k_values):
            lineage_rows.append(
                {
                    "drug": drug,
                    "heldout_lineage": lineage,
                    **metric_row,
                }
            )

    by_lineage = pd.DataFrame(lineage_rows).sort_values(["heldout_lineage", "k"])
    map_mar_dir.mkdir(parents=True, exist_ok=True)

    if heldout_lineage_filter is not None:
        # Running as one of several parallel per-lineage array tasks: write to a
        # lineage-specific file so concurrent tasks never clobber each other or
        # the shared *_by_lineage.csv / *_mean.csv files. Combine afterwards with
        # combine_lineage_results.py once every lineage has finished.
        lineage_path = map_mar_dir / f"map_mar_{drug}_heldout_lineage_{heldout_lineage_filter}.csv"
        by_lineage.to_csv(lineage_path, index=False)
        print(f"\nSaved lineage-level metrics to {lineage_path}")
        return lineage_path, None

    pd.DataFrame(selection_rows).to_csv(
        drug_shap_dir / "sample_selection_summary.csv", index=False
    )
    mean_rows = mean_map_mar_rows(by_lineage)
    mean_rows.insert(0, "drug", drug)

    by_lineage_path = map_mar_dir / f"map_mar_{drug}_by_lineage.csv"
    mean_path = map_mar_dir / f"map_mar_{drug}_mean.csv"
    by_lineage.to_csv(by_lineage_path, index=False)
    mean_rows.to_csv(mean_path, index=False)
    print(f"\nSaved lineage-level metrics to {by_lineage_path}")
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
            args.model_filename,
            args.explainer_shard_index,
            args.explainer_shard_count,
        )
    )


if __name__ == "__main__":
    main()
