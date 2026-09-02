"""Run SD-CNN interpretability for every eligible lineage of one drug.

This mirrors the random-split SHAP/MAP-MAR workflow.  Unlike random split, each
held-out lineage is interpreted with the finalized model that generated that
lineage's test predictions.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
import yaml


THIS_DIR = Path(__file__).resolve().parent
INTERPRETABILITY_DIR = THIS_DIR.parent

# Resolve this directory's MAP/MAR module before exposing the random-split
# directory, which contains another module with the same name.
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from map_mar import (  # noqa: E402
    K_VALUES,
    compute_map_mar_rows,
    load_relevant_features,
    mean_map_mar_rows,
)

if str(INTERPRETABILITY_DIR) not in sys.path:
    sys.path.insert(1, str(INTERPRETABILITY_DIR))
from parameters.locus_order import DRUG_TO_LOCI  # noqa: E402
from tb_cnn_codebase import (  # noqa: E402
    create_X,
    load_combined_geno_pheno,
    make_geno_pheno_dataset,
    masked_multi_weighted_bce,
    masked_weighted_accuracy,
    rs_encoding_to_numeric,
)


DEFAULT_MODEL_FILENAME = "sd-cnn_model_lineage_holdout.h5"
ISONIAZID_OUTPUT_ALIASES = ("INH", "ISONIAZID")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_file", help="Lineage interpretability YAML file")
    parser.add_argument(
        "--drug", help="Override the configured drug (used by run_all_drugs.sh)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute SHAP summaries even when a matching cache exists",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_config(
    parameter_file: str | Path, drug_override: str | None = None
) -> dict[str, Any]:
    config = load_yaml(parameter_file)
    if drug_override:
        config["drug"] = drug_override
    config["drug"] = str(config["drug"]).upper()
    return config


def newest_model_mtime(directory: Path, model_filename: str) -> float:
    """Return the newest finalized lineage-model time below a drug folder."""
    models_found = directory.glob(
        f"heldout_lineage_*/saved_model/{model_filename}"
    )
    return max((path.stat().st_mtime for path in models_found), default=-1.0)


def resolve_drug_training_dir(
    training_output_dir: str | Path,
    drug: str,
    model_filename: str = DEFAULT_MODEL_FILENAME,
) -> Path:
    """Resolve the output folder, choosing the newest INH/ISONIAZID run."""
    root = Path(training_output_dir)
    names = ISONIAZID_OUTPUT_ALIASES if drug == "ISONIAZID" else (drug,)
    candidates = [root / name for name in names if (root / name).is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No lineage-aware training output found for {drug} below {root}"
        )

    candidates = [
        path for path in candidates if newest_model_mtime(path, model_filename) >= 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No finalized lineage models for {drug} were found below {root}"
        )
    selected = max(candidates, key=lambda path: newest_model_mtime(path, model_filename))
    if len(candidates) > 1:
        details = ", ".join(
            f"{path.name}={newest_model_mtime(path, model_filename):.0f}"
            for path in candidates
        )
        print(f"{drug}: available output aliases ({details}); selected {selected.name}")
    return selected


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def discover_eligible_lineages(
    eligibility_file: str | Path,
    drug: str,
    drug_training_dir: Path,
    model_filename: str = DEFAULT_MODEL_FILENAME,
) -> list[tuple[str, Path]]:
    """Return feasible lineages and the exact finalized models used for testing."""
    eligibility_path = Path(eligibility_file)
    if not eligibility_path.is_file():
        raise FileNotFoundError(f"Eligibility CSV not found: {eligibility_path}")

    counts = pd.read_csv(eligibility_path, dtype={"heldout_lineage": str})
    required = {"drug", "heldout_lineage", "feasible"}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(
            f"Eligibility CSV is missing columns: {sorted(missing)}"
        )

    drug_rows = counts[counts["drug"].astype(str).str.upper() == drug]
    lineage_names = drug_rows.loc[
        drug_rows["feasible"].map(_as_bool), "heldout_lineage"
    ].astype(str)
    if lineage_names.empty:
        raise ValueError(f"No eligible held-out lineages are recorded for {drug}")

    eligible: list[tuple[str, Path]] = []
    missing_artifacts: list[str] = []
    for lineage in lineage_names:
        lineage = lineage.removesuffix(".0")
        lineage_dir = drug_training_dir / f"heldout_lineage_{lineage}"
        model_path = lineage_dir / "saved_model" / model_filename
        prediction_path = lineage_dir / "test_predictions.csv"
        if not model_path.is_file() or not prediction_path.is_file():
            missing_artifacts.append(
                f"lineage {lineage}: model={model_path.is_file()}, "
                f"test_predictions={prediction_path.is_file()}"
            )
            continue
        eligible.append((lineage, model_path))

    if missing_artifacts:
        raise FileNotFoundError(
            f"Eligible {drug} lineages are missing required test artifacts under "
            f"{drug_training_dir}: " + "; ".join(missing_artifacts)
        )

    def sort_key(item: tuple[str, Path]) -> tuple[int, int | str]:
        return (0, int(item[0])) if item[0].isdigit() else (1, item[0])

    return sorted(eligible, key=sort_key)


def find_training_config(training_config_dir: str | Path, drug: str) -> Path:
    """Find the lineage-training YAML whose declared drug matches ``drug``."""
    matches: list[Path] = []
    for config_path in sorted(Path(training_config_dir).glob("*_lineage_holdout.yaml")):
        candidate = load_yaml(config_path)
        if str(candidate.get("drug", "")).upper() == drug:
            matches.append(config_path)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one lineage training config for {drug}, found {matches}"
        )
    return matches[0]


def load_and_filter_data(
    training_config: dict[str, Any], dedup_output_dir: str | Path
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load, filter, and deduplicate data as in random-split interpretability."""
    metadata_path = Path(training_config["metadata_path"])
    h5_path = Path(training_config["h5_path"])
    if not metadata_path.is_file() or not h5_path.is_file():
        print("Creating missing genotype/phenotype parquet and HDF5 data")
        make_geno_pheno_dataset(**training_config)

    frame = load_combined_geno_pheno(**training_config)
    drug = str(training_config["drug"]).upper()
    sparse_path = Path(training_config["X_sparse_path"])
    if sparse_path.is_file():
        print(f"Loading cached input tensor from {sparse_path}")
        x_sparse = sparse.load_npz(sparse_path)
        if x_sparse.shape[0] != len(frame):
            raise ValueError(
                f"Cached X row count ({x_sparse.shape[0]}) does not match data ({len(frame)})"
            )
    else:
        print(f"Creating input tensor for {drug}")
        x_sparse = sparse.COO(create_X(frame, drug))
        sparse_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(sparse_path, x_sparse)

    y_df, _ = rs_encoding_to_numeric(frame, drug)
    labels = y_df.values.astype(int).reshape(-1)
    valid_mask = labels != -1
    x_valid = x_sparse[valid_mask].todense()
    y_valid = labels[valid_mask]
    frame_valid = frame.loc[valid_mask].reset_index(drop=True)

    dedup_path = Path(dedup_output_dir) / f"{drug}_full_dedup_indices.npy"
    if dedup_path.is_file():
        unique_indices = np.load(dedup_path)
        valid_cache = (
            unique_indices.ndim == 1
            and (len(unique_indices) == 0 or int(unique_indices.max()) < len(x_valid))
        )
        if not valid_cache:
            print(f"[warn] Ignoring stale deduplication cache {dedup_path}")
            unique_indices = _deduplicate_indices(x_valid, y_valid, dedup_path)
        else:
            print(
                f"Loaded {len(unique_indices)} deduplicated rows from {dedup_path}"
            )
    else:
        unique_indices = _deduplicate_indices(x_valid, y_valid, dedup_path)

    return (
        x_valid[unique_indices],
        y_valid[unique_indices].reshape(-1, 1),
        frame_valid.iloc[unique_indices].reset_index(drop=True),
    )


def _deduplicate_indices(
    X: np.ndarray, y: np.ndarray, output_path: Path
) -> np.ndarray:
    seen: set[tuple[bytes, int]] = set()
    indices: list[int] = []
    for index in range(len(X)):
        key = (X[index].tobytes(), int(y[index]))
        if key not in seen:
            seen.add(key)
            indices.append(index)
    result = np.asarray(indices, dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)
    print(f"Deduplicated {len(X)} -> {len(result)} rows and saved {output_path}")
    return result


def build_feature_names(frame: pd.DataFrame, drug: str) -> list[str]:
    """Build names in the same locus order used by ``create_X``."""
    available = [column for column in frame.columns if column.endswith("_one_hot")]
    selected: list[str] = []
    for locus in DRUG_TO_LOCI[drug]:
        candidates = [column for column in available if locus.lower() in column.lower()]
        selected.extend(candidates)
    if not selected:
        raise ValueError(f"No one-hot feature columns found for {drug}")

    longest = max(frame.iloc[0][column].shape[0] for column in selected)
    names: list[str] = []
    for column in selected:
        gene = column.replace("_one_hot", "").replace(".fasta", "")
        names.extend(f"{gene}_{position}" for position in range(longest))
    return names


def select_background_and_explanation(
    y: np.ndarray,
    background_fraction: float,
    max_background: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the random-split stratified SHAP sample selection."""
    labels = np.asarray(y).reshape(-1).astype(int)
    indices = np.arange(len(labels))
    if len(indices) < 2:
        raise ValueError("At least two samples are required for SHAP")
    if not 0 < background_fraction < 1:
        raise ValueError("shap_background_fraction must be between zero and one")
    if max_background < 1:
        raise ValueError("shap_max_background must be positive")

    counts = pd.Series(labels).value_counts()
    can_stratify = len(counts) > 1 and int(counts.min()) >= 2
    if can_stratify:
        background, explanation = train_test_split(
            indices,
            train_size=background_fraction,
            stratify=labels,
            random_state=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        size = min(len(indices) - 1, max(1, round(background_fraction * len(indices))))
        background, explanation = shuffled[:size], shuffled[size:]

    if len(background) > max_background:
        background_labels = labels[background]
        background_counts = pd.Series(background_labels).value_counts()
        if len(background_counts) > 1 and int(background_counts.min()) >= 2:
            background, _ = train_test_split(
                background,
                train_size=max_background,
                stratify=background_labels,
                random_state=seed,
            )
        else:
            background = np.random.default_rng(seed).choice(
                background, size=max_background, replace=False
            )
        mask = np.ones(len(indices), dtype=bool)
        mask[background] = False
        explanation = indices[mask]

    return np.asarray(background), np.asarray(explanation)


def collapse_shap_values(values: Any, feature_count: int) -> np.ndarray:
    """Collapse base channels and flatten loci exactly as the random workflow."""
    if isinstance(values, (list, tuple)):
        if len(values) != 1:
            raise ValueError(f"Expected one model output, got {len(values)}")
        values = values[0]
    array = np.asarray(values)
    if array.ndim == 5 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 4:
        raise ValueError(
            "Expected SHAP shape (samples, bases, positions, loci[, 1]); "
            f"got {array.shape}"
        )
    collapsed = np.abs(array).sum(axis=1).transpose(0, 2, 1)
    flattened = collapsed.reshape(collapsed.shape[0], -1)
    if flattened.shape[1] != feature_count:
        raise ValueError(
            f"Collapsed SHAP has {flattened.shape[1]} features; expected {feature_count}"
        )
    return flattened


def model_cache_signature(
    model_path: Path, X: np.ndarray, feature_names: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    stat = model_path.stat()
    return {
        "model_path": str(model_path.resolve()),
        "model_size": stat.st_size,
        "model_mtime_ns": stat.st_mtime_ns,
        "data_shape": list(X.shape),
        "feature_count": len(feature_names),
        "background_fraction": float(config.get("shap_background_fraction", 0.2)),
        "max_background": int(config.get("shap_max_background", 160)),
        "random_seed": int(config.get("random_seed", 42)),
        "save_shap_values": _as_bool(config.get("save_shap_values", False)),
    }


def compute_mean_abs_shap(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    config: dict[str, Any],
) -> pd.Series:
    """Compute mean absolute collapsed SHAP values in bounded-memory batches."""
    batch_size = int(config.get("shap_batch_size", 128))
    if batch_size < 1:
        raise ValueError("shap_batch_size must be positive")

    background, explanation = select_background_and_explanation(
        y,
        float(config.get("shap_background_fraction", 0.2)),
        int(config.get("shap_max_background", 160)),
        int(config.get("random_seed", 42)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "background_indices.npy", background)
    np.save(output_dir / "explanation_indices.npy", explanation)
    np.save(output_dir / "explained_labels.npy", np.asarray(y).reshape(-1)[explanation])

    explainer = shap.DeepExplainer(model, X[background])
    save_values = _as_bool(config.get("save_shap_values", False))
    shap_memmap = None
    shap_path = output_dir / "shap_values.npy"
    if save_values:
        shap_memmap = np.lib.format.open_memmap(
            shap_path,
            mode="w+",
            dtype=np.float64,
            shape=(len(explanation), len(feature_names)),
        )

    absolute_sum = np.zeros(len(feature_names), dtype=np.float64)
    for start in range(0, len(explanation), batch_size):
        stop = min(start + batch_size, len(explanation))
        print(f"  SHAP samples {start + 1}-{stop} of {len(explanation)}")
        values = collapse_shap_values(
            explainer.shap_values(X[explanation[start:stop]]), len(feature_names)
        )
        absolute_sum += values.sum(axis=0)
        if shap_memmap is not None:
            shap_memmap[start:stop] = values

    if shap_memmap is not None:
        shap_memmap.flush()
        del shap_memmap
    elif shap_path.exists():
        shap_path.unlink()

    return pd.Series(
        absolute_sum / len(explanation),
        index=feature_names,
        name="mean_abs_shap",
    ).sort_values(ascending=False)


def rank_selected_features(
    mean_abs_shap: pd.Series,
    importance_fraction_threshold: float,
    features_below_threshold: int,
) -> list[str]:
    """Apply the random-split 20%-plus-next-10 feature selection rule."""
    total = float(mean_abs_shap.sum())
    if total <= 0:
        raise ValueError("All mean absolute SHAP values are zero")
    above = mean_abs_shap / total > importance_fraction_threshold
    selected = mean_abs_shap.index[above].tolist()
    selected.extend(mean_abs_shap.index[~above][:features_below_threshold].tolist())
    return [str(feature) for feature in selected]


def save_shap_plot(mean_abs_shap: pd.Series, output_path: Path) -> None:
    """Save the random-split-style SHAP importance bar plot."""
    explanation = shap.Explanation(
        values=mean_abs_shap.to_numpy(),
        feature_names=mean_abs_shap.index.tolist(),
    )
    plt.figure()
    shap.plots.bar(explanation, show=False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def load_or_compute_lineage_importance(
    model_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    config: dict[str, Any],
    force: bool,
) -> pd.Series:
    signature = model_cache_signature(model_path, X, feature_names, config)
    signature_path = output_dir / "cache_metadata.json"
    importance_path = output_dir / "mean_abs_shap.csv"
    if not force and signature_path.is_file() and importance_path.is_file():
        with signature_path.open("r", encoding="utf-8") as handle:
            cached_signature = json.load(handle)
        if cached_signature == signature:
            print(f"Loading matching SHAP summary cache from {importance_path}")
            return pd.read_csv(importance_path, index_col="feature")[
                "mean_abs_shap"
            ].sort_values(ascending=False)

    model = models.load_model(
        model_path,
        custom_objects={
            "masked_weighted_accuracy": masked_weighted_accuracy,
            "masked_multi_weighted_bce": masked_multi_weighted_bce,
        },
    )
    expected_shape = tuple(model.input_shape[1:])
    if expected_shape != tuple(X.shape[1:]):
        raise ValueError(
            f"Model {model_path} expects {expected_shape}, but data has {X.shape[1:]}"
        )

    mean_abs = compute_mean_abs_shap(
        model, X, y, feature_names, output_dir, config
    )
    mean_abs.rename_axis("feature").to_csv(importance_path)
    save_shap_plot(mean_abs, output_dir / "shap_summary.png")
    with signature_path.open("w", encoding="utf-8") as handle:
        json.dump(signature, handle, indent=2, sort_keys=True)
        handle.write("\n")
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return mean_abs


def run(config: dict[str, Any], force: bool = False) -> tuple[Path, Path]:
    drug = config["drug"]
    model_filename = str(config.get("model_filename", DEFAULT_MODEL_FILENAME))
    drug_training_dir = resolve_drug_training_dir(
        config["training_output_dir"], drug, model_filename
    )
    eligible = discover_eligible_lineages(
        config["eligibility_file"],
        drug,
        drug_training_dir,
        model_filename,
    )
    print(
        f"{drug}: eligible lineages = {[lineage for lineage, _ in eligible]}; "
        f"training output = {drug_training_dir}"
    )

    training_config_path = find_training_config(config["training_config_dir"], drug)
    training_config = load_yaml(training_config_path)
    output_root = Path(config["output_dir"])
    X, y, frame = load_and_filter_data(
        training_config, output_root / "dedup_geno_data"
    )
    feature_names = build_feature_names(frame, drug)
    if X.shape[2] * X.shape[3] != len(feature_names):
        raise ValueError(
            f"Input has {X.shape[2] * X.shape[3]} flattened features but "
            f"{len(feature_names)} feature names were built"
        )
    print(f"{drug}: interpreting {len(X)} deduplicated labelled isolates")

    map_mar_dir = output_root / "map_mar"
    relevant = load_relevant_features(
        config["WHO_VCF_mapped_dir"],
        drug,
        _as_bool(config.get("has_neg_strand", False)),
        map_mar_dir / f"relevant_features_{drug}.csv",
    )
    print(f"{drug}: found {len(relevant)} resistant WHO features")

    k_values: Iterable[int] = config.get("k_values", K_VALUES)
    lineage_rows: list[dict[str, Any]] = []
    for lineage, model_path in eligible:
        print(f"{drug} lineage {lineage}: interpreting {model_path}")
        lineage_output = (
            output_root / "shap_values" / drug / f"heldout_lineage_{lineage}"
        )
        mean_abs = load_or_compute_lineage_importance(
            model_path,
            X,
            y,
            feature_names,
            lineage_output,
            config,
            force,
        )
        ranked = rank_selected_features(
            mean_abs,
            float(config.get("importance_fraction_threshold", 0.2)),
            int(config.get("features_below_threshold", 10)),
        )
        pd.DataFrame({"feature": ranked}).to_csv(
            lineage_output / "selected_features.csv", index=False
        )
        for metric_row in compute_map_mar_rows(ranked, relevant, k_values):
            lineage_rows.append(
                {"drug": drug, "heldout_lineage": lineage, **metric_row}
            )

    by_lineage = pd.DataFrame(lineage_rows).sort_values(
        ["heldout_lineage", "k"]
    )
    mean_rows = mean_map_mar_rows(by_lineage)
    mean_rows.insert(0, "drug", drug)

    map_mar_dir.mkdir(parents=True, exist_ok=True)
    by_lineage_path = map_mar_dir / f"map_mar_{drug}_by_lineage.csv"
    mean_path = map_mar_dir / f"map_mar_{drug}_mean.csv"
    by_lineage.to_csv(by_lineage_path, index=False)
    mean_rows.to_csv(mean_path, index=False)
    print(f"Saved lineage-level MAP/MAR metrics to {by_lineage_path}")
    print(f"Saved mean MAP/MAR metrics to {mean_path}")
    return by_lineage_path, mean_path


def main() -> None:
    args = parse_arguments()
    run(load_config(args.parameter_file, args.drug), force=args.force)


if __name__ == "__main__":
    main()
