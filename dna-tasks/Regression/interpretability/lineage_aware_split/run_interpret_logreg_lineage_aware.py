"""Run regression interpretability for every eligible held-out lineage of a drug.

This follows the random-split SHAP and MAP/MAR workflow, but loads the exact
``LogisticRegressionCV.model`` fitted for each lineage-aware test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
import yaml


THIS_DIR = Path(__file__).resolve().parent
INTERPRETABILITY_DIR = THIS_DIR.parent

# Import the lineage-aware metric module before adding the parent directory to
# sys.path (the parent also contains the random-split module named map_mar).
from map_mar import (  # noqa: E402
    K_VALUES,
    compute_map_mar_rows,
    load_relevant_features,
    mean_map_mar_rows,
)

if str(INTERPRETABILITY_DIR) not in sys.path:
    sys.path.insert(0, str(INTERPRETABILITY_DIR))

from parameters.locus_order import DRUG_TO_LOCI  # noqa: E402


DEFAULT_MODEL_FILENAME = "LogisticRegressionCV.model"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_file", help="YAML configuration file")
    parser.add_argument(
        "--drug",
        help="Override the drug in the YAML file (useful for the all-drug runner)",
    )
    return parser.parse_args()


def load_config(parameter_file: str | Path, drug_override: str | None = None) -> dict[str, Any]:
    with Path(parameter_file).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {parameter_file}")
    if drug_override:
        config["drug"] = drug_override
    config["drug"] = str(config["drug"]).upper()
    return config


def discover_eligible_lineages(
    model_dir: str | Path,
    drug: str,
    model_filename: str = DEFAULT_MODEL_FILENAME,
) -> list[tuple[str, Path]]:
    """Return held-out lineages that have the model used for test evaluation."""
    drug_dir = Path(model_dir) / drug
    if not drug_dir.is_dir():
        raise FileNotFoundError(f"Training output for {drug} not found: {drug_dir}")

    eligible: list[tuple[str, Path]] = []
    for lineage_dir in drug_dir.glob("heldout_lineage_*"):
        if not lineage_dir.is_dir():
            continue
        lineage = lineage_dir.name.removeprefix("heldout_lineage_")
        model_path = lineage_dir / "saved_models" / model_filename
        test_predictions = lineage_dir / "test_predictions.csv"
        if model_path.is_file() and test_predictions.is_file():
            eligible.append((lineage, model_path))
        elif model_path.is_file():
            print(
                f"[warn] Ignoring lineage {lineage}: model exists but test_predictions.csv is missing"
            )

    def lineage_sort_key(item: tuple[str, Path]) -> tuple[int, int | str]:
        lineage = item[0]
        return (0, int(lineage)) if lineage.isdigit() else (1, lineage)

    eligible.sort(key=lineage_sort_key)
    if not eligible:
        raise ValueError(
            f"No eligible held-out lineages for {drug}; expected both {model_filename} "
            f"and test_predictions.csv below {drug_dir}/heldout_lineage_*"
        )
    return eligible


def read_genotype_columns(genotype_sites_file: str | Path, drug: str) -> list[str]:
    if drug not in DRUG_TO_LOCI:
        raise ValueError(f"No locus definition is available for drug {drug}")
    genotypes_df = pd.read_csv(genotype_sites_file, index_col=0)
    selected_loci = {f"/{gene}" for gene in DRUG_TO_LOCI[drug]}
    drug_genotypes = genotypes_df[genotypes_df["locus"].isin(selected_loci)]
    columns = [
        f"{locus}_{site}"
        for locus, site in zip(drug_genotypes["locus"], drug_genotypes["sites"])
    ]
    if not columns:
        raise ValueError(f"No genotype columns were selected for {drug}")
    return columns


def deduplicate_indices(
    X: pd.DataFrame,
    y: pd.Series,
    cache_path: str | Path,
) -> np.ndarray:
    """Deduplicate (genotype, phenotype) pairs as in the random-split workflow."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        indices = np.load(cache_path)
        if len(indices) > 0 and int(indices.max()) < len(X):
            print(f"Loaded {len(indices)} cached deduplicated rows from {cache_path}")
            return indices
        print(f"[warn] Rebuilding stale deduplication cache {cache_path}")

    X_values = X.to_numpy()
    y_values = y.to_numpy()
    seen: set[tuple[bytes, object]] = set()
    unique_indices: list[int] = []
    for index in range(len(X_values)):
        y_value = y_values[index]
        if pd.isna(y_value):
            normalized_y: object = "NaN"
        else:
            normalized_y = y_value.item() if hasattr(y_value, "item") else y_value
        key = (X_values[index].tobytes(), normalized_y)
        if key not in seen:
            seen.add(key)
            unique_indices.append(index)

    indices = np.asarray(unique_indices, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, indices)
    print(f"Deduplicated {len(X)} -> {len(indices)} rows and saved {cache_path}")
    return indices


def prepare_interpretability_data(
    input_data_file: str | Path,
    genotype_columns: list[str],
    drug: str,
    dedup_cache: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    input_df = pd.read_csv(input_data_file, index_col=0, low_memory=False)
    missing = sorted(set(genotype_columns + [drug]) - set(input_df.columns))
    if missing:
        raise ValueError(f"Input data is missing columns: {missing[:10]}")

    indices = deduplicate_indices(input_df[genotype_columns], input_df[drug], dedup_cache)
    deduplicated = input_df.iloc[indices]
    valid = deduplicated.dropna(subset=[drug])
    X = valid[genotype_columns]
    y = pd.to_numeric(valid[drug], errors="raise").astype(int)
    if len(X) < 2:
        raise ValueError(f"Not enough non-missing samples for {drug}")
    return X, y


def select_background_indices(
    y: pd.Series,
    background_fraction: float,
    max_background: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the stratified background/explanation split used for random split."""
    if not 0 < background_fraction < 1:
        raise ValueError("shap_background_fraction must be between 0 and 1")
    if max_background < 1:
        raise ValueError("shap_max_background must be positive")

    labels = y.to_numpy(dtype=int)
    indices = np.arange(len(labels))
    class_counts = pd.Series(labels).value_counts()
    can_stratify = len(class_counts) > 1 and int(class_counts.min()) >= 2

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
        background_size = min(len(indices) - 1, max(1, round(background_fraction * len(indices))))
        background, explanation = shuffled[:background_size], shuffled[background_size:]

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


def _normalize_shap_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"Expected one SHAP output for binary regression; got {len(values)}")
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D SHAP matrix; got shape {values.shape}")
    return values


def compute_mean_abs_shap(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    *,
    background_fraction: float,
    max_background: int,
    batch_size: int,
    seed: int,
    save_shap_values: bool,
) -> pd.Series:
    """Compute mean absolute linear SHAP values in bounded-memory batches."""
    if batch_size < 1:
        raise ValueError("shap_batch_size must be positive")
    if model.coef_.shape[1] != X.shape[1]:
        raise ValueError(
            f"Model expects {model.coef_.shape[1]} features but {X.shape[1]} were selected"
        )

    background_indices, explanation_indices = select_background_indices(
        y, background_fraction, max_background, seed
    )
    X_values = X.to_numpy()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    explainer = shap.LinearExplainer(model, X_values[background_indices])
    shap_file = output_path / "shap_values.npy"
    shap_memmap = None
    if save_shap_values:
        shap_memmap = np.lib.format.open_memmap(
            shap_file,
            mode="w+",
            dtype=np.float64,
            shape=(len(explanation_indices), X.shape[1]),
        )

    absolute_sum = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, len(explanation_indices), batch_size):
        stop = min(start + batch_size, len(explanation_indices))
        batch_indices = explanation_indices[start:stop]
        values = _normalize_shap_values(explainer.shap_values(X_values[batch_indices]))
        absolute_sum += np.abs(values).sum(axis=0)
        if shap_memmap is not None:
            shap_memmap[start:stop] = values

    if shap_memmap is not None:
        shap_memmap.flush()
        del shap_memmap
    elif shap_file.exists():
        # Avoid leaving a stale full matrix after a later run disables saving.
        shap_file.unlink()

    np.save(output_path / "explained_labels.npy", y.to_numpy()[explanation_indices])
    mean_abs = pd.Series(
        absolute_sum / len(explanation_indices),
        index=X.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)
    mean_abs.rename_axis("feature").to_csv(output_path / "mean_abs_shap.csv")
    return mean_abs


def rank_selected_features(
    mean_abs_shap: pd.Series,
    importance_fraction_threshold: float,
    features_below_threshold: int,
) -> list[str]:
    """Apply the random-split 20%-plus-next-10 feature selection rule."""
    total_importance = float(mean_abs_shap.sum())
    if total_importance <= 0:
        raise ValueError("All mean absolute SHAP values are zero")
    above = mean_abs_shap / total_importance > importance_fraction_threshold
    selected = mean_abs_shap.index[above].tolist()
    selected.extend(mean_abs_shap.index[~above][:features_below_threshold].tolist())
    return [str(feature).strip("/") for feature in selected]


def run(config: dict[str, Any]) -> tuple[Path, Path]:
    drug = config["drug"]
    model_filename = config.get("model_filename", DEFAULT_MODEL_FILENAME)
    eligible = discover_eligible_lineages(config["model_dir"], drug, model_filename)
    print(f"{drug}: eligible held-out lineages = {[lineage for lineage, _ in eligible]}")

    genotype_columns = read_genotype_columns(config["genotype_sites_file"], drug)
    output_dir = Path(config["output_dir"])
    map_mar_dir = output_dir / "map_mar"
    dedup_cache = output_dir / "dedup_geno_data" / f"{drug}_full_dedup_indices.npy"
    X, y = prepare_interpretability_data(
        config["input_data_file"], genotype_columns, drug, dedup_cache
    )
    print(f"{drug}: interpreting {len(X)} deduplicated non-missing samples, {X.shape[1]} features")

    relevant = load_relevant_features(
        config["WHO_VCF_mapped_dir"],
        drug,
        bool(config.get("has_neg_strand", False)),
        map_mar_dir / f"relevant_features_{drug}.csv",
    )
    print(f"{drug}: found {len(relevant)} resistant WHO features")

    k_values: Iterable[int] = config.get("k_values", K_VALUES)
    lineage_rows: list[dict[str, Any]] = []
    for lineage, model_path in eligible:
        print(f"{drug} lineage {lineage}: loading {model_path}")
        model = joblib.load(model_path)
        lineage_shap_dir = output_dir / "shap_values" / drug / f"heldout_lineage_{lineage}"
        mean_abs = compute_mean_abs_shap(
            model,
            X,
            y,
            lineage_shap_dir,
            background_fraction=float(config.get("shap_background_fraction", 0.2)),
            max_background=int(config.get("shap_max_background", 160)),
            batch_size=int(config.get("shap_batch_size", 512)),
            seed=int(config.get("random_seed", 42)),
            save_shap_values=bool(config.get("save_shap_values", False)),
        )
        ranked_features = rank_selected_features(
            mean_abs,
            float(config.get("importance_fraction_threshold", 0.2)),
            int(config.get("features_below_threshold", 10)),
        )
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
    mean_rows = mean_map_mar_rows(by_lineage)
    mean_rows.insert(0, "drug", drug)

    map_mar_dir.mkdir(parents=True, exist_ok=True)
    by_lineage_path = map_mar_dir / f"map_mar_{drug}_by_lineage.csv"
    mean_path = map_mar_dir / f"map_mar_{drug}_mean.csv"
    by_lineage.to_csv(by_lineage_path, index=False)
    mean_rows.to_csv(mean_path, index=False)
    print(f"Saved lineage-level metrics to {by_lineage_path}")
    print(f"Saved mean metrics to {mean_path}")
    return by_lineage_path, mean_path


def main() -> None:
    args = parse_arguments()
    run(load_config(args.parameter_file, args.drug))


if __name__ == "__main__":
    main()
