"""Record the fold with the highest validation AUC for a trained drug."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_best_fold(
    model_root: Path,
    results_root: Path,
    drug: str,
    seed: int,
    model_name: str,
    embed_type: str,
    pca_components: int | None,
) -> dict:
    model_dir = model_root / drug / f"seed_{seed}"
    history_dir = results_root / drug / f"seed_{seed}"
    folds = []
    for fold in range(1, 6):
        history = history_dir / f"{model_name}_fold{fold}_history.csv"
        checkpoint = model_dir / f"fold_{fold}" / f"{model_name}_best_model.pt"
        if not history.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Fold {fold} needs both {history} and {checkpoint}")
        with history.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Empty history: {history}")
        best = max(rows, key=lambda row: float(row["val_auc"]))
        auc = float(best["val_auc"])
        if not math.isfinite(auc):
            raise ValueError(f"Invalid validation AUC in {history}: {auc}")
        folds.append({
            "fold": fold,
            "best_epoch": int(best["epoch"]),
            "best_val_auc": auc,
            "history": str(history.resolve()),
            "history_sha256": sha256(history),
        })

    winner = max(folds, key=lambda item: item["best_val_auc"])
    checkpoint = model_dir / f"fold_{winner['fold']}" / f"{model_name}_best_model.pt"
    record = {
        "drug": drug,
        "seed": seed,
        "model_name": model_name,
        "embed_type": embed_type,
        "pca_components": pca_components if embed_type == "pca" else None,
        "selection_metric": "highest validation AUC across five fold histories",
        "best_fold": winner["fold"],
        "best_epoch": winner["best_epoch"],
        "best_val_auc": winner["best_val_auc"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "folds": folds,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    link = model_dir / "best_model.pt"
    if link.exists() and not link.is_symlink():
        raise FileExistsError(f"Refusing to replace a regular file: {link}")
    manifest = model_dir / "best_fold.json"
    temporary = model_dir / f".best_fold.{os.getpid()}.json"
    try:
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)

    temporary_link = model_dir / f".best_model.{os.getpid()}.pt"
    try:
        temporary_link.symlink_to(checkpoint.relative_to(model_dir))
        os.replace(temporary_link, link)
    finally:
        temporary_link.unlink(missing_ok=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-name", default="DNABERTCNN")
    parser.add_argument("--embed-type", default="pca")
    parser.add_argument("--pca-components", type=int, default=10)
    args = parser.parse_args()
    record = select_best_fold(args.model_root, args.results_root, args.drug, args.seed,
                              args.model_name, args.embed_type, args.pca_components)
    print(f"{args.drug}: fold {record['best_fold']}, validation AUC {record['best_val_auc']:.6f}")


if __name__ == "__main__":
    main()
