"""Train Evo2 downstream resistance classifiers with leave-one-lineage-out holdout.

This is a minimal wrapper around ``evo2_downstream/train.py`` that:

1. Loads lineage annotations from ``geno_pheno_full_combined.csv`` and
   ``BIG_TB_isolates_with_lineages.csv``.
2. Monkey-patches ``resistance_classification_train.stratified_split_dataset``
   with a lineage-aware replacement (leave-one-major-lineage-out).
3. Re-routes output paths to include a ``heldout_lineage_<N>`` sub-directory.
4. Delegates all remaining training logic to the unchanged
   ``evo2_downstream/train.py`` pipeline.

The only change to ``evo2_downstream/`` code is that ``stratified_split_dataset``
is replaced at runtime — every other model, optimizer, loss, and k-fold CV step
is untouched.

Usage (example)::

    python train_lineage_holdout.py \\
        --drug ISONIAZID \\
        --heldout-lineage 2 \\
        --embed_type token

    # Dry-run (prints split stats, no training):
    python train_lineage_holdout.py --drug ISONIAZID --heldout-lineage 2 --dry-run
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# ── make evo2_downstream importable ───────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
EVO2_DIR = THIS_DIR.parents[1]
sys.path.insert(0, str(EVO2_DIR))

import evo2_downstream.train as evo2_train  # noqa: E402
from evo2_downstream.config import ensure_dnabert_transfer_learn_on_path  # noqa: E402

# lineage_split lives next to this file
sys.path.insert(0, str(THIS_DIR))
from lineage_split import (  # noqa: E402
    MAJOR_LINEAGES,
    load_isolate_id_map,
    load_lineage_map,
    make_lineage_aware_split_fn,
)

ensure_dnabert_transfer_learn_on_path()
import resistance_classification_train as dnabert_train  # noqa: E402
from utils.token_train_utils import train_on_token_embeddings  # noqa: E402

# ── default paths ─────────────────────────────────────────────────────────────
_DATA_DIR = EVO2_DIR / "data" / "multidrug_classification" / "training"
_GENO_PHENO_CSV = _DATA_DIR / "geno_pheno_full_combined.csv"

# BIG_TB_isolates_with_lineages.csv sits at Data-Curation-for-MTB/
_LINEAGE_CSV = EVO2_DIR.parents[1] / "BIG_TB_isolates_with_lineages.csv"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Extend the base evo2_downstream.train parser with lineage-holdout args."""
    parser = evo2_train.build_parser()
    parser.description = (
        "Train Evo2 downstream resistance classifiers with "
        "leave-one-major-lineage-out holdout split"
    )
    parser.add_argument(
        "--heldout-lineage",
        dest="heldout_lineage",
        required=True,
        choices=list(MAJOR_LINEAGES),
        help="Major MTB lineage (1–4) to hold out as the test set",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print split statistics only; do not train",
    )
    parser.add_argument(
        "--geno-pheno-csv",
        dest="geno_pheno_csv",
        default=str(_GENO_PHENO_CSV),
        help="Path to geno_pheno_full_combined.csv (maps full_N → isolate ID)",
    )
    parser.add_argument(
        "--lineage-csv",
        dest="lineage_csv",
        default=str(_LINEAGE_CSV),
        help="Path to BIG_TB_isolates_with_lineages.csv",
    )
    parser.add_argument(
        "--min-class-count",
        dest="min_class_count",
        type=int,
        default=50,
        help="Minimum samples required in each class (train_S, train_R, test_S, test_R)",
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    heldout = str(args.heldout_lineage)

    # ── fill in default paths from evo2_downstream.train ─────────────────────
    args = evo2_train.apply_defaults(args)

    # ── re-route outputs to a lineage-specific sub-directory ─────────────────
    lineage_tag = f"heldout_lineage_{heldout}"
    args.output_path = str(Path(args.output_path) / lineage_tag)
    args.saved_model_path = str(Path(args.saved_model_path) / lineage_tag)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(args.saved_model_path).mkdir(parents=True, exist_ok=True)

    print(f"[lineage_holdout] drug={args.drug}  heldout_lineage={heldout}")
    print(f"[lineage_holdout] output_path={args.output_path}")
    print(f"[lineage_holdout] saved_model_path={args.saved_model_path}")

    # ── load lineage data ─────────────────────────────────────────────────────
    print(f"[lineage_holdout] Loading isolate ID map: {args.geno_pheno_csv}")
    isolate_id_map = load_isolate_id_map(args.geno_pheno_csv)
    print(f"[lineage_holdout] {len(isolate_id_map)} isolates loaded")

    print(f"[lineage_holdout] Loading lineage map: {args.lineage_csv}")
    lineage_map = load_lineage_map(args.lineage_csv)
    print(f"[lineage_holdout] Lineage annotations for {len(lineage_map)} isolates")

    # ── patch DNABERT drug-index mapping (Evo2 NPZ column order) ─────────────
    evo2_train.enforce_evo2_drug_index_mapping(args.phenotype_label_path)

    # ── patch the random split → lineage-aware split ──────────────────────────
    min_class_count = getattr(args, 'min_class_count', 50)
    lineage_split_fn = make_lineage_aware_split_fn(
        heldout, isolate_id_map, lineage_map, min_class_count=min_class_count
    )
    dnabert_train.stratified_split_dataset = lineage_split_fn
    print(
        f"[lineage_holdout] Patched stratified_split_dataset "
        f"→ leave-one-lineage-out (heldout={heldout}, min_class_count={min_class_count})"
    )

    # ── replace k-fold CV training with single-model training ─────────────────
    # The protein-tasks lineage holdout (lineage_holdout_regression.py) trains a
    # single model on all non-heldout samples without k-fold CV.  We match that
    # behaviour here: one model trained on a random 80/20 train/val split of the
    # lineage-based training set (val is used only for monitoring, not selection).
    def _single_model_train(
        dataset,
        drug,
        num_sensitive,
        num_resistant,
        criterion,
        learning_rate,
        weight_decay,
        output_path,
        saved_model_path,
        model_name="DNABERTCNN",
        model_dim=768,
        model_seq_len=5000,
        k_folds=5,          # ignored — kept for API compatibility
        epochs=30,
        train_batch_size=64,
        val_batch_size=64,
        freeze_bias_frac=0.25,
        random_seed=42,
        fold=None,          # ignored
        data_loader_workers=0,
        skip_completed=False,
        device="cuda",
        early_stopping_min_epochs=5,  # early stopping params - ignored for now
        early_stopping_patience=5,
        early_stopping_min_relative_improvement=1e-3,
        early_stopping_smoothing_window=3,
        use_training_loss_early_stopping=False,
        use_auc_early_stopping=True,  # Changed to validation AUC early stopping
    ):
        """Single-model replacement for cross_val_train_on_token_embeddings.

        Trains one model on an 80/20 random split of ``dataset`` (val is used
        only for loss monitoring / early stopping, not for model selection).
        Saves the model to ``<saved_model_path>/<drug>/seed_<random_seed>/``.
        """
        import math
        import os

        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader, random_split

        val_frac = 0.2
        n_val = max(1, int(len(dataset) * val_frac))
        n_train = len(dataset) - n_val
        train_subset, val_subset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(random_seed),
        )

        loader_kwargs = {"num_workers": data_loader_workers, "pin_memory": True}
        if data_loader_workers > 0:
            loader_kwargs["prefetch_factor"] = 4  # Increased from 1 for better throughput
            loader_kwargs["persistent_workers"] = True  # Keep workers alive between epochs
        train_loader = DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            val_subset, batch_size=val_batch_size, shuffle=False, **loader_kwargs
        )

        save_dir = os.path.join(saved_model_path, f"{drug}/seed_{random_seed}")
        save_path = os.path.join(save_dir, f"{model_name}.pt")
        hist_dir = os.path.join(output_path, f"{drug}/seed_{random_seed}")
        history_file = os.path.join(hist_dir, f"{model_name}_history.csv")

        if skip_completed and os.path.exists(save_path) and os.path.exists(history_file):
            print(f"[lineage_holdout] Skipping completed run for drug: {drug}")
            return

        print(
            f"\n[lineage_holdout] Training single model on {n_train} samples "
            f"(val={n_val}) for drug={drug}"
        )

        train_on_token_embeddings(
            train_loader,
            val_loader,
            drug,
            num_sensitive,
            num_resistant,
            criterion,
            learning_rate,
            weight_decay,
            output_path,
            saved_model_path,
            model_name=model_name,
            model_dim=model_dim,
            model_seq_len=model_seq_len,
            epochs=epochs,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            freeze_bias_frac=freeze_bias_frac,
            random_seed=random_seed,
            device=device,
            early_stopping_min_epochs=early_stopping_min_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_relative_improvement=early_stopping_min_relative_improvement,
            early_stopping_smoothing_window=early_stopping_smoothing_window,
            use_training_loss_early_stopping=use_training_loss_early_stopping,
            use_auc_early_stopping=use_auc_early_stopping,
        )

    dnabert_train.cross_val_train_on_token_embeddings = _single_model_train
    print("[lineage_holdout] Patched cross_val_train_on_token_embeddings → single-model training")

    if args.dry_run:
        _run_dry_split(args, lineage_split_fn)
        return

    # ── delegate to DNABERT training ──────────────────────────────────────────
    dnabert_train.main(args)


def _run_dry_split(args: argparse.Namespace, lineage_split_fn) -> None:
    """Build the dataset and run the split; print statistics without training."""
    from resistance_classification_train import (  # noqa: E402
        TokenMemmapMap,
        MultiGeneConcatDataset,
        DRUG_TO_LOCI,
        build_label_map,
    )

    prefix = "full"
    memmap_dir = args.saved_embed_memmap_dir
    full_label_map, _ = build_label_map(args.phenotype_label_path, args.drug, prefix=prefix)

    loci = DRUG_TO_LOCI[args.drug]
    if len(loci) == 1:
        gene = loci[0]
        meta_paths = sorted(
            glob.glob(f"{memmap_dir}/{gene}/*_{args.embed_type}_meta.npz")
        )
        full_dataset = TokenMemmapMap(meta_paths, full_label_map)
    else:
        gene_memmap_dirs = [f"{memmap_dir}/{gene}/" for gene in loci]
        full_dataset = MultiGeneConcatDataset(gene_memmap_dirs, full_label_map)

    train_idx, test_idx, y_train, y_test = lineage_split_fn(full_dataset, full_label_map)

    print(f"\n[dry-run] Drug={args.drug}  heldout_lineage={args.heldout_lineage}")
    print(f"  train : {len(train_idx):5d} samples  "
          f"(S={int((y_train == 1).sum())}  R={int((y_train == 0).sum())})")
    print(f"  test  : {len(test_idx):5d} samples  "
          f"(S={int((y_test  == 1).sum())}  R={int((y_test  == 0).sum())})")
    print("[dry-run] Split statistics only. No model trained.")


if __name__ == "__main__":
    main(build_parser().parse_args())
