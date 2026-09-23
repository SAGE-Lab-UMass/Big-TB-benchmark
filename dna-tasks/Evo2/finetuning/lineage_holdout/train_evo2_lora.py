"""Supervised LoRA fine-tuning of Evo2 for drug-resistance prediction.

This script implements end-to-end supervised LoRA fine-tuning of Evo2 combined
with the DNABERTCNN classifier architecture. Key features:

1. **Drug-specific training**: Each drug trains independently from the same
   pretrained Evo2 checkpoint with drug-specific LoRA adapters.

2. **End-to-end backpropagation**: Evo2 (frozen) + LoRA (trainable) + 
   DNABERTCNN (trainable) are trained jointly. Gradients flow through the
   full computational graph to update LoRA and classifier parameters.

3. **Full token embeddings**: Preserves the existing architecture that uses
   full token-level Evo2 hidden states [seq_len, hidden_dim] rather than
   mean-pooled representations.

4. **Validation-based early stopping**: Creates deterministic validation split
   from training data only. Test set is never used for model selection.

5. **Lineage-aware splits**: Reuses the existing lineage holdout logic.

6. **Memory optimizations**: Supports gradient accumulation, mixed precision
   (BF16), gradient checkpointing, and configurable batch sizes.

7. **Benchmark mode**: Dry-run option to measure GPU memory, throughput,
   and estimated training time before launching full runs.

Usage::

    # Full training for one drug
    python train_evo2_lora.py --drug ISONIAZID --heldout-lineage 2 \\
        --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \\
        --lora_lr 1e-4 --classifier_lr 1e-3 --epochs 30

    # Benchmark mode (20 steps only)
    python train_evo2_lora.py --drug ISONIAZID --heldout-lineage 2 \\
        --benchmark_steps 20

    # Dry-run split statistics
    python train_evo2_lora.py --drug ISONIAZID --heldout-lineage 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

# ── make paths importable ─────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
EVO2_DIR = THIS_DIR.parents[1]
sys.path.insert(0, str(EVO2_DIR))
sys.path.insert(0, str(THIS_DIR))

from evo2_embed_gen.model.evo2_model import Evo2ModelConfig, Evo2Embedder
from evo2_downstream.config import ensure_finetune_utils_on_path
from utils.lineage_split import (
    MAJOR_LINEAGES,
    load_isolate_id_map,
    load_lineage_map,
    make_lineage_aware_split_fn,
)

try:
    from finetuning.modules.dataloader.locus_order import DRUG_TO_LOCI
    from finetuning.modules.downstream_cnn_model import DNABERTCNN
except ImportError:  # pragma: no cover - legacy fallback path
    ensure_finetune_utils_on_path()
    from dataloader.locus_order import DRUG_TO_LOCI  # type: ignore  # noqa: E402
    from downstream_cnn_model import DNABERTCNN  # type: ignore  # noqa: E402

# ── default paths ─────────────────────────────────────────────────────────────
_DATA_DIR = EVO2_DIR / "data" / "multidrug_classification" / "training"
_GENO_PHENO_CSV = _DATA_DIR / "geno_pheno_full_combined.csv"
_LINEAGE_CSV = EVO2_DIR.parents[1] / "BIG_TB_isolates_with_lineages.csv"
_FASTA_DIR = EVO2_DIR / "data" / "aligned_fasta"


# ──────────────────────────────────────────────────────────────────────────────
# LoRA Model Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class Evo2LoRAClassifier(nn.Module):
    """Frozen Evo2 backbone + trainable LoRA + existing DNABERTCNN head.

    ``Evo2Embedder`` is intentionally a plain Python wrapper, so the underlying
    StripedHyena module is not registered as a child of this ``nn.Module``.
    Helper methods below therefore access LoRA parameters explicitly rather than
    relying on ``self.parameters()`` / ``self.named_parameters()``.
    """

    def __init__(
        self,
        evo2_embedder: Evo2Embedder,
        lora_config: dict[str, Any],
        seq_len: int = 5000,
        hidden_dim: int | None = None,
        enable_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.evo2_embedder = evo2_embedder
        self.seq_len = seq_len

        actual_hidden_dim = evo2_embedder.hidden_size
        if hidden_dim is not None and hidden_dim != actual_hidden_dim:
            raise ValueError(
                f"--hidden-dim={hidden_dim} does not match loaded {evo2_embedder.config.model_name} "
                f"hidden_size={actual_hidden_dim}. Do not hard-code a DNABERT width for Evo2."
            )
        self.hidden_dim = actual_hidden_dim

        # Vortex loads/converts checkpoint tensors under torch.inference_mode().
        # Those inference tensors cannot participate in an autograd-recorded
        # forward, even when the base weights are frozen.  Materialize ordinary
        # Parameter/buffer objects before freezing, LoRA injection, or optimizer
        # construction.  This is a no-op if the loaded Vortex version already
        # returns normal tensors.
        evo2_embedder.ensure_autograd_compatible()

        # Freeze every pretrained Evo2 parameter before adapter injection.
        for param in self.inner_model.parameters():
            param.requires_grad = False

        self._apply_lora(lora_config)

        # Preserve the existing full-token CNN+MLP classifier architecture; only
        # its input width is derived from the actual Evo2 layer output.
        self.classifier = DNABERTCNN(seq_len=seq_len, in_dim=self.hidden_dim, stem_out=64)

        if enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    @property
    def inner_model(self) -> nn.Module:
        return self.evo2_embedder.model.model

    def _apply_lora(self, lora_config: dict[str, Any]) -> None:
        """Inject LoRA in-place into MLP linears in blocks 0..extraction_layer.

        ``get_peft_model`` is intentionally not used because it wraps
        StripedHyena in a HuggingFace-style forward interface.  Low-level PEFT
        injection preserves StripedHyena's original methods and forward
        signature.
        """
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as exc:
            raise ImportError(
                "PEFT is required. Install a version exposing "
                "peft.inject_adapter_in_model."
            ) from exc

        extraction_layer = int(lora_config.get("extraction_layer", 20))
        num_layers = self.evo2_embedder.num_layers
        if not (0 <= extraction_layer < num_layers):
            raise ValueError(
                f"extraction_layer={extraction_layer} is outside loaded model range 0..{num_layers - 1}"
            )

        requested_targets = lora_config.get("target_modules", ["l1", "l2", "l3"])
        leaf_names = [str(name).split(".")[-1] for name in requested_targets]
        if not leaf_names:
            raise ValueError("At least one LoRA target module must be specified")

        layer_nums = "|".join(str(i) for i in range(extraction_layer + 1))
        leaf_regex = "|".join(re.escape(name) for name in leaf_names)
        target_modules_regex = rf"blocks\.(?:{layer_nums})\.mlp\.(?:{leaf_regex})"
        pattern = re.compile(target_modules_regex)

        inner_model = self.inner_model
        target_modules_found: list[tuple[str, str, int]] = []
        for name, module in inner_model.named_modules():
            if pattern.fullmatch(name):
                block_match = re.search(r"blocks\.(\d+)\.", name)
                block_idx = int(block_match.group(1)) if block_match else -1
                target_modules_found.append((name, type(module).__name__, block_idx))

        expected_blocks = list(range(extraction_layer + 1))
        blocks_covered = sorted({b for _, _, b in target_modules_found if b >= 0})
        if not target_modules_found:
            raise ValueError(f"No modules matched LoRA regex: {target_modules_regex}")
        if blocks_covered != expected_blocks:
            raise ValueError(
                f"LoRA block coverage mismatch: expected {expected_blocks}, got {blocks_covered}"
            )
        expected_target_names = {
            f"blocks.{block}.mlp.{leaf}"
            for block in expected_blocks
            for leaf in leaf_names
        }
        found_target_names = {name for name, _, _ in target_modules_found}
        if found_target_names != expected_target_names:
            missing_targets = sorted(expected_target_names - found_target_names)
            extra_targets = sorted(found_target_names - expected_target_names)
            raise ValueError(
                "LoRA target set mismatch. "
                f"Missing={missing_targets[:5]}, extra={extra_targets[:5]}"
            )
        invalid_targets = [
            name for name, _, block in target_modules_found
            if block > extraction_layer or name.split(".")[-1] not in leaf_names
        ]
        if invalid_targets:
            raise ValueError(f"Unexpected LoRA targets: {invalid_targets[:5]}")

        peft_config = LoraConfig(
            r=int(lora_config["rank"]),
            lora_alpha=int(lora_config["alpha"]),
            lora_dropout=float(lora_config["dropout"]),
            target_modules=target_modules_regex,
            bias="none",
            init_lora_weights=True,
        )

        print(f"\n[LoRA] In-place injection into blocks 0-{extraction_layer}")
        print(f"[LoRA] Extraction module: {self.evo2_embedder.config.layer_name}")
        print(f"[LoRA] Target regex: {target_modules_regex}")
        print(f"[LoRA] Matched {len(target_modules_found)} modules before injection")
        for name, module_type, block_idx in target_modules_found[:3]:
            print(f"  {name} (block={block_idx}, type={module_type})")
        if len(target_modules_found) > 3:
            print(f"  ... and {len(target_modules_found) - 3} more")

        # PEFT's low-level tuner inspects ``model.config``.  Vortex uses a
        # ``dotdict`` whose __getattr__ returns None for missing attributes, so
        # ``hasattr(config, 'to_dict')`` is True even though config.to_dict is
        # None.  Current PEFT then tries to call None.  During injection only,
        # expose the same configuration as an ordinary dict; restore the exact
        # Vortex config object immediately afterwards because StripedHyena uses
        # attribute-style access at runtime.
        original_config = getattr(inner_model, "config", None)
        patched_config = False
        if isinstance(original_config, dict) and not callable(getattr(original_config, "to_dict", None)):
            inner_model.config = dict(original_config)
            patched_config = True

        try:
            returned_model = inject_adapter_in_model(
                peft_config,
                inner_model,
                adapter_name="default",
            )
        finally:
            if patched_config:
                inner_model.config = original_config

        # The documented API modifies in-place.  Fail loudly if a future PEFT
        # release changes that contract rather than silently breaking Evo2's
        # wrapper reference.
        if returned_model is not inner_model:
            raise RuntimeError(
                "inject_adapter_in_model unexpectedly returned a different model object; "
                "refusing to replace StripedHyena implicitly."
            )

        # Explicitly freeze all non-LoRA backbone parameters.  PEFT normally
        # does this as part of injection, but keeping the invariant explicit
        # protects this custom non-HF integration.
        for name, param in inner_model.named_parameters():
            param.requires_grad = "lora_" in name

        self.extraction_layer = extraction_layer
        self.lora_target_regex = target_modules_regex
        self.lora_config_dict = {
            "rank": int(lora_config["rank"]),
            "alpha": int(lora_config["alpha"]),
            "dropout": float(lora_config["dropout"]),
            "target_modules": leaf_names,
            "target_regex": target_modules_regex,
            "extraction_layer": extraction_layer,
        }
        self.target_module_names = [name for name, _, _ in target_modules_found]

        # Verify actual adapter parameter placement after injection.
        lora_named = self.lora_named_parameters()
        if not lora_named:
            raise RuntimeError("PEFT injection completed but no LoRA parameters were found")
        bad_lora: list[str] = []
        for name, _ in lora_named:
            block_match = re.search(r"blocks\.(\d+)\.", name)
            if block_match is None or int(block_match.group(1)) > extraction_layer:
                bad_lora.append(name)
        if bad_lora:
            raise RuntimeError(f"LoRA parameters found outside blocks 0-{extraction_layer}: {bad_lora[:5]}")

        lora_a = [(n, p) for n, p in lora_named if "lora_A" in n]
        lora_b = [(n, p) for n, p in lora_named if "lora_B" in n]
        unexpected = [(n, p) for n, p in inner_model.named_parameters() if p.requires_grad and "lora_" not in n]
        if not lora_a or not lora_b:
            raise RuntimeError("Expected both LoRA A and LoRA B parameters after injection")
        if unexpected:
            raise RuntimeError(f"Unexpected trainable Evo2 parameters: {[n for n, _ in unexpected[:5]]}")

        total = sum(p.numel() for p in inner_model.parameters())
        trainable = sum(p.numel() for _, p in lora_named)
        print("[LoRA] Parameter verification:")
        print(f"  Evo2 total parameters: {total:,}")
        print(f"  LoRA parameter tensors: {len(lora_named):,}")
        print(f"  LoRA trainable elements: {trainable:,} ({100 * trainable / total:.4f}% of Evo2)")
        print(f"  LoRA A tensors: {len(lora_a)}, LoRA B tensors: {len(lora_b)}")
        print(f"  No adapters exist after block {extraction_layer}: verified\n")

    def _enable_gradient_checkpointing(self) -> None:
        """Enable checkpointing only if the loaded Vortex build exposes it."""
        if hasattr(self.inner_model, "gradient_checkpointing_enable"):
            self.inner_model.gradient_checkpointing_enable()
            print("[LoRA] Enabled gradient checkpointing for Evo2")
        else:
            print(
                "[LoRA] gradient_checkpointing_enable() is not exposed by this Vortex "
                "StripedHyena build; continuing without checkpointing."
            )

    def train(self, mode: bool = True):
        """Keep the unregistered StripedHyena/LoRA modules in the same mode."""
        super().train(mode)
        self.inner_model.train(mode)
        return self

    def forward(self, sequences: list[str]) -> torch.Tensor:
        embeddings = self._generate_embeddings(sequences)
        return self.classify_hidden(embeddings)

    def classify_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Feed full token states to DNABERTCNN without breaking gradients.

        Evo2-7B activations are typically BF16 while the existing CNN/MLP head
        is initialized in FP32.  Outside an autocast region Conv1d requires a
        compatible input/weight dtype, so cast the activation to the classifier
        dtype.  The cast remains differentiable.  Under AMP/autocast the
        operator dispatcher handles the compute dtype and we avoid an eager
        FP32 activation copy.
        """
        x = hidden_states.transpose(1, 2)  # [B, hidden_dim, seq_len]
        if not torch.is_autocast_enabled():
            classifier_dtype = self.classifier.inp_project.weight.dtype
            if x.dtype != classifier_dtype:
                x = x.to(dtype=classifier_dtype)
        return self.classifier(x)

    def _generate_embeddings(self, sequences: list[str]) -> torch.Tensor:
        """Differentiably extract the exact baseline layer-20 token tensor."""
        hidden_states = self.evo2_embedder.extract_layer_tensor(
            sequences,
            layer_name=self.evo2_embedder.config.layer_name,
            mask_padding=True,
        )

        if hidden_states.shape[1] != self.seq_len:
            raise RuntimeError(
                f"Sequence length mismatch: got {hidden_states.shape[1]}, expected {self.seq_len}"
            )
        if hidden_states.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"Hidden width mismatch: got {hidden_states.shape[-1]}, expected {self.hidden_dim}"
            )

        # Only assert graph connectivity when autograd is actually enabled.  The
        # same method is also used under torch.no_grad() for validation/testing.
        if self.training and torch.is_grad_enabled():
            if not hidden_states.requires_grad or hidden_states.grad_fn is None:
                raise RuntimeError(
                    "Layer-20 representation is detached from autograd. The training path "
                    "must not use Evo2(return_embeddings=True), torch.no_grad(), or .detach()."
                )

        return hidden_states

    def lora_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, param)
            for name, param in self.inner_model.named_parameters()
            if "lora_" in name
        ]

    def trainable_lora_parameters(self) -> list[nn.Parameter]:
        return [param for _, param in self.lora_named_parameters() if param.requires_grad]

    def trainable_classifier_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.classifier.parameters() if param.requires_grad]

    def all_trainable_parameters(self) -> list[nn.Parameter]:
        return self.trainable_lora_parameters() + self.trainable_classifier_parameters()

    def zero_all_grads(self, set_to_none: bool = True) -> None:
        # ``model.zero_grad()`` alone would miss LoRA because Evo2Embedder is a
        # plain Python object, not a registered nn.Module child.
        self.inner_model.zero_grad(set_to_none=set_to_none)
        self.classifier.zero_grad(set_to_none=set_to_none)

    def count_parameters(self) -> dict[str, int | float]:
        total_evo2 = sum(p.numel() for p in self.inner_model.parameters())
        trainable_lora = sum(p.numel() for p in self.trainable_lora_parameters())
        trainable_classifier = sum(p.numel() for p in self.trainable_classifier_parameters())
        frozen_evo2 = total_evo2 - trainable_lora
        return {
            "total_evo2": total_evo2,
            "frozen_evo2": frozen_evo2,
            "trainable_lora": trainable_lora,
            "trainable_classifier": trainable_classifier,
            "total_trainable": trainable_lora + trainable_classifier,
            "percent_trainable": 100.0 * trainable_lora / total_evo2,
        }


# Import data loading utilities
from utils.lora_data import (  # noqa: E402
    SequenceDataset,
    collate_sequences,
    load_drug_sequences_and_labels,
    apply_lineage_split,
    create_validation_split,
)


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: Evo2LoRAClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    optimizer.zero_grad()
    
    for step_idx, (sequences, targets) in enumerate(tqdm(train_loader, desc="Training")):
        targets = targets.to(device)
        
        # Forward pass
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
            logits = model(sequences)
            loss = criterion(logits, targets)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (step_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.all_trainable_parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.all_trainable_parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * gradient_accumulation_steps
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targets.append(targets.cpu())
    
    # Final step if incomplete accumulation
    if (len(train_loader) % gradient_accumulation_steps) != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.all_trainable_parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.all_trainable_parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()
    
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    accuracy = ((all_probs > 0.5) == all_targets).mean()
    
    return {
        "loss": total_loss / len(train_loader),
        "accuracy": accuracy,
    }


# FINAL_METRIC_UNIFORMITY_V2

def evaluate(
    model: Evo2LoRAClassifier,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = False,
) -> dict[str, float]:
    """Compute baseline-aligned per-epoch validation metrics.

    Label convention:
      R = 0
      S = 1

    The model emits one raw logit per isolate.  Exactly as in the frozen-Evo2
    token-embedding training loop:
      prob     = sigmoid(logit) = P(S=1)
      pred     = prob > 0.5
      accuracy = mean(pred == label)
      ROC-AUC  = roc_auc_score(label, prob)

    The reported loss is the arithmetic mean of per-batch criterion values,
    matching the baseline implementation.
    """
    if len(val_loader) == 0:
        raise ValueError("Evaluation DataLoader is empty")

    model.eval()
    total_loss = 0.0
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for sequences, targets in tqdm(val_loader, desc="Validation"):
            targets = targets.to(device)

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=torch.bfloat16
            ):
                logits = model(sequences)
                loss = criterion(logits, targets)

            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).detach().float().cpu())
            all_targets.append(targets.detach().cpu())

    probs_np = torch.cat(all_probs).numpy().reshape(-1)
    targets_np = torch.cat(all_targets).numpy().reshape(-1)

    accuracy = float(((probs_np > 0.5) == targets_np).mean())

    if np.unique(targets_np).size < 2:
        auc = float("nan")
        print("[Evaluation] AUC undefined: only one class is present in this split.")
    else:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(targets_np, probs_np))

    return {
        "loss": total_loss / len(val_loader),
        "accuracy": accuracy,
        "auc": auc,
    }


def get_threshold_val_uniform(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Exact threshold-selection rule used by the supplied frozen-Evo2 baseline.

    ``y_pred`` MUST be probabilities P(S=1) in [0, 1].

    R is encoded as 0 and S as 1, so smaller probabilities indicate resistance.
    Candidate thresholds are 0.00, 0.01, ..., 1.00.  The selected threshold
    maximizes sensitivity_R + specificity_S.  If multiple thresholds tie, the
    largest threshold is selected, matching the supplied function exactly.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.float64)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Target/probability size mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Threshold calibration labels must contain only R=0 and S=1")
    if not np.all(np.isfinite(y_pred)):
        raise FloatingPointError("Threshold calibration probabilities contain NaN/Inf")
    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("get_threshold_val_uniform expects probabilities in [0,1]")

    num_samples = y_pred.shape[0]
    fpr_ = []
    tpr_ = []
    thresholds = np.linspace(0, 1, 101)
    num_sensitive = np.sum(y_true == 1)
    num_resistant = np.sum(y_true == 0)

    if num_sensitive == 0 or num_resistant == 0:
        raise ValueError(
            "Threshold calibration requires both classes; "
            f"got S={int(num_sensitive)}, R={int(num_resistant)}."
        )

    # Keep the same comparison and counting convention as the supplied baseline.
    for threshold in thresholds:
        fp_ = 0  # S incorrectly predicted R
        tp_ = 0  # R correctly predicted R

        for i in range(num_samples):
            # If y is predicted resistant.
            if y_pred[i] < threshold:
                if y_true[i] == 1:
                    fp_ += 1
                if y_true[i] == 0:
                    tp_ += 1

        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    valid_inds = np.arange(101)
    sens_spec_sum = (1 - fpr_) + tpr_
    best_sens_spec_sum = np.max(sens_spec_sum[valid_inds])
    best_inds = np.where(best_sens_spec_sum == sens_spec_sum[valid_inds])

    if best_inds[0].shape[0] == 1:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))
    else:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))[-1]

    return {
        "threshold": float(np.squeeze(thresholds[valid_inds][best_sens_spec_ind])),
        "spec": float(1 - fpr_[valid_inds][best_sens_spec_ind]),
        "sens": float(tpr_[valid_inds][best_sens_spec_ind]),
    }


def collect_probability_scores(
    model: Evo2LoRAClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = False,
    *,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collect labels and sigmoid probabilities P(S=1) under one fixed model."""
    if len(loader) == 0:
        raise ValueError(f"{desc} DataLoader is empty")

    model.eval()
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0

    with torch.no_grad():
        for sequences, targets in tqdm(loader, desc=desc):
            targets = targets.to(device)
            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=torch.bfloat16
            ):
                logits = model(sequences)
                loss = criterion(logits, targets)

            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).detach().float().cpu().reshape(-1))
            all_targets.append(targets.detach().cpu().reshape(-1))

    y = torch.cat(all_targets, dim=0).numpy().astype(np.int64)
    probs = torch.cat(all_probs, dim=0).numpy().astype(np.float64)
    return y, probs, total_loss / len(loader)


def calculate_thresholded_test_metrics(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    threshold: float,
    *,
    loss: float,
) -> dict[str, float | int | str]:
    """Compute final held-out metrics in the same probability/label convention.

    Exact threshold decision rule is kept consistent with ``get_threshold_val``:
        P(S=1) < threshold  -> R (0)
        P(S=1) >= threshold -> S (1)

    Therefore an equality is classified as S, matching the threshold-calibration
    function's strict ``< threshold`` definition of predicted resistance.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_test).reshape(-1).astype(np.int64)
    probs = np.asarray(test_probs).reshape(-1).astype(np.float64)

    if y.shape[0] != probs.shape[0]:
        raise ValueError(
            f"Target/probability size mismatch: {y.shape[0]} vs {probs.shape[0]}"
        )
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Final test labels must contain only R=0 and S=1")
    if not np.all(np.isfinite(probs)):
        raise FloatingPointError("Final test probabilities contain NaN/Inf")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Final test scores must be sigmoid probabilities in [0,1]")

    num_sensitive = int(np.sum(y == 1))
    num_resistant = int(np.sum(y == 0))
    if num_sensitive == 0 or num_resistant == 0:
        raise ValueError(
            "Final held-out metrics require both classes; "
            f"got S={num_sensitive}, R={num_resistant}."
        )

    auc = float(roc_auc_score(y, probs))

    # IMPORTANT: use the exact same strict comparison as threshold calibration.
    binary_pred = np.where(probs < float(threshold), 0, 1).astype(np.int64)

    specificity_S = float(
        np.sum((binary_pred == 1) & (y == 1)) / num_sensitive
    )
    sensitivity_R = float(
        np.sum((binary_pred == 0) & (y == 0)) / num_resistant
    )
    accuracy = float((binary_pred == y).mean())
    balanced_accuracy = float(0.5 * (sensitivity_R + specificity_S))

    return {
        "loss": float(loss),
        "auc": auc,
        "threshold": float(threshold),
        "accuracy": accuracy,
        "specificity_S": specificity_S,
        "sensitivity_R": sensitivity_R,
        "balanced_accuracy": balanced_accuracy,
        "num_sensitive": num_sensitive,
        "num_resistant": num_resistant,
        "score_space": "sigmoid_probability_P(S=1)",
        "label_convention": "R=0,S=1",
        "decision_rule": "P(S)<threshold => R; P(S)>=threshold => S",
    }


class EarlyStopping:
    """Early stopping based on validation ROC-AUC (higher is better)."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        min_epochs: int = 3,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_auc = float("-inf")
        self.reference_auc = None
        self.should_stop = False
        self.best_epoch = -1

    def __call__(self, epoch: int, val_auc: float) -> bool:
        if not math.isfinite(val_auc):
            raise ValueError(
                "Validation AUC is undefined/non-finite. AUC-based checkpointing "
                "and early stopping require both R=0 and S=1 in validation."
            )

        # Track the absolute best AUC from epoch 1 onward.
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.best_epoch = epoch

        # min_delta controls what counts as a patience-resetting improvement.
        # Stopping itself is forbidden until min_epochs complete epochs have run.
        if self.reference_auc is None or val_auc > self.reference_auc + self.min_delta:
            self.reference_auc = val_auc
            self.counter = 0
        elif (epoch + 1) >= self.min_epochs:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train(
    model: Evo2LoRAClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Full training loop with early stopping."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        if exc.name == "tensorboard":
            raise ModuleNotFoundError(
                "TensorBoard is required for training logs. Install it with "
                "`uv pip install --python /path/to/.venv/bin/python tensorboard` "
                "or reinstall `requirements_lora.txt`."
            ) from exc
        raise
    
    # CLAUDE.md PART 7: Optimizer with explicit verification
    # Use .model.model (the inner nn.Module) not .model (the Evo2 Python wrapper
    # which has no .parameters() method).
    lora_params = [p for p in model.evo2_embedder.model.model.parameters() if p.requires_grad]
    classifier_params = list(model.classifier.parameters())
    
    # Verify no overlap or omission
    lora_param_ids = {id(p) for p in lora_params}
    classifier_param_ids = {id(p) for p in classifier_params}
    overlap = lora_param_ids & classifier_param_ids
    assert len(overlap) == 0, f"Found {len(overlap)} params in both LoRA and classifier groups"
    
    # Verify all trainable params are included
    all_trainable = [
        p for p in model.evo2_embedder.model.model.parameters() if p.requires_grad
    ] + [
        p for p in model.classifier.parameters() if p.requires_grad
    ]
    all_trainable_ids = {id(p) for p in all_trainable}
    optimizer_param_ids = lora_param_ids | classifier_param_ids
    missing = all_trainable_ids - optimizer_param_ids
    assert len(missing) == 0, f"Found {len(missing)} trainable params not in optimizer"
    
    # Verify no frozen params in optimizer
    all_frozen_evo2 = [p for p in model.evo2_embedder.model.model.parameters() if not p.requires_grad]
    frozen_in_optimizer = {id(p) for p in all_frozen_evo2} & optimizer_param_ids
    assert len(frozen_in_optimizer) == 0, f"Found {len(frozen_in_optimizer)} frozen params in optimizer"
    
    print(f"\\n[Optimizer] Verification:")
    print(f"  LoRA parameters: {len(lora_params):,}")
    print(f"  Classifier parameters: {len(classifier_params):,}")
    print(f"  LoRA LR: {args.lora_lr}")
    print(f"  Classifier LR: {args.classifier_lr}")
    print(f"  ✓ No overlaps, omissions, or frozen params\\n")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr},
        {"params": classifier_params, "lr": args.classifier_lr},
    ], weight_decay=args.weight_decay)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision scaler
    use_amp = args.use_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        min_epochs=args.min_epochs,
    )
    
    history = []
    best_val_auc = float("-inf")
    best_epoch = -1
    start_epoch = 0

    resume_checkpoint = resolve_resume_checkpoint(args.resume_from, output_dir)
    if resume_checkpoint is not None:
        resume_state = load_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            early_stopping=early_stopping,
            checkpoint_dir=resume_checkpoint,
            output_dir=output_dir,
            args=args,
            device=device,
        )
        start_epoch = int(resume_state["next_epoch_index"])
        history = list(resume_state["history"])
        best_val_auc = float(resume_state["best_val_auc"])
        best_epoch = int(resume_state["best_epoch_index"])

        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint has already completed {start_epoch} epochs, "
                f"but --epochs={args.epochs}. --epochs is the TOTAL target epoch "
                "count, not the number of additional epochs."
            )

        if early_stopping.should_stop:
            print(
                "[Resume] Early stopping had already triggered in the saved "
                "checkpoint; no additional training epochs will run."
            )
            start_epoch = args.epochs
        elif start_epoch == args.epochs:
            print(
                f"[Resume] All requested {args.epochs} epochs are already complete; "
                "proceeding directly to final best-checkpoint evaluation."
            )

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_amp=use_amp,
            scaler=scaler,
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        if not math.isfinite(val_metrics["auc"]):
            raise FloatingPointError(
                f"Validation AUC is undefined/non-finite at epoch {epoch + 1}. "
                "AUC checkpointing requires both R=0 and S=1 in validation."
            )
        
        # Logging
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.3f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.3f} | "
            f"Val AUC: {val_metrics['auc']:.3f}"
        )
        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/auc", val_metrics["auc"], epoch)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
        })
        # Save best model by validation ROC-AUC (higher is better).
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, output_dir / "best", args)
            print(f"✓ Saved best model (val_auc={best_val_auc:.4f})")

        # Update AUC-based early-stopping state first so the resume checkpoint
        # contains the exact patience/counter state for the next invocation.
        should_stop = early_stopping(epoch, val_metrics["auc"])

        # Persist human-readable history every epoch as well as the full restart
        # state. The restart checkpoint is the source of truth for resumption.
        _atomic_write_json(output_dir / "history.json", history)
        writer.flush()
        save_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            early_stopping=early_stopping,
            epoch=epoch,
            history=history,
            best_val_auc=best_val_auc,
            best_epoch=best_epoch,
            output_dir=output_dir,
            args=args,
        )

        if should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best checkpoint epoch: {best_epoch + 1}")
            print(f"Best validation AUC: {best_val_auc:.6f}")
            break
    
    writer.close()
    
    # Save final history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return history


def save_checkpoint(
    model: Evo2LoRAClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Save the PEFT adapter, classifier, optimizer, and reconstruction metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        from peft import get_peft_model_state_dict
    except ImportError:
        try:
            from peft.utils.save_and_load import get_peft_model_state_dict  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Installed PEFT does not expose get_peft_model_state_dict; "
                "use the same PEFT environment used for adapter injection."
            ) from exc

    lora_state_dict = get_peft_model_state_dict(
        model.inner_model,
        adapter_name="default",
    )
    if not lora_state_dict:
        raise RuntimeError("PEFT returned an empty LoRA state dict")
    torch.save(lora_state_dict, checkpoint_dir / "lora_adapter.pt")

    torch.save(model.classifier.state_dict(), checkpoint_dir / "classifier_head.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    config = {
        "epoch": epoch,
        "drug": args.drug,
        "heldout_lineage": args.heldout_lineage,
        "base_evo2_model": args.evo2_model_name,
        "evo2_layer": model.evo2_embedder.config.layer_name,
        "classifier_extraction_layer": model.extraction_layer,
        "hidden_dim": model.hidden_dim,
        "seq_len": model.seq_len,
        "lora": model.lora_config_dict,
        "lora_target_modules_matched": model.target_module_names,
        "classifier": {
            "class": type(model.classifier).__name__,
            "in_dim": model.hidden_dim,
            "seq_len": model.seq_len,
            "stem_out": 64,
        },
        "lora_lr": args.lora_lr,
        "classifier_lr": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "max_length": model.evo2_embedder.config.max_length,
        "pad_char": model.evo2_embedder.config.pad_char,
        "use_kernels": model.evo2_embedder.config.use_kernels,
    }
    with open(checkpoint_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_adapter_and_classifier(
    model: Evo2LoRAClassifier,
    checkpoint_dir: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load a low-level PEFT adapter and classifier into an already reconstructed model."""
    try:
        from peft import set_peft_model_state_dict
    except ImportError:
        try:
            from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Installed PEFT does not expose set_peft_model_state_dict."
            ) from exc

    lora_state = torch.load(
        checkpoint_dir / "lora_adapter.pt",
        map_location=map_location,
        weights_only=True,
    )
    load_result = set_peft_model_state_dict(
        model.inner_model,
        lora_state,
        adapter_name="default",
    )
    # PEFT versions differ in the exact return type; surface incompatibilities
    # without assuming a particular namedtuple implementation.
    missing = getattr(load_result, "missing_keys", None)
    unexpected = getattr(load_result, "unexpected_keys", None)
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA checkpoint keys: {unexpected[:10]}")
    if missing:
        adapter_missing = [key for key in missing if "lora_" in key]
        if adapter_missing:
            raise RuntimeError(f"Missing LoRA checkpoint keys: {adapter_missing[:10]}")

    classifier_state = torch.load(
        checkpoint_dir / "classifier_head.pt",
        map_location=map_location,
        weights_only=True,
    )
    model.classifier.load_state_dict(classifier_state, strict=True)


# RESUME_TRAINING_V3


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace a small JSON metadata/history file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _resume_signature(args: argparse.Namespace) -> dict[str, Any]:
    """Training settings that must remain identical across a resume."""
    def _path_value(name: str) -> str | None:
        value = getattr(args, name, None)
        if value is None:
            return None
        return str(Path(value).expanduser().resolve())

    return {
        "drug": str(args.drug),
        "heldout_lineage": str(args.heldout_lineage),
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "geno_pheno_csv": _path_value("geno_pheno_csv"),
        "lineage_csv": _path_value("lineage_csv"),
        "fasta_dir": _path_value("fasta_dir"),
        "seq_len": int(args.seq_len),
        "evo2_model_name": str(args.evo2_model_name),
        "evo2_layer": str(args.evo2_layer),
        "classifier_extraction_layer": int(args.classifier_extraction_layer),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": list(args.lora_target_modules),
        "lora_lr": float(args.lora_lr),
        "classifier_lr": float(args.classifier_lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "max_grad_norm": float(args.max_grad_norm),
        "use_amp": bool(args.use_amp),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "min_epochs": int(args.min_epochs),
        "num_workers": int(args.num_workers),
    }


def _capture_rng_state() -> dict[str, Any]:
    """Capture RNG states needed to continue stochastic training."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state after model/optimizer reconstruction."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    saved_cuda = state.get("torch_cuda")
    if saved_cuda is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Resume checkpoint contains CUDA RNG state but CUDA is unavailable."
            )
        current_devices = torch.cuda.device_count()
        if len(saved_cuda) != current_devices:
            raise RuntimeError(
                "CUDA device-count mismatch while restoring RNG state: "
                f"checkpoint={len(saved_cuda)}, current={current_devices}."
            )
        torch.cuda.set_rng_state_all(saved_cuda)


def _early_stopping_state(early_stopping: EarlyStopping) -> dict[str, Any]:
    """Persist every scalar field that controls AUC-based stopping."""
    return dict(early_stopping.__dict__)


def _restore_early_stopping_state(
    early_stopping: EarlyStopping,
    state: dict[str, Any],
) -> None:
    required = {
        "patience",
        "min_delta",
        "min_epochs",
        "counter",
        "best_auc",
        "reference_auc",
        "should_stop",
        "best_epoch",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(
            f"Resume checkpoint is missing early-stopping fields: {missing}"
        )
    for key, value in state.items():
        setattr(early_stopping, key, value)


def _validate_resume_signature(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Fail rather than silently resume with a different experiment."""
    keys = sorted(set(saved) | set(current))
    mismatches = [
        (key, saved.get(key), current.get(key))
        for key in keys
        if saved.get(key) != current.get(key)
    ]
    if mismatches:
        preview = "; ".join(
            f"{key}: checkpoint={old!r}, current={new!r}"
            for key, old, new in mismatches[:12]
        )
        if len(mismatches) > 12:
            preview += f"; ... +{len(mismatches) - 12} more"
        raise ValueError(
            "Refusing to resume because training configuration changed. " + preview
        )


def save_resume_checkpoint(
    model: Evo2LoRAClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    early_stopping: EarlyStopping,
    epoch: int,
    history: list[dict[str, Any]],
    best_val_auc: float,
    best_epoch: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Save one crash-consistent checkpoint for the latest completed epoch.

    ``epoch`` is zero-based internally.  A checkpoint is not advertised through
    ``latest_checkpoint.json`` until every required file has been written.
    """
    output_dir = Path(output_dir)
    checkpoint_root = output_dir / "resume_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = checkpoint_root / f"epoch_{epoch + 1:04d}"
    tmp_dir = checkpoint_root / f".epoch_{epoch + 1:04d}.tmp.{os.getpid()}"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    try:
        # Reuse the verified PEFT/classifier/optimizer serializer.
        save_checkpoint(model, optimizer, epoch, tmp_dir, args)

        trainer_state = {
            "format_version": 1,
            "completed_epoch_index": int(epoch),
            "completed_epoch": int(epoch + 1),
            "next_epoch_index": int(epoch + 1),
            "history": history,
            "best_val_auc": float(best_val_auc),
            "best_epoch_index": int(best_epoch),
            "best_epoch": int(best_epoch + 1) if best_epoch >= 0 else None,
            "early_stopping": _early_stopping_state(early_stopping),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "rng_state": _capture_rng_state(),
            "resume_signature": _resume_signature(args),
            "output_dir": str(output_dir.resolve()),
        }
        torch.save(trainer_state, tmp_dir / "trainer_state.pt")

        # Written LAST inside the checkpoint directory.
        (tmp_dir / "CHECKPOINT_COMPLETE").write_text("complete\n")

        # The final epoch directory appears only after the temporary checkpoint
        # is complete.  os.replace is atomic on the same filesystem.
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        os.replace(tmp_dir, checkpoint_dir)

        pointer = {
            "format_version": 1,
            "checkpoint": str(checkpoint_dir.relative_to(output_dir)),
            "completed_epoch": int(epoch + 1),
            "next_epoch": int(epoch + 2),
        }
        _atomic_write_json(output_dir / "latest_checkpoint.json", pointer)

        # Only after the pointer references the new completed checkpoint do we
        # remove older restart snapshots.  Thus an interrupted write never
        # destroys the previous resume point.
        for old_dir in checkpoint_root.glob("epoch_*"):
            if old_dir != checkpoint_dir and old_dir.is_dir():
                shutil.rmtree(old_dir)

        print(
            f"✓ Saved resume checkpoint after epoch {epoch + 1}: {checkpoint_dir}"
        )
        return checkpoint_dir
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def resolve_resume_checkpoint(
    resume_from: str | None,
    output_dir: Path,
) -> Path | None:
    """Resolve ``--resume-from`` to a fully completed checkpoint directory."""
    if resume_from is None:
        return None

    output_dir = Path(output_dir)
    value = str(resume_from).strip()

    if value.lower() == "auto":
        pointer_path = output_dir / "latest_checkpoint.json"
        if not pointer_path.exists():
            # Starting fresh is safe only when the directory has no evidence of
            # an older/non-resumable training run.
            prior_artifacts = [
                p
                for p in (
                    output_dir / "best",
                    output_dir / "history.json",
                    output_dir / "decision_threshold.json",
                    output_dir / "test_metrics.json",
                )
                if p.exists()
            ]
            if prior_artifacts:
                raise RuntimeError(
                    "--resume-from auto found no completed resume checkpoint, but "
                    "this output directory already contains training artifacts: "
                    + ", ".join(str(p) for p in prior_artifacts)
                    + ". Use a clean output directory to start fresh, or provide "
                    "an explicit valid resume checkpoint."
                )
            print(
                "[Resume] No completed checkpoint found; starting a fresh run. "
                "Epoch-level checkpoints will be created after each completed epoch."
            )
            return None

        with open(pointer_path) as f:
            pointer = json.load(f)
        checkpoint_value = pointer.get("checkpoint")
        if not checkpoint_value:
            raise RuntimeError(f"Invalid resume pointer: {pointer_path}")
        checkpoint_dir = Path(checkpoint_value)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = output_dir / checkpoint_dir
    else:
        checkpoint_dir = Path(value).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = checkpoint_dir.resolve()

        # Also allow users to point at the output directory itself.
        if (checkpoint_dir / "latest_checkpoint.json").exists() and not (
            checkpoint_dir / "trainer_state.pt"
        ).exists():
            with open(checkpoint_dir / "latest_checkpoint.json") as f:
                pointer = json.load(f)
            pointed = Path(pointer["checkpoint"])
            checkpoint_dir = (
                pointed if pointed.is_absolute() else checkpoint_dir / pointed
            )

    required = [
        "CHECKPOINT_COMPLETE",
        "lora_adapter.pt",
        "classifier_head.pt",
        "optimizer.pt",
        "training_config.json",
        "trainer_state.pt",
    ]
    missing = [name for name in required if not (checkpoint_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"Resume checkpoint is incomplete: {checkpoint_dir}; missing {missing}"
        )
    return checkpoint_dir


def load_resume_checkpoint(
    model: Evo2LoRAClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    early_stopping: EarlyStopping,
    checkpoint_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    """Restore model + optimizer + trainer state and return the trainer metadata."""
    checkpoint_dir = Path(checkpoint_dir)

    # ``trainer_state.pt`` is produced locally by this training script and
    # intentionally contains Python/NumPy RNG tuples, so weights_only=False is
    # required.  Never use this path with an untrusted third-party checkpoint.
    trainer_state = torch.load(
        checkpoint_dir / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )

    if int(trainer_state.get("format_version", -1)) != 1:
        raise RuntimeError(
            f"Unsupported resume checkpoint format: {trainer_state.get('format_version')}"
        )

    saved_output_dir = str(trainer_state.get("output_dir", ""))
    current_output_dir = str(Path(output_dir).resolve())
    if saved_output_dir != current_output_dir:
        raise ValueError(
            "Resume checkpoint belongs to a different output directory: "
            f"checkpoint={saved_output_dir!r}, current={current_output_dir!r}. "
            "Resume into the same drug/lineage output directory."
        )

    _validate_resume_signature(
        trainer_state["resume_signature"],
        _resume_signature(args),
    )

    # Restore model weights first, then optimizer/scaler state.
    load_adapter_and_classifier(model, checkpoint_dir, map_location=device)

    optimizer_state = torch.load(
        checkpoint_dir / "optimizer.pt",
        map_location=device,
        weights_only=True,
    )
    optimizer.load_state_dict(optimizer_state)

    saved_scaler = trainer_state.get("scaler_state_dict")
    if scaler is None:
        if saved_scaler is not None:
            raise RuntimeError(
                "Checkpoint used AMP/GradScaler but current run has AMP disabled."
            )
    else:
        if saved_scaler is None:
            raise RuntimeError(
                "Current run uses AMP/GradScaler but checkpoint did not."
            )
        scaler.load_state_dict(saved_scaler)

    _restore_early_stopping_state(
        early_stopping,
        trainer_state["early_stopping"],
    )

    # Restore RNG LAST so reconstruction/loading does not perturb the random
    # stream that should be used for the next epoch's shuffle/dropout.
    _restore_rng_state(trainer_state["rng_state"])

    print(
        f"[Resume] Restored completed epoch {trainer_state['completed_epoch']} "
        f"from {checkpoint_dir}"
    )
    print(
        f"[Resume] Next epoch: {trainer_state['next_epoch_index'] + 1}; "
        f"best epoch: {trainer_state.get('best_epoch')}; "
        f"best val AUC: {trainer_state['best_val_auc']:.6f}; "
        f"early-stop counter: {early_stopping.counter}/{early_stopping.patience}"
    )
    return trainer_state


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_sequences_and_labels(
    drug: str,
    geno_pheno_csv: Path,
    lineage_csv: Path,
    heldout_lineage: str,
    val_frac: float = 0.2,
    random_seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset, int, int]:
    """Load sequences and labels, create train/val/test splits.
    
    Returns:
        train_dataset, val_dataset, test_dataset, num_sensitive, num_resistant
    """
    # Load data (this is a simplified version - adapt from actual data loading)
    # For now, returning placeholder structure
    # TODO: Integrate with actual FASTA loading and lineage splitting
    
    # Placeholder - replace with actual implementation
    raise NotImplementedError(
        "Data loading integration needed. "
        "Must load FASTA sequences, join with phenotypes, "
        "apply lineage-aware splitting."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark Mode
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    model: Evo2LoRAClassifier,
    train_loader: DataLoader,
    device: str,
    num_steps: int = 20,
    use_amp: bool = False,
    lora_lr: float = 1e-4,
    classifier_lr: float = 1e-3,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """Benchmark the real LoRA + CNN/MLP training path."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK MODE ({num_steps} real training steps)")
    print(f"{'='*60}")

    model.train()
    lora_params = model.trainable_lora_parameters()
    classifier_params = model.trainable_classifier_parameters()
    if not lora_params or not classifier_params:
        raise RuntimeError("Benchmark requires both trainable LoRA and classifier parameters")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_lr},
            {"params": classifier_params, "lr": classifier_lr},
        ],
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    times: list[float] = []
    total_examples = 0
    total_tokens = 0

    for step_idx, (sequences, targets) in enumerate(train_loader):
        if step_idx >= num_steps:
            break

        targets = targets.to(device)
        batch_size = len(sequences)
        model.zero_all_grads()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp and device.startswith("cuda"),
            dtype=torch.bfloat16,
        ):
            logits = model(sequences)
            loss = criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.all_trainable_parameters(), max_grad_norm)
        optimizer.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

        times.append(elapsed)
        total_examples += batch_size
        total_tokens += batch_size * model.seq_len
        print(
            f"  step {step_idx + 1}/{num_steps}: loss={loss.item():.6f}, "
            f"seconds={elapsed:.3f}"
        )

    if not times:
        raise RuntimeError("Benchmark did not execute any training steps")

    total_time = float(sum(times))
    avg_time_per_step = float(np.mean(times))
    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else float("nan")
    )
    examples_per_sec = total_examples / total_time
    tokens_per_sec = total_tokens / total_time

    total_batches = len(train_loader)
    estimated_seconds_per_epoch = avg_time_per_step * total_batches

    results = {
        "num_measured_steps": len(times),
        "avg_seconds_per_step": avg_time_per_step,
        "peak_gpu_memory_gb": peak_memory_gb,
        "examples_per_second": examples_per_sec,
        "tokens_per_second": tokens_per_sec,
        "estimated_steps_per_epoch": total_batches,
        "estimated_seconds_per_epoch": estimated_seconds_per_epoch,
        "estimated_hours_per_epoch": estimated_seconds_per_epoch / 3600,
    }

    print("\nBenchmark Results:")
    print(f"  Measured steps: {len(times)}")
    print(f"  Avg time/step: {avg_time_per_step:.3f} sec")
    if device.startswith("cuda"):
        print(f"  Peak GPU memory: {peak_memory_gb:.2f} GB")
    print(f"  Examples/sec: {examples_per_sec:.3f}")
    print(f"  Tokens/sec: {tokens_per_sec:.0f}")
    print(f"  Estimated steps/epoch: {total_batches}")
    print(f"  Estimated time/epoch: {estimated_seconds_per_epoch/3600:.2f} hours")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervised LoRA fine-tuning of Evo2 for drug resistance prediction"
    )
    
    # Data arguments
    parser.add_argument("--drug", type=str, required=True, help="Drug name (e.g., ISONIAZID)")
    parser.add_argument(
        "--heldout-lineage",
        dest="heldout_lineage",
        required=True,
        choices=list(MAJOR_LINEAGES),
        help="Major MTB lineage (1–4) to hold out as test set",
    )
    parser.add_argument(
        "--geno-pheno-csv",
        dest="geno_pheno_csv",
        default=str(_GENO_PHENO_CSV),
        help="Path to geno_pheno_full_combined.csv",
    )
    parser.add_argument(
        "--lineage-csv",
        dest="lineage_csv",
        default=str(_LINEAGE_CSV),
        help="Path to BIG_TB_isolates_with_lineages.csv",
    )
    parser.add_argument(
        "--fasta-dir",
        dest="fasta_dir",
        type=str,
        default=str(_FASTA_DIR),
        help="Directory containing aligned per-gene FASTA files",
    )
    parser.add_argument(
        "--val-frac",
        dest="val_frac",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation (default: 0.2)",
    )
    
    # Model arguments
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=5000, help="Maximum sequence length")
    parser.add_argument(
        "--hidden-dim",
        dest="hidden_dim",
        type=int,
        default=None,
        help=(
            "Optional assertion for Evo2 hidden width. By default it is derived "
            "from the loaded Evo2/StripedHyena config (4096 for the supplied evo2_7b)."
        ),
    )
    parser.add_argument(
        "--evo2-model-name",
        dest="evo2_model_name",
        type=str,
        default="evo2_7b",
        help="Evo2 model name",
    )
    parser.add_argument(
        "--evo2-layer",
        dest="evo2_layer",
        type=str,
        default="blocks.20.mlp.l3",
        help="Evo2 layer to extract embeddings from (default: blocks.20.mlp.l3 to match frozen-embedding baseline)",
    )
    parser.add_argument(
        "--classifier-extraction-layer",
        dest="classifier_extraction_layer",
        type=int,
        default=20,
        help="User-facing layer number for classifier input (default: 20, maps to blocks.20.mlp.l3)",
    )
    
    # LoRA arguments
    parser.add_argument("--lora-rank", dest="lora_rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument(
        "--lora-dropout",
        dest="lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)",
    )
    parser.add_argument(
        "--lora-target-modules",
        dest="lora_target_modules",
        type=str,
        nargs="+",
        default=["l1", "l2", "l3"],
        help=(
            "Leaf module names to target with LoRA inside each block's MLP "
            "(default: l1 l2 l3). The layer range 0..extraction_layer is baked "
            "into the regex automatically; do not pass full paths here."
        ),
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument(
        "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4, effective batch = 16)",
    )
    parser.add_argument(
        "--lora-lr",
        dest="lora_lr",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA parameters (default: 1e-4)",
    )
    parser.add_argument(
        "--classifier-lr",
        dest="classifier_lr",
        type=float,
        default=1e-3,
        help="Learning rate for classifier parameters (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--max-grad-norm",
        dest="max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
    )
    parser.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Use automatic mixed precision (BF16) training",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    
    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--min-delta",
        dest="min_delta",
        type=float,
        default=1e-4,
        help="Early stopping min delta (default: 1e-4)",
    )
    parser.add_argument(
        "--min-epochs",
        dest="min_epochs",
        type=int,
        default=3,
        help="Minimum epochs before AUC-based early stopping (default: 3)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Output directory (default: training_output/lora_finetuned/<drug>/heldout_lineage_<N>)",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        type=str,
        default=None,
        help=(
            "Resume epoch-level training state. Use 'auto' to resume from "
            "<output-dir>/latest_checkpoint.json; if no prior artifacts exist, "
            "'auto' starts fresh. --epochs remains the TOTAL target epoch count."
        ),
    )
    
    # Modes
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print split statistics only, no training",
    )
    parser.add_argument(
        "--benchmark-steps",
        dest="benchmark_steps",
        type=int,
        default=0,
        help="Run benchmark mode for N steps instead of full training (default: 0 = disabled)",
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=1, help="DataLoader workers (default: 1)")
    
    return parser


def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Output directory
    if args.output_dir is None:
        lineage_tag = f"heldout_lineage_{args.heldout_lineage}"
        args.output_dir = str(
            EVO2_DIR / "training_output" / "lora_finetuned" / args.drug / lineage_tag
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"LoRA Fine-tuning Configuration")
    print(f"{'='*60}")
    print(f"Drug: {args.drug}")
    print(f"Heldout lineage: {args.heldout_lineage}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"LoRA target modules: {args.lora_target_modules}")
    print(f"LoRA LR: {args.lora_lr}, Classifier LR: {args.classifier_lr}")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading sequences and labels...")
    fasta_dir = Path(args.fasta_dir)

    isolate_ids, sequences, labels = load_drug_sequences_and_labels(
        args.drug,
        Path(args.geno_pheno_csv),
        fasta_dir,
        prefix="",
    )
    
    # Load lineage annotations
    print(f"\nLoading lineage annotations...")
    print(f"  Geno-pheno CSV: {args.geno_pheno_csv}")
    print(f"  Lineage CSV: {args.lineage_csv}")
    
    isolate_id_map = load_isolate_id_map(args.geno_pheno_csv)
    lineage_map = load_lineage_map(args.lineage_csv)
    
    print(f"  Loaded {len(isolate_id_map)} isolate ID mappings")
    print(f"  Loaded {len(lineage_map)} lineage annotations")
    
    # Apply lineage-aware train/test split
    print(f"\nApplying lineage-aware split (heldout={args.heldout_lineage})...")
    train_indices, test_indices = apply_lineage_split(
        isolate_ids,
        sequences,
        labels,
        args.heldout_lineage,
        isolate_id_map,
        lineage_map,
    )
    
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Test samples: {len(test_indices)}")
    
    # Create validation split from training data
    print(f"\nCreating validation split ({args.val_frac*100:.0f}% of training)...")
    train_indices, val_indices = create_validation_split(
        train_indices,
        labels,
        val_frac=args.val_frac,
        random_seed=args.seed,
    )
    
    print(f"  Final training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Test samples: {len(test_indices)}")
    
    # Count class distribution
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]
    
    num_resistant_train = sum(1 for l in train_labels if l == 0)
    num_susceptible_train = sum(1 for l in train_labels if l == 1)
    num_resistant_val = sum(1 for l in val_labels if l == 0)
    num_susceptible_val = sum(1 for l in val_labels if l == 1)
    num_resistant_test = sum(1 for l in test_labels if l == 0)
    num_susceptible_test = sum(1 for l in test_labels if l == 1)
    
    print(f"\nClass distribution:")
    print(f"  Train: R={num_resistant_train}, S={num_susceptible_train}")
    print(f"  Val:   R={num_resistant_val}, S={num_susceptible_val}")
    print(f"  Test:  R={num_resistant_test}, S={num_susceptible_test}")
    
    # Create datasets
    train_dataset = SequenceDataset(sequences, labels, train_indices)
    val_dataset = SequenceDataset(sequences, labels, val_indices)
    test_dataset = SequenceDataset(sequences, labels, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
        pin_memory=True,
    )
    
    # Dry-run mode: print statistics and exit
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY-RUN MODE: Split statistics only")
        print(f"{'='*60}")
        print(f"Drug: {args.drug}")
        print(f"Heldout lineage: {args.heldout_lineage}")
        print(f"Total isolates: {len(sequences)}")
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_indices):5d} (R={num_resistant_train:4d}, S={num_susceptible_train:4d})")
        print(f"  Val:   {len(val_indices):5d} (R={num_resistant_val:4d}, S={num_susceptible_val:4d})")
        print(f"  Test:  {len(test_indices):5d} (R={num_resistant_test:4d}, S={num_susceptible_test:4d})")
        print(f"\nNo model trained.")
        return
    
    # Initialize Evo2 model
    print("Loading Evo2 model...")
    evo2_config = Evo2ModelConfig(
        model_name=args.evo2_model_name,
        layer_name=args.evo2_layer,
        max_length=args.seq_len,
    )
    evo2_embedder = Evo2Embedder(evo2_config)
    
    # Initialize LoRA model
    # Extract layer number from layer_name for LoRA restriction
    # e.g., "blocks.20.mlp.l3" -> 20
    import re
    layer_match = re.search(r'blocks\.(\d+)\.', args.evo2_layer)
    if layer_match:
        extraction_layer_num = int(layer_match.group(1))
        if extraction_layer_num != args.classifier_extraction_layer:
            raise ValueError(
                f"--evo2-layer={args.evo2_layer!r} maps to block {extraction_layer_num}, "
                f"but --classifier-extraction-layer={args.classifier_extraction_layer}. "
                "These must agree for a controlled layer-20 experiment."
            )
    else:
        raise ValueError(
            f"Could not parse a StripedHyena block index from --evo2-layer={args.evo2_layer!r}"
        )

    print(f"\n{'='*60}")
    print(f"Layer-20 Extraction Configuration")
    print(f"{'='*60}")
    print(f"User-facing layer number: {args.classifier_extraction_layer}")
    print(f"Evo2 internal layer: {args.evo2_layer}")
    print(f"Block index: {extraction_layer_num}")
    print(f"LoRA will be applied to blocks 0-{extraction_layer_num} only")
    print(f"Blocks {extraction_layer_num + 1}-{evo2_embedder.num_layers - 1} remain fully frozen")
    print(f"{'='*60}\n")
    
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_modules": args.lora_target_modules,
        "extraction_layer": extraction_layer_num,
    }
    
    model = Evo2LoRAClassifier(
        evo2_embedder,
        lora_config,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        enable_gradient_checkpointing=args.gradient_checkpointing,
    )
    model = model.to(device)
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"\nParameter Counts:")
    print(f"  Total Evo2 parameters: {param_counts['total_evo2']:,}")
    print(f"  Frozen Evo2 parameters: {param_counts['frozen_evo2']:,}")
    print(f"  Trainable LoRA parameters: {param_counts['trainable_lora']:,}")
    print(f"  Trainable classifier parameters: {param_counts['trainable_classifier']:,}")
    print(f"  Total trainable: {param_counts['total_trainable']:,}")
    print(f"  Percent trainable: {param_counts['percent_trainable']:.2f}%")
    
    # Assertions to verify correctness
    assert param_counts['trainable_lora'] > 0, "LoRA parameters should be trainable"
    assert param_counts['trainable_classifier'] > 0, "Classifier parameters should be trainable"
    assert param_counts['frozen_evo2'] > 0, "Base Evo2 parameters should be frozen"
    
    # Verify LoRA modules exist
    print("\nVerifying LoRA setup...")
    has_lora = False
    for name, module in model.evo2_embedder.model.model.named_modules():
        if "lora" in name.lower():
            has_lora = True
            break
    
    if not has_lora:
        print("[WARNING] No LoRA modules found in model!")
        print("Check that target_modules match actual layer names in Evo2.")
    else:
        print("  ✓ LoRA modules detected in model")
    
    # Lightweight correctness smoke test before benchmark/full training.
    print("\nTesting differentiable forward/backward path...")
    try:
        test_seqs, test_labels = next(iter(train_loader))
        # One sample is enough to verify the graph and keeps the diagnostic cheap.
        test_seqs = test_seqs[:1]
        test_labels = test_labels[:1].to(device)

        model.eval()
        with torch.no_grad():
            inference_hidden = model._generate_embeddings(test_seqs)
            inference_logits = model.classify_hidden(inference_hidden)
        print(f"  Input sequences: {len(test_seqs)}")
        print(f"  Layer-{extraction_layer_num} shape: {tuple(inference_hidden.shape)}")
        print(f"  Layer-{extraction_layer_num} dtype: {inference_hidden.dtype}")
        print(f"  Expected shape: ({len(test_seqs)}, {args.seq_len}, {model.hidden_dim})")
        assert inference_hidden.shape == (len(test_seqs), args.seq_len, model.hidden_dim)
        assert inference_logits.shape == (len(test_seqs),)

        non_zero_ratio = (inference_hidden.abs() > 1e-6).float().mean().item()
        print(f"  Non-zero ratio after padding mask: {non_zero_ratio:.3f}")
        if non_zero_ratio == 0.0:
            raise RuntimeError("Layer representation is entirely zero after masking")

        model.train()
        model.zero_all_grads()
        differentiable_hidden = model._generate_embeddings(test_seqs)
        if not differentiable_hidden.requires_grad or differentiable_hidden.grad_fn is None:
            raise RuntimeError(
                "Layer-20 hidden state is detached. The LoRA training path is not differentiable."
            )

        logits = model.classify_hidden(differentiable_hidden)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, test_labels)
        if loss.grad_fn is None:
            raise RuntimeError("Classification loss has no grad_fn")
        loss.backward()

        lora_named = model.lora_named_parameters()
        lora_connected = [(n, p) for n, p in lora_named if p.grad is not None]
        lora_b_nonzero = [
            (n, p.grad.norm().item())
            for n, p in lora_named
            if "lora_B" in n and p.grad is not None and p.grad.norm().item() > 0
        ]
        classifier_connected = [
            p for p in model.classifier.parameters()
            if p.requires_grad and p.grad is not None
        ]

        if not lora_connected:
            raise RuntimeError("No trainable LoRA parameter is connected to the classification loss")
        if not lora_b_nonzero:
            raise RuntimeError(
                "No LoRA-B parameter received a nonzero gradient on the first backward pass"
            )
        if not classifier_connected:
            raise RuntimeError("Classifier parameters did not receive gradients")

        post_extraction = []
        for name, _ in lora_named:
            match = re.search(r"blocks\.(\d+)\.", name)
            if match and int(match.group(1)) > extraction_layer_num:
                post_extraction.append(name)
        if post_extraction:
            raise RuntimeError(
                f"Found LoRA parameters after extraction block {extraction_layer_num}: "
                f"{post_extraction[:3]}"
            )

        print(f"  hidden.requires_grad: {differentiable_hidden.requires_grad}")
        print(f"  hidden.grad_fn: {type(differentiable_hidden.grad_fn).__name__}")
        print(f"  loss: {loss.item():.6f}")
        print(f"  LoRA tensors connected: {len(lora_connected)}/{len(lora_named)}")
        print(f"  LoRA-B tensors with nonzero first-step gradient: {len(lora_b_nonzero)}")
        print(f"  Classifier tensors connected: {len(classifier_connected)}")
        print("  ✓ Differentiable Evo2→layer20→CNN/MLP path verified")
        model.zero_all_grads()

    except Exception as exc:
        print(f"  ✗ Smoke test failed: {exc}")
        raise

    print("\n" + "="*60)
    print("Correctness smoke test passed.")
    print("="*60 + "\n")

    # Benchmark mode
    if args.benchmark_steps > 0:
        run_benchmark(
            model,
            train_loader,
            device,
            num_steps=args.benchmark_steps,
            use_amp=args.use_amp,
            lora_lr=args.lora_lr,
            classifier_lr=args.classifier_lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )
        return
    
    # Full training
    print("Starting training...")
    history = train(model, train_loader, val_loader, args, output_dir, device)

    # Test is touched only after validation-based model selection is complete.
    # Reload the best validation-AUROC checkpoint, calibrate the decision
    # threshold on the FINAL TRAINING SUBSET only, freeze it, then evaluate the
    # untouched held-out lineage.  The held-out test set never influences the
    # threshold or checkpoint choice.
    best_dir = output_dir / "best"
    print(
        f"\nLoading best validation-AUC checkpoint from {best_dir} "
        "for final threshold calibration and held-out testing..."
    )
    load_adapter_and_classifier(model, best_dir, map_location=device)
    criterion = nn.BCEWithLogitsLoss()

    # Non-shuffled calibration loader: order is irrelevant to the threshold,
    # but deterministic ordering makes reproduction/debugging easier.
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
        pin_memory=True,
    )

    y_train_final, train_probs, _ = collect_probability_scores(
        model,
        train_eval_loader,
        criterion,
        device,
        use_amp=args.use_amp,
        desc="Threshold calibration (train)",
    )

    threshold_result = get_threshold_val_uniform(y_train_final, train_probs)
    threshold = float(threshold_result["threshold"])
    if not math.isfinite(threshold):
        raise FloatingPointError(f"Training-derived threshold is non-finite: {threshold}")

    print(
        f"[Metrics] Training-derived probability threshold: {threshold:.4f} | "
        f"train sens_R={threshold_result['sens']:.4f} | "
        f"train spec_S={threshold_result['spec']:.4f} | "
        f"train bal_acc={0.5 * (threshold_result['sens'] + threshold_result['spec']):.4f}"
    )

    y_test_final, test_probs, test_loss = collect_probability_scores(
        model,
        test_loader,
        criterion,
        device,
        use_amp=args.use_amp,
        desc="Held-out test",
    )
    test_metrics = calculate_thresholded_test_metrics(
        y_test_final,
        test_probs,
        threshold,
        loss=test_loss,
    )

    threshold_metadata = {
        "threshold": threshold,
        "score_space": "sigmoid_probability_P(S=1)",
        "threshold_source_split": "final_training_subset",
        "threshold_selection": "max sensitivity_R + specificity_S over np.linspace(0,1,101); ties choose largest threshold",
        "train_specificity_S": float(threshold_result["spec"]),
        "train_sensitivity_R": float(threshold_result["sens"]),
        "train_balanced_accuracy": float(
            0.5 * (threshold_result["sens"] + threshold_result["spec"])
        ),
        "label_convention": "R=0,S=1",
        "decision_rule": "P(S)<threshold => R; P(S)>=threshold => S",
    }

    with open(output_dir / "decision_threshold.json", "w") as f:
        json.dump(threshold_metadata, f, indent=2)
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(
        f"Held-out test: auc={test_metrics['auc']:.3f} | "
        f"threshold={test_metrics['threshold']:.4f} | "
        f"acc={test_metrics['accuracy']:.3f} | "
        f"sens_R={test_metrics['sensitivity_R']:.3f} | "
        f"spec_S={test_metrics['specificity_S']:.3f} | "
        f"bal_acc={test_metrics['balanced_accuracy']:.3f}"
    )
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best model saved to: {best_dir}")
    print(f"Test metrics saved to: {output_dir / 'test_metrics.json'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
