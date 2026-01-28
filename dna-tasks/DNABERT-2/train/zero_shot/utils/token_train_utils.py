"""
Training utilities for DNABERT-2 CNN models on token embeddings.

This module provides training and evaluation utilities for drug resistance
prediction models, including early stopping, cross-validation, threshold
calculation, and metric computation.
"""

import os
import math
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader

from dataloader.locus_order import DRUGS as drugs
from downstream_cnn_model import *


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.

    Args:
        monitor: Metric to monitor ('val_loss' or 'val_auc').
        mode: 'min' for loss metrics, 'max' for accuracy-like metrics.
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        restore_best: Whether to restore the best model state.
    """

    def __init__(
        self, monitor='val_loss', mode='min', patience=5, 
        min_delta=1e-4, restore_best=True
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_value = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, current_value, model):
        """
        Check if training should stop.

        Args:
            current_value: Current value of the monitored metric.
            model: Model to save/restore.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        # Initialization
        if self.best_value is None:
            self.best_value = current_value
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
            return False

        # Calculate improvement
        if self.mode == 'min':
            improvement = self.best_value - current_value
        else:
            improvement = current_value - self.best_value

        # Check if improved
        if improvement > self.min_delta:
            self.best_value = current_value
            self.counter = 0
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(
                    f"Early stopping triggered ({self.monitor} has not "
                    f"improved for {self.patience} epochs)."
                )
                self.should_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_state)
                return True

        return False


def get_model_class(model_name, in_dim=768, seq_len=5000, num_classes=11, device='cuda'):
    """
    Instantiate a model based on the model name.

    Args:
        model_name: Name of the model class.
        in_dim: Input embedding dimension.
        seq_len: Maximum sequence length.
        num_classes: Number of output classes.
        device: Device to place model on.

    Returns:
        Initialized model on the specified device.

    Raises:
        ValueError: If model_name is unknown.
    """
    if model_name == 'MDDNABERTCNN':
        print("Using MDDNABERTCNN model")
        return MDDNABERTCNN(num_classes=num_classes, dropout_rate=0).to(device)
    elif model_name == 'DNABERTCNN':
        print("Using DNABERTCNN model")
        return DNABERTCNN(seq_len=seq_len, in_dim=in_dim, stem_out=64).to(device)
    elif model_name == 'DNABERTMLP':
        print("Using DNABERTMLP model")
        return DNABERTMLP(seq_len=seq_len, in_dim=in_dim).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_optimizer(model_parameters, learning_rate=5e-5, weight_decay=1e-5):
    """
    Create an Adam optimizer.

    Args:
        model_parameters: Model parameters to optimize.
        learning_rate: Learning rate for optimizer.
        weight_decay: L2 regularization coefficient.

    Returns:
        torch.optim.Adam: Configured optimizer.
    """
    return torch.optim.Adam(
        model_parameters, lr=learning_rate, weight_decay=weight_decay
    )


def calculate_alphas(res_phenotypes_label, weight=1.0):
    """
    Calculate alpha weighting matrix for multi-drug resistance.

    Args:
        res_phenotypes_label: Resistance labels tensor of shape (num_strains, num_drugs).
        weight: Weighting factor for sensitive samples.

    Returns:
        torch.Tensor: Alpha matrix of shape (num_strains, num_drugs) with weighted values.
    """
    num_strains, num_drugs = res_phenotypes_label.shape
    alphas = torch.zeros(num_drugs, dtype=torch.float32)
    alpha_matrix = torch.zeros_like(res_phenotypes_label, dtype=torch.float32)

    for drug_index, drug in enumerate(drugs):
        # Identify resistant (0) and sensitive (1) strains, ignoring unknowns (-1)
        resistant_mask = res_phenotypes_label[:, drug_index] == 0
        sensitive_mask = res_phenotypes_label[:, drug_index] == 1
        
        resistant_num = torch.sum(resistant_mask).item()
        sensitive_num = torch.sum(sensitive_mask).item()

        # Calculate alpha value for the drug
        if resistant_num + sensitive_num > 0:
            alphas[drug_index] = resistant_num / (resistant_num + sensitive_num)
        else:
            alphas[drug_index] = 0

        # Populate the alpha matrix with weighted values
        alpha_matrix[sensitive_mask, drug_index] = weight * alphas[drug_index]
        alpha_matrix[resistant_mask, drug_index] = -alphas[drug_index]

    return alpha_matrix


def train(
    model, train_loader, optimizer, criterion, acc_metric, 
    summary_writer, epochs=40, device='cuda'
):
    """
    Train model on training data.

    Args:
        model: Model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        acc_metric: Accuracy metric function.
        summary_writer: TensorBoard writer.
        epochs: Number of epochs.
        device: Device to use.

    Returns:
        tuple: (trained model, training history)
    """
    model.train()
    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_emb, batch_labels in train_loader:
            inputs = batch_emb.to(device)
            targets = batch_labels.to(device)
            alphas = calculate_alphas(targets).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            per_sample_loss = criterion(alphas, outputs)
            loss = torch.mean(per_sample_loss)
            accuracy = acc_metric(alphas, outputs)

            assert not torch.isnan(loss), "Loss is NaN"

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc_train = accuracy.item()
        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'acc': acc_train})
        print(
            f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} "
            f"Accuracy: {accuracy.item():.4f}"
        )

        if summary_writer:
            summary_writer.add_scalar('Loss/train', avg_loss, epoch)
            summary_writer.add_scalar('Accuracy/train', acc_train, epoch)

    return model, history


def token_embed_train(
    model, train_loader, val_loader, optimizer, criterion, 
    summary_writer, freeze_epoch, epochs=40, device='cuda'
):
    """
    Train model on token embeddings with validation and early stopping.

    Args:
        model: Model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        summary_writer: TensorBoard writer.
        freeze_epoch: Epoch to unfreeze bias layer.
        epochs: Number of epochs.
        device: Device to use.

    Returns:
        tuple: (trained model, training history)
    """
    history = []
    early_stopper = EarlyStopping(monitor='val_auc', mode='max', patience=5, min_delta=1e-4)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Unfreeze bias at specified epoch
        if epoch == freeze_epoch:
            model.fc_out.bias.requires_grad = True
            print(f"Unfreezing bias at epoch {epoch+1}")

        # ---- Training Phase ----
        model.train()
        train_loss = 0.0
        train_probs = []
        train_targets = []

        for batch_emb, batch_labels in tqdm(
            train_loader, total=len(train_loader), desc="Training", leave=False
        ):
            inputs = batch_emb.to(device)
            targets = batch_labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            assert not torch.isnan(loss), "Loss is NaN"

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Collect predictions for accuracy/AUC
            probs = torch.sigmoid(outputs).detach().cpu()
            train_probs.append(probs)
            train_targets.append(targets.detach().cpu())

        # Compute training metrics
        train_probs = torch.cat(train_probs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_accuracy = ((train_probs > 0.5) == train_targets).mean()
        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation Phase ----
        model.eval()
        val_probs = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                inputs = batch_emb.to(device)
                targets = batch_labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu()
                val_probs.append(probs)
                val_targets.append(targets.cpu())

        val_probs = torch.cat(val_probs).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_accuracy = ((val_probs > 0.5) == val_targets).mean()
        val_auc = roc_auc_score(val_targets, val_probs)
        avg_val_loss = val_loss / len(val_loader)

        # ---- Logging ----
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.3f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.3f} | "
            f"Val AUC: {val_auc:.3f}"
        )

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy,
            'val_auc': val_auc
        })

        # Early stopping
        if early_stopper.step(val_auc, model):
            print(
                f"Early stopping at epoch {epoch+1}. "
                f"Best val_auc: {early_stopper.best_value:.4f}"
            )
            break

        # TensorBoard logging
        if summary_writer:
            summary_writer.add_scalar('Loss/train', avg_train_loss, epoch)
            summary_writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            summary_writer.add_scalar('Loss/val', avg_val_loss, epoch)
            summary_writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            summary_writer.add_scalar('AUC/val', val_auc, epoch)

    return model, history


def evaluate(model, val_loader, device='cuda'):
    """
    Evaluate model on validation data.

    Args:
        model: Model to evaluate.
        val_loader: DataLoader for validation data.
        device: Device to use.

    Returns:
        tuple: (targets array, predictions array)
    """
    model.eval()
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for batch_emb, batch_labels in tqdm(val_loader, desc="Evaluating", leave=False):
            inputs = batch_emb.to(device, non_blocking=True)
            outputs = model(inputs).cpu().numpy()

            all_outputs.append(outputs.reshape(-1, 1))
            all_targets.append(batch_labels.cpu().numpy().reshape(-1, 1))

    y_val = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    return y_val, y_pred


def calculate_single_drug_threshold(y_train, y_train_pred, get_threshold_val=None):
    """
    Calculate decision threshold for a single drug based on training data.

    Args:
        y_train: True labels for training set.
        y_train_pred: Predicted probabilities for training set.
        get_threshold_val: Function to compute threshold.

    Returns:
        float: Optimal threshold value.
    """
    print("Calculating threshold from training set...")
    result = get_threshold_val(y_train, y_train_pred)
    threshold = result['threshold']
    print(
        f"Computed threshold: {threshold:.4f} | "
        f"Specificity: {result['spec']:.4f}, Sensitivity: {result['sens']:.4f}"
    )
    return threshold


def calculate_test_metrics_single_drug(
    y_test, y_test_pred, threshold, drug_name='Drug', 
    model_type="SD-DNABERT-CNN"
):
    """
    Calculate evaluation metrics for a single drug on test set.

    Args:
        y_test: True labels (values in {0, 1, -1}).
        y_test_pred: Predicted probabilities.
        threshold: Decision threshold.
        drug_name: Drug name for reporting.
        model_type: Model type for reporting.

    Returns:
        pd.DataFrame: Single row with test metrics.
    """
    column_names = [
        'Algorithm', 'Drug', 'num_sensitive', 'num_resistant', 
        'AUC', 'threshold', 'spec', 'sens'
    ]

    # Filter to non-missing phenotypes
    non_missing_idx = np.where(y_test != -1)[0]

    if len(non_missing_idx) == 0:
        print(f"No valid phenotypes found for {drug_name}")
        return pd.DataFrame(
            [[model_type, drug_name, 0, 0, np.nan, threshold, np.nan, np.nan]],
            columns=column_names
        )

    y_valid = y_test[non_missing_idx].astype(int)
    y_pred_valid = y_test_pred[non_missing_idx]

    num_sensitive = np.sum(y_valid == 1)
    num_resistant = np.sum(y_valid == 0)

    # Need both classes for evaluation
    if num_sensitive == 0 or num_resistant == 0:
        print(f"Only one class present (S={num_sensitive}, R={num_resistant})")
        return pd.DataFrame(
            [[model_type, drug_name, num_sensitive, num_resistant, np.nan, threshold, np.nan, np.nan]],
            columns=column_names
        )

    # Compute metrics
    auc = roc_auc_score(y_valid, y_pred_valid) if len(np.unique(y_valid)) > 1 else np.nan
    binary_pred = (y_pred_valid > threshold).astype(int)

    # Specificity = TN / (TN + FP), Sensitivity = TP / (TP + FN)
    specificity = (
        np.sum(np.logical_and(binary_pred == 1, y_valid == 1)) / num_sensitive
        if num_sensitive > 0 else np.nan
    )
    sensitivity = (
        np.sum(np.logical_and(binary_pred == 0, y_valid == 0)) / num_resistant
        if num_resistant > 0 else np.nan
    )

    return pd.DataFrame(
        [[model_type, drug_name, num_sensitive, num_resistant, auc, threshold, specificity, sensitivity]],
        columns=column_names
    )


def train_kfold_mod(
    dataset, drugs, criterion, learning_rate, weight_decay, 
    acc_metric, get_threshold_val, output_path, saved_model_path,
    model_name='DNABERTCNN', model_seq_len=5000, k_folds=5, 
    epochs=30, train_batch_size=64, val_batch_size=64, 
    random_seed=1, device='cuda'
):
    """
    Perform k-fold cross-validation training.

    Args:
        dataset: Dataset to train on.
        drugs: List of drugs.
        criterion: Loss function.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        acc_metric: Accuracy metric.
        get_threshold_val: Threshold calculation function.
        output_path: Path to save outputs.
        saved_model_path: Path to save models.
        model_name: Name of model class.
        model_seq_len: Model sequence length.
        k_folds: Number of folds.
        epochs: Number of epochs per fold.
        train_batch_size: Training batch size.
        val_batch_size: Validation batch size.
        random_seed: Random seed.
        device: Device to use.

    Returns:
        pd.DataFrame: Cross-validation results.
    """
    os.makedirs(output_path, exist_ok=True)

    results_df = pd.DataFrame(
        columns=[
            'Validation Split #', 'Algorithm', 'Drug', 'num_sensitive', 
            'num_resistant', 'AUC', 'AUC_PR', 'Threshold', 'Spec', 'Sens'
        ]
    )
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Fold {fold+1}/{k_folds} ===")

        model = get_model_class(
            model_name=model_name, seq_len=model_seq_len, device=device
        )
        optimizer = get_optimizer(
            model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay
        )

        summary_writer = SummaryWriter(
            log_dir=os.path.join(output_path, f"runs/cv_seed_{random_seed}/fold_{fold}")
        )

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=train_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=val_batch_size, shuffle=False
        )

        # Train
        model, history = train(
            model, train_loader, optimizer, criterion, acc_metric, 
            summary_writer, epochs, device
        )

        # Save model
        seed_path = os.path.join(saved_model_path, f"cv_seed_{random_seed}")
        os.makedirs(seed_path, exist_ok=True)
        save_path = os.path.join(seed_path, f"dnabert_{model_name}_cv_split_{fold}.pt")
        torch.save(model.state_dict(), save_path)

        # Save history
        history_path = os.path.join(output_path, f"cv_seed_{random_seed}")
        os.makedirs(history_path, exist_ok=True)
        pd.DataFrame(history).to_csv(
            os.path.join(history_path, f"history_cv_split_{fold}.csv"), index=False
        )

        # Evaluate
        y_val, y_pred = evaluate(model, val_loader, device)
        fold_metrics = compute_val_metrics_per_drug(y_val, y_pred, drugs, fold, get_threshold_val)

        for row in fold_metrics:
            results_df.loc[len(results_df)] = row

    # Final save
    results_df.to_csv(
        os.path.join(output_path, f"cv_seed_{random_seed}/crossval_auc.csv"), 
        index=False
    )
    print(f"\nK-Fold CV Complete. Results saved to {output_path}")
    return results_df


def compute_val_metrics_per_drug(y_val, y_pred, drugs, fold, get_threshold_val, model_type='SD-DNABERT-CNN'):
    """
    Compute validation metrics for each drug.

    Args:
        y_val: Validation targets.
        y_pred: Validation predictions.
        drugs: List of drugs.
        fold: Fold number.
        get_threshold_val: Threshold calculation function.
        model_type: Model type for reporting.

    Returns:
        list: List of metric rows.
    """
    results = []
    for idx, drug in enumerate(drugs):
        non_missing = np.where(y_val[:, idx] != -1)[0]
        if len(non_missing) == 0:
            print(f"[{drug}] Skipped (no data)")
            results.append([
                f"val_split{fold}", model_type, drug, 0, 0, 
                np.nan, np.nan, np.nan, np.nan, np.nan
            ])
            continue

        y_true = y_val[non_missing, idx]
        y_score = y_pred[non_missing, idx]
        num_sensitive = np.sum(y_true == 1)
        num_resistant = np.sum(y_true == 0)

        if num_sensitive == 0 or num_resistant == 0:
            results.append([
                f"val_split{fold}", model_type, drug, num_sensitive, 
                num_resistant, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
            continue

        auc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(1 - y_true, 1 - y_score)
        thresh = get_threshold_val(y_true, y_score)
        results.append([
            f"val_split{fold}", model_type, drug, num_sensitive, num_resistant, 
            auc, auc_pr, thresh["threshold"], thresh["spec"], thresh["sens"]
        ])

    return results


def calculate_auc_thresholds(y_train, y_train_pred, get_threshold_val, thresholds_path=None):
    """
    Calculate thresholds for each drug based on training data.

    Args:
        y_train: Training targets.
        y_train_pred: Training predictions.
        get_threshold_val: Threshold calculation function.
        thresholds_path: Path to save thresholds.

    Returns:
        tuple: (threshold DataFrame, drug to threshold dict)
    """
    print("Calculating thresholds for each drug...")
    threshold_data = []

    for idx, drug in enumerate(drugs):
        print(f"Calculating threshold for {drug}...")
        train_metrics = get_threshold_val(y_train[:, idx], y_train_pred[:, idx])
        train_metrics["drug"] = drug
        threshold_data.append(train_metrics)

    threshold_df = pd.DataFrame(threshold_data)
    drug_to_threshold = {x: y for x, y in zip(threshold_df.drug, threshold_df.threshold)}

    return threshold_df, drug_to_threshold


def calculate_test_auc(y, y_pred, drug_to_threshold, model_type='SD-DNABERT-CNN'):
    """
    Compute AUC, sensitivity, specificity for test set using pre-computed thresholds.

    Args:
        y: Test targets.
        y_pred: Test predictions.
        drug_to_threshold: Drug to threshold mapping.
        model_type: Model type for reporting.

    Returns:
        pd.DataFrame: Test metrics for each drug.
    """
    column_names = [
        'Algorithm', 'Drug', 'num_sensitive', 'num_resistant', 
        'AUC', 'threshold', 'spec', 'sens'
    ]
    results = []

    for idx, drug in enumerate(drugs):
        print(f"Calculating test metrics for drug: {drug}")

        threshold = float(drug_to_threshold[drug])
        non_missing = np.where(y[:, idx] != -1)[0]

        if len(non_missing) == 0:
            print(f"No valid data for drug: {drug}")
            results.append([model_type, drug, 0, 0, np.nan, threshold, np.nan, np.nan])
            continue

        y_true = y[non_missing, idx]
        y_score = y_pred[non_missing, idx]

        num_sensitive = np.sum(y_true == 1)
        num_resistant = np.sum(y_true == 0)

        if num_sensitive == 0 or num_resistant == 0:
            results.append([
                model_type, drug, num_sensitive, num_resistant, 
                np.nan, threshold, np.nan, np.nan
            ])
            continue

        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
        binary_pred = (y_score > threshold).astype(int)

        spec = (
            np.sum((binary_pred == 1) & (y_true == 1)) / num_sensitive 
            if num_sensitive > 0 else np.nan
        )
        sens = (
            np.sum((binary_pred == 0) & (y_true == 0)) / num_resistant 
            if num_resistant > 0 else np.nan
        )

        results.append([model_type, drug, num_sensitive, num_resistant, auc, threshold, spec, sens])

    return pd.DataFrame(results, columns=column_names)


def conditionally_standardize_embeddings(embeddings, std_threshold=0.2):
    """
    Conditionally standardize embeddings based on standard deviation threshold.

    Args:
        embeddings: Input embeddings tensor.
        std_threshold: Threshold for triggering standardization.

    Returns:
        torch.Tensor: Optionally standardized embeddings.
    """
    mean = embeddings.mean()
    std = embeddings.std()

    print(f"Initial Embeddings Mean: {mean:.6f}, Std: {std:.6f}")

    if std < std_threshold:
        print(f"Standardizing embeddings (std < {std_threshold})...")
        embeddings = (embeddings - mean) / std
        print(f"Standardized Embeddings Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}")
    else:
        print(f"Standardization not applied (std >= {std_threshold})")

    return embeddings


def train_on_token_embeddings(
    train_loader, val_loader, drug, num_sensitive, num_resistant,
    criterion, learning_rate, weight_decay, output_path, saved_model_path,
    model_name='DNABERTCNN', model_dim=768, model_seq_len=5000, k_folds=5,
    epochs=30, train_batch_size=64, val_batch_size=64,
    freeze_bias_frac=0.25, random_seed=1, device='cuda'
):
    """
    Train a model on token embeddings for a specific drug.

    Args:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        drug: Drug name.
        num_sensitive: Number of sensitive samples.
        num_resistant: Number of resistant samples.
        criterion: Loss function.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        output_path: Path to save outputs.
        saved_model_path: Path to save models.
        model_name: Model class name.
        model_dim: Model embedding dimension.
        model_seq_len: Model sequence length.
        k_folds: Number of folds.
        epochs: Number of epochs.
        train_batch_size: Training batch size.
        val_batch_size: Validation batch size.
        freeze_bias_frac: Fraction of epochs before unfreezing bias.
        random_seed: Random seed.
        device: Device to use.
    """
    model = get_model_class(
        model_name=model_name, in_dim=model_dim, 
        seq_len=model_seq_len, device=device
    )
    optimizer = get_optimizer(
        model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay
    )

    summary_writer = SummaryWriter(
        log_dir=os.path.join(output_path, f"runs/{drug}/seed_{random_seed}")
    )

    # Initialize bias
    freeze_epoch = max(1, int(epochs * freeze_bias_frac))
    with torch.no_grad():
        res_prob = num_resistant / (num_resistant + num_sensitive + 1e-8)
        model.fc_out.bias.fill_(math.log(res_prob / (1 - res_prob)))
    model.fc_out.bias.requires_grad = False

    model, history = token_embed_train(
        model, train_loader, val_loader, optimizer, criterion,
        summary_writer, freeze_epoch, epochs, device
    )
    print(f"Training complete for drug: {drug}\n")

    # Save model
    seed_path = os.path.join(saved_model_path, f"{drug}/seed_{random_seed}")
    os.makedirs(seed_path, exist_ok=True)

    print("fc1.weight shape:", model.fc1.weight.shape)
    print("fc1.bias shape:", model.fc1.bias.shape)

    save_path = os.path.join(seed_path, f"{model_name}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")

    # Save history
    history_path = os.path.join(output_path, f"{drug}/seed_{random_seed}")
    os.makedirs(history_path, exist_ok=True)
    pd.DataFrame(history).to_csv(
        os.path.join(history_path, f"{model_name}_history.csv"), index=False
    )
    print(f"History saved to {history_path}")


def cross_val_train_on_token_embeddings(
    dataset, drug, num_sensitive, num_resistant, criterion, learning_rate,
    weight_decay, output_path, saved_model_path, model_name='DNABERTCNN',
    model_dim=768, model_seq_len=5000, k_folds=5, epochs=30,
    train_batch_size=64, val_batch_size=64, freeze_bias_frac=0.25,
    random_seed=42, device='cuda'
):
    """
    Train a model on token embeddings for a specific drug using k-fold cross-validation.

    Args:
        dataset: Dataset to train on.
        drug: Drug name.
        num_sensitive: Number of sensitive samples.
        num_resistant: Number of resistant samples.
        criterion: Loss function.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        output_path: Path to save outputs.
        saved_model_path: Path to save models.
        model_name: Model class name.
        model_dim: Model embedding dimension.
        model_seq_len: Model sequence length.
        k_folds: Number of folds.
        epochs: Number of epochs per fold.
        train_batch_size: Training batch size.
        val_batch_size: Validation batch size.
        freeze_bias_frac: Fraction of epochs before unfreezing bias.
        random_seed: Random seed.
        device: Device to use.

    Args:
        dataset: Dataset to train on.
        drug: Drug name.
        num_sensitive: Number of sensitive samples.
        num_resistant: Number of resistant samples.
        criterion: Loss function.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        output_path: Path to save outputs.
        saved_model_path: Path to save models.
        model_name: Model class name.
        model_dim: Model embedding dimension.
        model_seq_len: Model sequence length.
        k_folds: Number of folds.
        epochs: Number of epochs per fold.
        train_batch_size: Training batch size.
        val_batch_size: Validation batch size.
        freeze_bias_frac: Fraction of epochs before unfreezing bias.
        random_seed: Random seed.
        device: Device to use.
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    all_histories = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f"\n==== Fold {fold+1}/{k_folds} for drug: {drug} ====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=val_batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )

        model = get_model_class(
            model_name=model_name, in_dim=model_dim, 
            seq_len=model_seq_len, device=device
        )
        optimizer = get_optimizer(
            model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay
        )

        log_dir = os.path.join(output_path, f"runs/{drug}/seed_{random_seed}/fold_{fold+1}")
        summary_writer = SummaryWriter(log_dir=log_dir)

        # Initialize bias
        freeze_epoch = max(1, int(epochs * freeze_bias_frac))
        with torch.no_grad():
            res_prob = num_resistant / (num_resistant + num_sensitive + 1e-8)
            model.fc_out.bias.fill_(math.log(res_prob / (1 - res_prob)))
        model.fc_out.bias.requires_grad = False

        # Train
        model, history = token_embed_train(
            model, train_loader, val_loader, optimizer, criterion,
            summary_writer, freeze_epoch, epochs, device
        )

        all_histories.append(pd.DataFrame(history))
        fold_models.append(model)

        # Save model
        fold_path = os.path.join(saved_model_path, f"{drug}/seed_{random_seed}/fold_{fold+1}")
        os.makedirs(fold_path, exist_ok=True)

        print("fc1.weight shape:", model.fc1.weight.shape)
        print("fc1.bias shape:", model.fc1.bias.shape)

        save_path = os.path.join(fold_path, f"{model_name}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Model for fold {fold+1} saved to {save_path}")

        # Save history
        hist_path = os.path.join(output_path, f"{drug}/seed_{random_seed}")
        os.makedirs(hist_path, exist_ok=True)
        all_histories[-1].to_csv(
            os.path.join(hist_path, f"{model_name}_fold{fold+1}_history.csv"), 
            index=False
        )
        print(f"History for fold {fold+1} saved to {hist_path}")

    print("\nCross-validation complete.")
