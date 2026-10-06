import os
import pandas as pd
import numpy as np
import torch
import copy
import math
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
from dataloader.locus_order import DRUGS as drugs
from downstream_cnn_model import *

START_FOLD = {
    "ETHAMBUTOL": 3,
}

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, monitor='val_loss', mode='min', patience=5, min_delta=1e-4, restore_best=True):
        self.monitor = monitor
        self.mode = mode  # 'min' for loss, 'max' for metrics like AUC
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_value = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, current_value, model):
        # Initialization
        if self.best_value is None:
            self.best_value = current_value
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
            return False

        # Improvement condition
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
                print(f"Early stopping triggered ({self.monitor} has not improved for {self.patience} epochs).")
                self.should_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_state)
                return True
        return False


class TrainingLossEarlyStopping:
    """
    Early stopping based on smoothed training loss with relative improvement threshold.
    
    This implementation matches SD-CNN's early stopping strategy:
    - Uses only training loss (never validation/test data)
    - Smooths loss over a moving window
    - Requires minimum relative improvement (as a fraction)
    - Has a minimum number of epochs before early stopping can trigger
    - Saves checkpoint when loss improves
    
    Parameters
    ----------
    checkpoint_path : str
        Path to save model checkpoints
    min_epochs : int
        Minimum number of epochs before early stopping can trigger (default: 5)
    patience : int
        Number of epochs to wait without improvement before stopping (default: 5)
    min_relative_improvement : float
        Minimum relative improvement threshold as a fraction (default: 1e-3 = 0.1%)
    smoothing_window : int
        Window size for smoothing training loss (default: 3)
    """
    def __init__(
        self,
        checkpoint_path,
        min_epochs=5,
        patience=5,
        min_relative_improvement=1e-3,
        smoothing_window=3,
    ):
        self.checkpoint_path = checkpoint_path
        # Path for resumable checkpoint (different from best model checkpoint)
        self.resume_checkpoint_path = checkpoint_path.replace('.pt', '_resume.pt')
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.smoothing_window = smoothing_window
        self.train_losses = []
        self.best_smoothed_loss = None
        self.best_epoch = None
        self.relative_improvement = np.nan
        self.patience_counter = 0
        self.stopped_epoch = None

    def step(self, epoch, mean_train_loss, model, optimizer):
        """
        Check if training should stop based on smoothed training loss.
        
        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed)
        mean_train_loss : float
            Mean training loss for the current epoch
        model : torch.nn.Module
            The model to save if loss improves
        optimizer : torch.optim.Optimizer
            The optimizer (state will be saved for resumption)
            
        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        self.train_losses.append(mean_train_loss)

        # Compute smoothed loss over the window
        window_losses = self.train_losses[-self.smoothing_window :]
        smoothed_loss = float(np.mean(window_losses))

        # Check for improvement
        if self.best_smoothed_loss is None:
            improved = True
            self.relative_improvement = np.inf
        else:
            self.relative_improvement = (
                self.best_smoothed_loss - smoothed_loss
            ) / max(abs(self.best_smoothed_loss), 1e-12)
            improved = self.relative_improvement >= self.min_relative_improvement

        current_epoch = epoch + 1
        if improved:
            self.best_smoothed_loss = smoothed_loss
            self.best_epoch = current_epoch
            self.patience_counter = 0
            # Save best model checkpoint (for final model loading)
            torch.save(model.state_dict(), self.checkpoint_path)
        elif current_epoch > self.min_epochs:
            self.patience_counter += 1
        
        # Always save resumable checkpoint after each epoch (for wall time recovery)
        self._save_resume_checkpoint(epoch, model, optimizer)

        # Log progress
        print(
            f"epoch={current_epoch} "
            f"mean_train_loss={mean_train_loss:.6f} "
            f"smoothed_train_loss={smoothed_loss:.6f} "
            f"best_smoothed_train_loss={self.best_smoothed_loss:.6f} "
            f"relative_improvement={self.relative_improvement:.6g} "
            f"patience_counter={self.patience_counter}"
        )

        # Check if we should stop
        if current_epoch > self.min_epochs and self.patience_counter >= self.patience:
            self.stopped_epoch = current_epoch
            threshold_pct = self.min_relative_improvement * 100
            print(
                f"Training converged at epoch {current_epoch}.\n"
                f"Smoothed training loss did not improve by >= {threshold_pct:.1f}%\n"
                f"for {self.patience} consecutive epochs.\n"
                f"Best checkpoint was from epoch {self.best_epoch}."
            )
            return True
        
        return False
    
    def _save_resume_checkpoint(self, epoch, model, optimizer):
        """
        Save a complete checkpoint for resuming training after interruption.
        
        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed)
        model : torch.nn.Module
            Model to save
        optimizer : torch.optim.Optimizer
            Optimizer to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': self.train_losses,
            'best_smoothed_loss': self.best_smoothed_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'relative_improvement': self.relative_improvement,
        }
        torch.save(checkpoint, self.resume_checkpoint_path)
    
    def restore_from_checkpoint(self, model, optimizer):
        """
        Restore training state from a resumable checkpoint.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to load state into
        optimizer : torch.optim.Optimizer
            Optimizer to load state into
            
        Returns
        -------
        int or None
            The epoch to resume from (next epoch to run), or None if no checkpoint exists
        """
        if not os.path.exists(self.resume_checkpoint_path):
            print(f"No resume checkpoint found at {self.resume_checkpoint_path}. Starting from scratch.")
            return None
        
        print(f"Found resume checkpoint at {self.resume_checkpoint_path}. Restoring training state...")
        checkpoint = torch.load(self.resume_checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.best_smoothed_loss = checkpoint['best_smoothed_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.patience_counter = checkpoint['patience_counter']
        self.relative_improvement = checkpoint.get('relative_improvement', np.nan)
        
        resume_epoch = checkpoint['epoch'] + 1  # Next epoch to run
        print(f"Resumed from epoch {checkpoint['epoch'] + 1}. Continuing from epoch {resume_epoch + 1}...")
        print(f"  Best smoothed loss so far: {self.best_smoothed_loss:.6f} (epoch {self.best_epoch})")
        print(f"  Patience counter: {self.patience_counter}/{self.patience}")
        
        return resume_epoch


class ValidationAUCEarlyStopping:
    """
    Early stopping based on validation AUC with resume checkpoint support.

    Parameters
    ----------
    checkpoint_path : str
        Path to save best model checkpoint
    patience : int
        Number of epochs to wait without sufficient AUC improvement
    min_delta : float
        Minimum absolute AUC improvement to count as better
    min_epochs : int
        Minimum epochs before early stopping can trigger
    """

    def __init__(
        self,
        checkpoint_path,
        patience=5,
        min_delta=1e-4,
        min_epochs=0,
    ):
        self.checkpoint_path = checkpoint_path
        self.resume_checkpoint_path = checkpoint_path.replace('.pt', '_resume.pt')
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_auc = None
        self.best_epoch = None
        self.patience_counter = 0
        self.stopped_epoch = None

    def step(self, epoch, val_auc, model, optimizer):
        """Update AUC early stopping state for one epoch."""
        current_epoch = epoch + 1

        if self.best_auc is None:
            improved = True
        else:
            improved = (val_auc - self.best_auc) > self.min_delta

        if improved:
            self.best_auc = float(val_auc)
            self.best_epoch = current_epoch
            self.patience_counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        elif current_epoch > self.min_epochs:
            self.patience_counter += 1

        self._save_resume_checkpoint(epoch, model, optimizer)

        print(
            f"epoch={current_epoch} "
            f"val_auc={val_auc:.6f} "
            f"best_val_auc={self.best_auc:.6f} "
            f"patience_counter={self.patience_counter}"
        )

        if current_epoch > self.min_epochs and self.patience_counter >= self.patience:
            self.stopped_epoch = current_epoch
            print(
                f"Validation-AUC early stopping triggered at epoch {current_epoch}.\n"
                f"Validation AUC did not improve by > {self.min_delta} for {self.patience} consecutive epochs.\n"
                f"Best checkpoint was from epoch {self.best_epoch} with val_auc={self.best_auc:.6f}."
            )
            return True

        return False

    def _save_resume_checkpoint(self, epoch, model, optimizer):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
        }
        torch.save(checkpoint, self.resume_checkpoint_path)

    def restore_from_checkpoint(self, model, optimizer):
        if not os.path.exists(self.resume_checkpoint_path):
            print(f"No resume checkpoint found at {self.resume_checkpoint_path}. Starting from scratch.")
            return None

        print(f"Found resume checkpoint at {self.resume_checkpoint_path}. Restoring training state...")
        checkpoint = torch.load(self.resume_checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_auc = checkpoint.get('best_auc', None)
        self.best_epoch = checkpoint.get('best_epoch', None)
        self.patience_counter = checkpoint.get('patience_counter', 0)

        resume_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch'] + 1}. Continuing from epoch {resume_epoch + 1}...")
        if self.best_auc is not None:
            print(f"  Best val_auc so far: {self.best_auc:.6f} (epoch {self.best_epoch})")
        print(f"  Patience counter: {self.patience_counter}/{self.patience}")

        return resume_epoch


def get_model_class(model_name, in_dim=768, seq_len=5000, num_classes=11, device='cuda'):
    """
    Get the model class based on the model name.
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
    Get the optimizer based on the model and arguments.
    """
    return torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    

def calculate_alphas(res_phenotypes_label, weight=1.0):
        # Get the phenotype label for the batch index
        num_strains, num_drugs = res_phenotypes_label.shape 
        # print(f"num_strains: {num_strains}, num_drugs: {num_drugs}")

        alphas = torch.zeros(num_drugs, dtype=torch.float32)
        alpha_matrix = torch.zeros_like(res_phenotypes_label, dtype=torch.float32)
        
        for drug_index, drug in enumerate(drugs):
            # Identify resistant (0) and sensitive (1) strains, ignoring unknowns (-1)
            resistant_mask = res_phenotypes_label[:, drug_index] == 0
            sensitive_mask = res_phenotypes_label[:, drug_index] == 1
            unknown_mask = res_phenotypes_label[:, drug_index] == -1
            
            # Count the number of resistant and sensitive strains
            resistant_num = torch.sum(resistant_mask).item()
            sensitive_num = torch.sum(sensitive_mask).item()
            unknown_num = torch.sum(unknown_mask).item()
            
            # Calculate alpha value for the drug, handling cases where both counts are zero
            if resistant_num + sensitive_num > 0:
                alphas[drug_index] = resistant_num / (resistant_num + sensitive_num)
            else:
                alphas[drug_index] = 0

            # Populate the alpha matrix with weighted values
            alpha_matrix[sensitive_mask, drug_index] = weight * alphas[drug_index]
            alpha_matrix[resistant_mask, drug_index] = -alphas[drug_index]

        # print(f"alpha matrix shape: {alpha_matrix.shape}\n")

        return alpha_matrix


def train(model, train_loader, optimizer, criterion, acc_metric, summary_writer, epochs=40, device='cuda'):
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


            # After loss computation
            if torch.isnan(loss).any():
                # check min and max value of outputs
                print("Min/Max Output:", outputs.min().item(), outputs.max().item())
                print("Loss before backward:", loss.item())
                
            assert not torch.isnan(loss), "Loss is NaN"

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc_train = accuracy.item()
        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'acc': acc_train})
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy.item():.4f}")

        if summary_writer:
            summary_writer.add_scalar('Loss/train', avg_loss, epoch)
            summary_writer.add_scalar('Accuracy/train', acc_train, epoch)

    return model, history

def token_embed_train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    summary_writer,
    freeze_epoch,
    epochs=40,
    device='cuda',
    checkpoint_path=None,
    early_stopping_min_epochs=5,
    early_stopping_patience=5,
    early_stopping_min_relative_improvement=1e-3,
    early_stopping_smoothing_window=3,
    use_training_loss_early_stopping=True,
    use_auc_early_stopping=False,
):
    """
    Train a model on token embeddings with optional training-loss-based early stopping.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    optimizer : torch.optim.Optimizer
        Optimizer for training
    criterion : torch.nn.Module
        Loss function
    summary_writer : SummaryWriter
        TensorBoard writer
    freeze_epoch : int
        Epoch at which to unfreeze bias
    epochs : int
        Maximum number of epochs (default: 40)
    device : str
        Device to use (default: 'cuda')
    checkpoint_path : str or None
        Path to save best model checkpoint. If None, checkpointing is disabled.
    early_stopping_min_epochs : int
        Minimum epochs before early stopping can trigger (default: 5)
    early_stopping_patience : int
        Patience for early stopping (default: 5)
    early_stopping_min_relative_improvement : float
        Minimum relative improvement threshold (default: 1e-3)
    early_stopping_smoothing_window : int
        Window size for smoothing training loss (default: 3)
    use_training_loss_early_stopping : bool
        If True, use training-loss-based early stopping (like SD-CNN).
        If False, use validation-AUC-based early stopping (old behavior).
        (default: True)
    
    Returns
    -------
    model : torch.nn.Module
        Trained model with best checkpoint loaded
    history : list of dict
        Training history
    """
    history = []
    
    # Initialize early stopping
    start_epoch = 0
    if use_auc_early_stopping:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided when using AUC early stopping")
        early_stopper = ValidationAUCEarlyStopping(
            checkpoint_path=checkpoint_path,
            patience=early_stopping_patience,
            min_delta=1e-4,
            min_epochs=early_stopping_min_epochs,
        )
        resumed_epoch = early_stopper.restore_from_checkpoint(model, optimizer)
        if resumed_epoch is not None:
            start_epoch = resumed_epoch
    elif use_training_loss_early_stopping:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided when using training-loss early stopping")
        early_stopper = TrainingLossEarlyStopping(
            checkpoint_path=checkpoint_path,
            min_epochs=early_stopping_min_epochs,
            patience=early_stopping_patience,
            min_relative_improvement=early_stopping_min_relative_improvement,
            smoothing_window=early_stopping_smoothing_window,
        )
        # Try to resume from checkpoint
        resumed_epoch = early_stopper.restore_from_checkpoint(model, optimizer)
        if resumed_epoch is not None:
            start_epoch = resumed_epoch
    else:
        # Use old validation-based early stopping
        early_stopper = EarlyStopping(monitor='val_auc', mode='max', patience=5, min_delta=1e-4)


    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        i=0
        #---- Train Phase ----#
        if epoch==freeze_epoch: 
            model.fc_out.bias.requires_grad=True
            print(f"Unfreezing bias at epoch {epoch+1}")
        
        model.train()
        train_loss = 0.0
        train_probs = []
        train_targets = []

        for batch_emb, batch_labels in tqdm(train_loader, total=len(train_loader), desc="Training", leave=False):
            inputs = batch_emb.to(device)
            targets = batch_labels.to(device).float()  # Ensure targets are float for BCEWithLogitsLoss

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets) # this will be a mean loss of samples across the batch on using BCEWithLogitsLoss directly

            # After loss computation
            if torch.isnan(loss).any():
                # check min and max value of outputs
                print("Min/Max Output:", outputs.min().item(), outputs.max().item())
                print("Loss before backward:", loss.item())
                
            assert not torch.isnan(loss), "Loss is NaN"

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Collect predictions for accuracy/AUC
            probs = torch.sigmoid(outputs).detach().cpu()
            train_probs.append(probs)
            train_targets.append(targets.detach().cpu())

            i+=1

            # if i == 10:
            #     break


        # Obtaining train metrics
        train_probs = torch.cat(train_probs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_accuracy = ((train_probs > 0.5) == train_targets).mean()
        avg_train_loss = train_loss / len(train_loader)


        #---- Validation Phase ----#
        model.eval()
        val_probs = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                inputs = batch_emb.to(device)
                targets = batch_labels.to(device)

                outputs = model(inputs)  # logits
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
        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.3f} \n "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.3f} | Val AUC: {val_auc:.3f}"
            )
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy,
            'val_auc': val_auc
        })

        # Early stopping step
        if use_auc_early_stopping:
            if early_stopper.step(epoch, val_auc, model, optimizer):
                print(f"Early stopping triggered. Loading best model from epoch {early_stopper.best_epoch}.")
                break
        elif use_training_loss_early_stopping:
            # Use training loss for early stopping (SD-CNN style)
            if early_stopper.step(epoch, avg_train_loss, model, optimizer):
                print(f"Early stopping triggered. Loading best model from epoch {early_stopper.best_epoch}.")
                break
        else:
            # Use validation AUC for early stopping (old behavior)
            if early_stopper.step(val_auc, model):
                print(f"Early stopping at epoch {epoch+1} due to negligible increase in validation AUC. Restoring best model with val_loss: {early_stopper.best_value:.4f}")
                break

        # ---- Logging to TensorBoard ----
        if summary_writer:
            summary_writer.add_scalar('Loss/train', avg_train_loss, epoch)
            summary_writer.add_scalar('Accuracy/train', train_accuracy, epoch)

            summary_writer.add_scalar('Loss/val', avg_val_loss, epoch)
            summary_writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            summary_writer.add_scalar('AUC/val', val_auc, epoch)

    # Load best checkpoint if using training-loss or AUC early stopping
    if (use_training_loss_early_stopping or use_auc_early_stopping) and early_stopper.best_epoch is not None:
        print(f"Loading best model checkpoint from epoch {early_stopper.best_epoch}")
        model.load_state_dict(torch.load(checkpoint_path))

    return model, history


def evaluate(model, val_loader, device='cuda'):
    model.eval()
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for batch_emb, batch_labels in tqdm(val_loader, desc="Evaluating", leave=False):
            inputs = batch_emb.to(device, non_blocking=True)
            outputs = model(inputs).cpu().numpy()

            # flatten both outputs and labels so shapes always match
            all_outputs.append(outputs.reshape(-1, 1))
            all_targets.append(batch_labels.cpu().numpy().reshape(-1, 1))

    y_val = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    return y_val, y_pred


def calculate_single_drug_threshold(y_train, y_train_pred, get_threshold_val=None):
    """
    Calculate threshold for a single drug based on training data.
    
    Parameters
    ----------
    y_train: np.ndarray
        True labels for training set (1D array)
    y_train_pred: np.ndarray
        Predicted probabilities for training set (1D array)
    get_threshold_val: callable, ThresholdValue module, or None
        Function/module to compute threshold. If None, uses the get_threshold_val function defined in this module.
    
    Returns
    -------
    float: Optimal threshold value
    """
    print("Calculating threshold from training set...")
    
    # If no function provided, use the one defined in this module
    # if get_threshold_val is None:
    #     threshold_func = default_threshold_func
    # elif hasattr(get_threshold_val, 'forward'):
    #     # It's a ThresholdValue module - wrap it
    #     threshold_func = lambda y_true, y_pred: get_threshold_val.forward(y_true, y_pred)
    # else:
    #     # It's already a callable function
          # threshold_func = get_threshold_val
    
    result = get_threshold_val(y_train, y_train_pred)
    
    threshold = result['threshold']
    print(f"Computed threshold: {threshold:.4f}")
    print(f"  Specificity: {result['spec']:.4f}, Sensitivity: {result['sens']:.4f}")
    return threshold


def calculate_test_metrics_single_drug(y_test, y_test_pred, threshold, drug_name='Drug', model_type="SD-DNABERT-CNN"):
    """
    Calculate evaluation metrics for a single drug on test set.
    
    Parameters
    ----------
    y_test: np.ndarray
        True labels for test set (1D array, values in {0, 1, -1})
    y_test_pred: np.ndarray
        Predicted probabilities for test set (1D array)
    threshold: float
        Decision threshold for classification
    drug_name: str
        Name of the drug (for reporting)
    
    Returns
    -------
    pd.DataFrame: Single row with test metrics
    """
    column_names = ['Algorithm', 'Drug', 'num_sensitive', 'num_resistant', 'AUC', 'threshold', 'spec', 'sens']
    
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
        print(f"Only one class present in test set (S={num_sensitive}, R={num_resistant})")
        return pd.DataFrame(
            [[model_type, drug_name, num_sensitive, num_resistant, np.nan, threshold, np.nan, np.nan]],
            columns=column_names
        )
    
    # Compute metrics
    auc = roc_auc_score(y_valid, y_pred_valid) if len(np.unique(y_valid)) > 1 else np.nan
    
    # Binarize predictions using threshold
    binary_pred = (y_pred_valid > threshold).astype(int)
    
    # Specificity = TN / (TN + FP) = correctly classified sensitive (1)
    specificity = np.sum(np.logical_and(binary_pred == 1, y_valid == 1)) / num_sensitive if num_sensitive > 0 else np.nan
    
    # Sensitivity = TP / (TP + FN) = correctly classified resistant (0)
    sensitivity = np.sum(np.logical_and(binary_pred == 0, y_valid == 0)) / num_resistant if num_resistant > 0 else np.nan
    
    return pd.DataFrame(
        [[model_type, drug_name, num_sensitive, num_resistant, auc, threshold, specificity, sensitivity]],
        columns=column_names
    )


def train_kfold_mod(dataset,
                drugs,
                criterion,
                learning_rate,
                weight_decay,
                acc_metric,
                get_threshold_val,
                output_path,
                saved_model_path,
                model_name='DNABERTCNN',
                model_seq_len=5000,
                k_folds=5,
                epochs=30,
                train_batch_size=64,
                val_batch_size=64,
                random_seed=1,
                device='cuda'):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_df = pd.DataFrame(columns=['Validation Split #', 'Algorithm', 'Drug', "num_sensitive", "num_resistant", 'AUC', 'AUC_PR', "Threshold", "Spec", "Sens"])
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Fold {fold+1}/{k_folds} ===")

        # Initialize model and optimizer
        model = get_model_class(model_name=model_name, seq_len=model_seq_len, device=device)
        optimizer = get_optimizer(model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)

        # Set up TensorBoard logging
        summary_writer = SummaryWriter(log_dir=os.path.join(output_path, f"runs/cv_seed_{random_seed}/fold_{fold}"))

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=val_batch_size, shuffle=False)

        # Train
        model, history = train(model, train_loader, optimizer, criterion, acc_metric, summary_writer, epochs, device)

    
        # Ensure the saved model path exists
        seed_path = os.path.join(saved_model_path, f"cv_seed_{random_seed}")
        os.makedirs(seed_path, exist_ok=True)

        # Save the model state dictionary
        save_path = os.path.join(seed_path, f"dnabert_{model_name}_cv_split_{fold}.pt")
        torch.save(model.state_dict(), save_path)

        # Save the training history as a CSV
        history_path = os.path.join(output_path, f"cv_seed_{random_seed}")
        os.makedirs(history_path, exist_ok=True)
        pd.DataFrame(history).to_csv(os.path.join(history_path, f"history_cv_split_{fold}.csv"), index=False)

        # Evaluate
        y_val, y_pred = evaluate(model, val_loader, device)
        fold_metrics = compute_val_metrics_per_drug(y_val, y_pred, drugs, fold, get_threshold_val)

        # Log
        for row in fold_metrics:
            results_df.loc[len(results_df)] = row

    # Final save
    results_df.to_csv(os.path.join(output_path, f"cv_seed_{random_seed}/crossval_auc.csv"), index=False)
    print(f"\n K-Fold CV Complete. Results saved to {output_path}")
    return results_df


def compute_val_metrics_per_drug(y_val, y_pred, drugs, fold, get_threshold_val, model_type='SD-DNABERT-CNN'):

    results = []
    for idx, drug in enumerate(drugs):
        non_missing = np.where(y_val[:, idx] != -1)[0]
        if len(non_missing) == 0:
            print(f"[{drug}] Skipped (no data)")
            results.append([f"val_split{fold}", model_type, drug, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        y_true = y_val[non_missing, idx]
        y_score = y_pred[non_missing, idx]
        num_sensitive = np.sum(y_true == 1)
        num_resistant = np.sum(y_true == 0)

        if num_sensitive == 0 or num_resistant == 0:
            results.append([f"val_split{fold}", model_type, drug, num_sensitive, num_resistant, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        auc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(1 - y_true, 1 - y_score)
        thresh = get_threshold_val(y_true, y_score)
        results.append([f"val_split{fold}", model_type, drug, num_sensitive, num_resistant, auc, auc_pr, thresh["threshold"], thresh["spec"], thresh["sens"]])

    return results


# Threshold selection for each drug based on training data
def calculate_auc_thresholds(y_train, y_train_pred, get_threshold_val, thresholds_path=None):
    """
    Calculate the thresholds for each drug based on the training data
    Parameters
    ----------
    y_train: np.array
        actual values for y
    y_train_pred: np.array
        predicted values for y
    thresholds_path: str
        Path to save the thresholds

    Returns
    -------
    pd.DataFrame with thresholds for each drug
    Drug to threshold mapping dict
    """
    
    print("Calculating thresholds for each drug...")
    threshold_data = []

    for idx, drug in enumerate(drugs):
        print(f"Calculating threshold for {drug}...")
        train_metrics = get_threshold_val(y_train[:, idx], y_train_pred[:, idx])
        train_metrics["drug"] = drug
        threshold_data.append(train_metrics)

    threshold_df = pd.DataFrame(threshold_data)

    drug_to_threshold = {x:y for x,y in zip(threshold_df.drug, threshold_df.threshold)}

    return threshold_df, drug_to_threshold


def calculate_test_auc(y, y_pred, drug_to_threshold, model_type='SD-DNABERT-CNN'):
    """
    Computes the AUC, sensitivity, specificity, for given threshold

    Parameters
    ----------
    y_train: np.array
        actual values for y
    y_train_pred: np.array
        predicted values for y
    drug_to_threshold: dict of str->float
        The prediction threshold for each drug
    Returns
    -------
    pd.DataFrame with columns: 'Algorithm', 'Drug', "num_sensitive", "num_resistant",'AUC', "threshold", "spec", "sens"
    """
    column_names = ['Algorithm', 'Drug', "num_sensitive", "num_resistant",'AUC', "threshold", "spec", "sens"]
    results = []

    for idx, drug in enumerate(drugs):
        print(f"calculating test metrics for drug: {drug}")

        # Calculate the threshold from the TRAINING data, not the test data
        threshold = float(drug_to_threshold[drug])
        non_missing = np.where(y[:, idx] != -1)[0]
        
        # Check if non_missing_val is empty (no valid data for this drug) -> no phenotype
        if len(non_missing)==0:
            # If no valid data, insert NaN values for metrics
            print(f"No valid data for drug: {drug} as all the rows are missing")
            results.append([model_type, drug, 0, 0, np.nan, threshold, np.nan, np.nan])
            continue


        y_true = y[non_missing, idx]
        y_score = y_pred[non_missing, idx]

        num_sensitive = np.sum(y_true == 1)
        num_resistant = np.sum(y_true == 0)

        # If we don't have at least 1 R and 1 S isolate we can't assess model
        if num_sensitive==0 or num_resistant==0:
            results.loc[idx] = [model_type, drug, num_sensitive, num_resistant, np.nan, threshold, np.nan, np.nan]
            continue  

        # Compute the AUC
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
        binary_pred = (y_score > threshold).astype(int)

        # Be careful - RS encoding to numeric, resistant==0
        # Specificity = #TN / #Condition Negative,  # Sensitivity = #TP / #Condition Positive, Here defining "positive" as resistant
        spec = np.sum((binary_pred == 1) & (y_true == 1)) / num_sensitive if num_sensitive > 0 else np.nan
        sens = np.sum((binary_pred == 0) & (y_true == 0)) / num_resistant if num_resistant > 0 else np.nan

        results.append([model_type, drug, num_sensitive, num_resistant, auc, threshold, spec, sens])

    return pd.DataFrame(results, columns=column_names)


def conditionally_standardize_embeddings(embeddings, std_threshold=0.2):
    """
    Conditionally standardize embeddings only if their standard deviation is below a threshold.

    Args:
        embeddings (torch.Tensor): The input embeddings (samples, num_genes, hidden_dim).
        std_threshold (float): The threshold for standard deviation to trigger standardization.

    Returns:
        torch.Tensor: The standardized embeddings (if std is below threshold).
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


def train_on_token_embeddings(train_loader,
                val_loader,
                drug,
                num_sensitive,
                num_resistant,
                criterion,
                learning_rate,
                weight_decay,
                output_path,
                saved_model_path,
                model_name='DNABERTCNN',
                model_dim=768,
                model_seq_len=5000,
                k_folds=5,
                epochs=30,
                train_batch_size=64,
                val_batch_size=64,
                freeze_bias_frac=0.25,
                random_seed=1,
                device='cuda',
                early_stopping_min_epochs=5,
                early_stopping_patience=5,
                early_stopping_min_relative_improvement=1e-3,
                early_stopping_smoothing_window=3,
                use_training_loss_early_stopping=True,
                use_auc_early_stopping=False):
    """
    Train a model on token embeddings for a specific drug using k-fold cross-validation.
    
    Parameters
    ----------
    early_stopping_min_epochs : int
        Minimum epochs before early stopping can trigger (default: 5)
    early_stopping_patience : int
        Patience for early stopping (default: 5)
    early_stopping_min_relative_improvement : float
        Minimum relative improvement threshold (default: 1e-3)
    early_stopping_smoothing_window : int
        Window size for smoothing training loss (default: 3)
    use_training_loss_early_stopping : bool
        If True, use training-loss-based early stopping (default: True)
    """
    model = get_model_class(model_name=model_name, in_dim=model_dim, seq_len=model_seq_len, device=device)
    optimizer = get_optimizer(model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)

    # Set up TensorBoard logging
    summary_writer = SummaryWriter(log_dir=os.path.join(output_path, f"runs/{drug}/seed_{random_seed}"))

    # # Need more num workers due to high data dimensions
    # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # print("Len val data:", len(val_dataset))
    # val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # bias freeze init
    freeze_epoch = max(1,int(epochs*freeze_bias_frac))
    with torch.no_grad():
        res_prob = num_resistant / (num_resistant + num_sensitive + 1e-8)
        model.fc_out.bias.fill_(math.log(res_prob / (1 - res_prob)))
    model.fc_out.bias.requires_grad=False

    # Generate checkpoint path for early stopping
    seed_path = os.path.join(saved_model_path, f"{drug}/seed_{random_seed}")
    os.makedirs(seed_path, exist_ok=True)
    checkpoint_path = os.path.join(seed_path, f"{model_name}_best_model.pt")

    model, history = token_embed_train(
        model, train_loader, val_loader, optimizer, criterion, summary_writer, 
        freeze_epoch, epochs, device, checkpoint_path=checkpoint_path,
        early_stopping_min_epochs=early_stopping_min_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_relative_improvement=early_stopping_min_relative_improvement,
        early_stopping_smoothing_window=early_stopping_smoothing_window,
        use_training_loss_early_stopping=use_training_loss_early_stopping,
        use_auc_early_stopping=use_auc_early_stopping,
    )
    print(f"Training complete for drug: {drug}\n")
    
    # Ensure the saved model path exists
    os.makedirs(seed_path, exist_ok=True)

    # Inspect fc1
    print("fc1.weight shape:", model.fc1.weight.shape)
    print("fc1.bias shape:", model.fc1.bias.shape)

    # Save the model state dictionary
    save_path = os.path.join(seed_path, f"{model_name}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")

    # Save the training history as a CSV
    history_path = os.path.join(output_path, f"{drug}/seed_{random_seed}")
    os.makedirs(history_path, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(history_path, f"{model_name}_history.csv"), index=False)
    print(f"History saved to {history_path}")


def cross_val_train_on_token_embeddings(
        dataset,
        drug,
        num_sensitive,
        num_resistant,
        criterion,
        learning_rate,
        weight_decay,
        output_path,
        saved_model_path,
        model_name='DNABERTCNN',
        model_dim=768,
        model_seq_len=5000,
        k_folds=5,
        epochs=30,
        train_batch_size=64,
        val_batch_size=64,
        freeze_bias_frac=0.25,
        random_seed=42,
        fold=None,
        data_loader_workers=0,
        skip_completed=False,
        device='cuda',
        early_stopping_min_epochs=5,
        early_stopping_patience=5,
        early_stopping_min_relative_improvement=1e-3,
        early_stopping_smoothing_window=3,
        use_training_loss_early_stopping=True,
        use_auc_early_stopping=False,
):
    """
    Train a model on token embeddings for a specific drug using k-fold cross-validation.
    
    Parameters
    ----------
    dataset : Dataset
        Dataset that returns (embedding, label) pairs
    drug : str
        Drug name
    num_sensitive : int
        Number of sensitive samples
    num_resistant : int
        Number of resistant samples
    criterion : torch.nn.Module
        Loss function
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay
    output_path : str
        Path to save outputs
    saved_model_path : str
        Path to save models
    model_name : str
        Model name (default: 'DNABERTCNN')
    model_dim : int
        Model input dimension (default: 768)
    model_seq_len : int
        Model sequence length (default: 5000)
    k_folds : int
        Number of folds (default: 5)
    epochs : int
        Maximum number of epochs (default: 30)
    train_batch_size : int
        Training batch size (default: 64)
    val_batch_size : int
        Validation batch size (default: 64)
    freeze_bias_frac : float
        Fraction of epochs to freeze bias (default: 0.25)
    random_seed : int
        Random seed (default: 42)
    fold : int or None
        If specified, only train this fold (default: None trains all folds)
    data_loader_workers : int
        Number of data loader workers (default: 0)
    skip_completed : bool
        Skip completed folds (default: False)
    device : str
        Device to use (default: 'cuda')
    early_stopping_min_epochs : int
        Minimum epochs before early stopping can trigger (default: 5)
    early_stopping_patience : int
        Patience for early stopping (default: 5)
    early_stopping_min_relative_improvement : float
        Minimum relative improvement threshold (default: 1e-3)
    early_stopping_smoothing_window : int
        Window size for smoothing training loss (default: 3)
    use_training_loss_early_stopping : bool
        If True, use training-loss-based early stopping (like SD-CNN).
        If False, use validation-AUC-based early stopping (old behavior).
        (default: True)
    
    Returns
    -------
    None
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    all_histories = []
    fold_models = []

    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        fold_number = fold_index + 1
        if fold is not None and fold_number != fold:
            continue
        if fold_index < START_FOLD.get(drug, 0):
            print(f"Skipping fold {fold_number} for drug: {drug}")
            continue

        fold_path = os.path.join(saved_model_path, f"{drug}/seed_{random_seed}/fold_{fold_number}")
        save_path = os.path.join(fold_path, f"{model_name}.pt")
        hist_path = os.path.join(output_path, f"{drug}/seed_{random_seed}")
        history_file = os.path.join(hist_path, f"{model_name}_fold{fold_number}_history.csv")
        
        # Checkpoint path for early stopping
        os.makedirs(fold_path, exist_ok=True)
        checkpoint_path = os.path.join(fold_path, f"{model_name}_best_model.pt")
        
        if skip_completed and os.path.exists(save_path) and os.path.exists(history_file):
            print(f"Skipping completed fold {fold_number} for drug: {drug}")
            continue

        print(f"\n\n==== Fold {fold_number}/{k_folds} for drug: {drug} ====")

        # Subsets for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        loader_kwargs = {
            "num_workers": data_loader_workers,
            "pin_memory": True,
        }
        if data_loader_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        train_loader = DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            val_subset, batch_size=val_batch_size, shuffle=False, **loader_kwargs
        )

        # Fresh model + optimizer
        model = get_model_class(model_name=model_name, in_dim=model_dim, seq_len=model_seq_len, device=device)
        optimizer = get_optimizer(model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)

        # TensorBoard logging per fold
        log_dir = os.path.join(output_path, f"runs/{drug}/seed_{random_seed}/fold_{fold_number}")
        summary_writer = SummaryWriter(log_dir=log_dir)

        # Bias initialization
        freeze_epoch = max(1, int(epochs * freeze_bias_frac))
        with torch.no_grad():
            res_prob = num_resistant / (num_resistant + num_sensitive + 1e-8)
            model.fc_out.bias.fill_(math.log(res_prob / (1 - res_prob)))
        model.fc_out.bias.requires_grad = False

        # Train model on this fold
        model, history = token_embed_train(
            model, train_loader, val_loader,
            optimizer, criterion,
            summary_writer, freeze_epoch,
            epochs=epochs,
            device=device,
            checkpoint_path=checkpoint_path,
            early_stopping_min_epochs=early_stopping_min_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_relative_improvement=early_stopping_min_relative_improvement,
            early_stopping_smoothing_window=early_stopping_smoothing_window,
            use_training_loss_early_stopping=use_training_loss_early_stopping,
            use_auc_early_stopping=use_auc_early_stopping,
        )
        summary_writer.close()

        all_histories.append(pd.DataFrame(history))
        fold_models.append(model)

        # ---- Save model + history ----
        os.makedirs(fold_path, exist_ok=True)

        # Inspect fc1
        print("fc1.weight shape:", model.fc1.weight.shape)
        print("fc1.bias shape:", model.fc1.bias.shape)

        torch.save(model.state_dict(), save_path)
        print(f"Model for fold {fold_number} saved to {save_path}")

        os.makedirs(hist_path, exist_ok=True)
        all_histories[-1].to_csv(history_file, index=False)
        print(f"History for fold {fold_number} saved to {history_file}")

    # ---- Aggregate CV results ----
    print("\nCross-validation complete.")
    # return fold_models, all_histories
