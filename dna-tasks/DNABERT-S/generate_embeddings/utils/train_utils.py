import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
from dataloader.locus_order import DRUGS as drugs
from downstream_models import *

def get_model_class(model_name, device='cuda'):
    """
    Get the model class based on the model name.
    """
    if model_name == 'MDMLP':
        return MDMLP(dropout_rate=0).to(device)
    elif model_name == 'MDCNN':
        return MDCNN(num_classes=11, dropout_rate=0).to(device)
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


def evaluate(model, val_loader, device='cuda'):
    model.eval()
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for batch_emb, batch_labels in val_loader:
            inputs = batch_emb.to(device)
            outputs = model(inputs).cpu().numpy()
            all_outputs.append(outputs)
            all_targets.append(batch_labels.numpy())

    y_val = np.vstack(all_targets)
    y_pred = np.vstack(all_outputs)
    return y_val, y_pred


def train_kfold_mod(dataset,
                drugs,
                criterion,
                learning_rate,
                weight_decay,
                acc_metric,
                get_threshold_val,
                output_path,
                saved_model_path,
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
        model = get_model_class('MDCNN', device=device)
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
        save_path = os.path.join(seed_path, f"dnabert-mdcnn_cv_split_{fold}.pt")
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


def compute_val_metrics_per_drug(y_val, y_pred, drugs, fold, get_threshold_val):

    results = []
    for idx, drug in enumerate(drugs):
        non_missing = np.where(y_val[:, idx] != -1)[0]
        if len(non_missing) == 0:
            print(f"[{drug}] Skipped (no data)")
            results.append([f"val_split{fold}", 'DNABERT-MDCNN', drug, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        y_true = y_val[non_missing, idx]
        y_score = y_pred[non_missing, idx]
        num_sensitive = np.sum(y_true == 1)
        num_resistant = np.sum(y_true == 0)

        if num_sensitive == 0 or num_resistant == 0:
            results.append([f"val_split{fold}", 'DNABERT-MDCNN', drug, num_sensitive, num_resistant, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        auc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(1 - y_true, 1 - y_score)
        thresh = get_threshold_val(y_true, y_score)
        results.append([f"val_split{fold}", 'DNABERT-MDCNN', drug, num_sensitive, num_resistant, auc, auc_pr, thresh["threshold"], thresh["spec"], thresh["sens"]])

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


def calculate_test_auc(y, y_pred, drug_to_threshold):
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
            results.append(['MD-CNN', drug, 0, 0, np.nan, threshold, np.nan, np.nan])
            continue


        y_true = y[non_missing, idx]
        y_score = y_pred[non_missing, idx]

        num_sensitive = np.sum(y_true == 1)
        num_resistant = np.sum(y_true == 0)

        # If we don't have at least 1 R and 1 S isolate we can't assess model
        if num_sensitive==0 or num_resistant==0:
            results.loc[idx] = ['MD-CNN', drug, num_sensitive, num_resistant, np.nan, threshold, np.nan, np.nan]
            continue  

        # Compute the AUC
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
        binary_pred = (y_score > threshold).astype(int)

        # Be careful - RS encoding to numeric, resistant==0
        # Specificity = #TN / #Condition Negative,  # Sensitivity = #TP / #Condition Positive, Here defining "positive" as resistant
        spec = np.sum((binary_pred == 1) & (y_true == 1)) / num_sensitive if num_sensitive > 0 else np.nan
        sens = np.sum((binary_pred == 0) & (y_true == 0)) / num_resistant if num_resistant > 0 else np.nan

        results.append(['MD-CNN', drug, num_sensitive, num_resistant, auc, threshold, spec, sens])

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