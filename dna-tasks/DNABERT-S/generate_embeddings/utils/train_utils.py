import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
from dataloader.locus_order import DRUGS as drugs

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
            
            # print(f"Drug {drug} has {resistant_num} R; {sensitive_num} S; {unknown_num} unknown strains")

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


def train_kfold(model, 
                dataset, 
                optimizer, 
                criterion, 
                acc_metric, 
                k_folds=5, 
                epochs=40, 
                train_batch_size=64, 
                val_batch_size=64,
                device='cuda'):
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        # Prepare data loaders for this fold
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=val_batch_size, shuffle=False)

        # Training
        model.train()
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

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"[Fold {fold+1}] Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {accuracy.item():.4f}")

        # Evaluation
        model.eval()
        val_accuracies = []
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                inputs = batch_emb.to(device)
                targets = batch_labels.to(device)
                alphas = calculate_alphas(targets).to(device)

                outputs = model(inputs)
                acc = acc_metric(alphas, outputs)
                val_accuracies.append(acc.item())

        val_acc_mean = sum(val_accuracies) / len(val_accuracies)
        fold_metrics.append(val_acc_mean)
        print(f"Fold {fold+1} Validation Accuracy: {val_acc_mean:.4f}")

    print(f"\nAverage Validation Accuracy across folds: {sum(fold_metrics) / k_folds:.4f}")
    return model, fold_metrics


def train_kfold_mod(model,
                dataset,
                drugs,
                optimizer,
                criterion,
                acc_metric,
                get_threshold_val,
                output_path,
                saved_model_path,
                k_folds=5,
                epochs=40,
                train_batch_size=64,
                val_batch_size=64,
                device='cuda'):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_df = pd.DataFrame(columns=['Validation Split #', 'Algorithm', 'Drug', "num_sensitive", "num_resistant", 'AUC', 'AUC_PR', "Threshold", "Spec", "Sens"])
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Fold {fold+1}/{k_folds} ===")

        # Set up TensorBoard logging
        summary_writer = SummaryWriter(log_dir=os.path.join(output_path, f"runs/fold_{fold}"))

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=val_batch_size, shuffle=False)

        # Train
        model, history = train(model, train_loader, optimizer, criterion, acc_metric, summary_writer, epochs, device)

    
        # Save model + history
        torch.save(model.state_dict(), f"{saved_model_path}_cv_split_{fold}.pt")
        pd.DataFrame(history).to_csv(os.path.join(output_path, f"history_cv_split{fold}.csv"), index=False)

        # Evaluate
        y_val, y_pred = evaluate(model, val_loader, device)
        fold_metrics = compute_metrics_per_drug(y_val, y_pred, drugs, fold, get_threshold_val)

        # Log
        for row in fold_metrics:
            results_df.loc[len(results_df)] = row
        results_df.to_csv(os.path.join(output_path, f"cv_split_{fold}_auc.csv"), index=False)

    # Final save
    results_df.to_csv(os.path.join(output_path, "auc.csv"), index=False)
    print(f"\n✓ K-Fold CV Complete. Results saved to {output_path}")
    return results_df


def compute_metrics_per_drug(y_val, y_pred, drugs, fold, get_threshold_val):

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
        results.append([f"val_split{fold}", 'CNN', drug, num_sensitive, num_resistant, auc, auc_pr, thresh["threshold"], thresh["spec"], thresh["sens"]])

    return results