import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import tqdm
import os

from utils.data_utils import prepare_multigene_input_fast
from dataloader.locus_order import DRUGS as drugs

def train_dnabert_classifier(model, train_loader, val_loader, tokenizer, optimizer, criterion, scheduler, device, args, resume_checkpoint=None):
    scaler = GradScaler()

    # Resume training if a checkpoint is provided
    start_epoch = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            input_ids, attention_mask, labels = prepare_multigene_input_fast(batch, tokenizer, args.max_length)
            input_ids = torch.stack([ids.to(device, non_blocking=True) for ids in input_ids], dim=1)
            attention_mask = torch.stack([mask.to(device, non_blocking=True) for mask in attention_mask], dim=1)
            labels = labels.to(device).float()

            with autocast(device_type="cuda"):
                logits = model(input_ids, attention_mask)
                alphas = calculate_alphas(labels)
                per_sample_loss = criterion(alphas, logits)
                loss = torch.mean(per_sample_loss) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum_steps

        print(f"Epoch {epoch} Training Loss: {running_loss / len(train_loader):.4f}")

        # Scheduler step
        scheduler.step()

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, args.output_dir)

        # Validation
        if (epoch + 1) % args.val_every == 0:
            validate(model, val_loader, tokenizer, criterion, device, args)

        save_dnabert_weights_only(model, args.output_dir)


def save_checkpoint(model, optimizer, scheduler, epoch, output_dir):
    save_dir = os.path.join(output_dir, "saved_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def save_dnabert_weights_only(model, output_dir):
    # Ensure the saved_models directory exists
    save_dir = os.path.join(output_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model state dict
    dnabert_weights = model.base_model.state_dict()
    save_path = os.path.join(save_dir, "dnabert_only_finetuned.pth")
    torch.save(dnabert_weights, save_path)
    print(f"Model saved at {save_path}")


def validate(model, val_loader, tokenizer, criterion, device, args):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = prepare_multigene_input_fast(batch, tokenizer, args.max_length)
            input_ids = torch.stack([ids.to(device, non_blocking=True) for ids in input_ids], dim=1)
            attention_mask = torch.stack([mask.to(device, non_blocking=True) for mask in attention_mask], dim=1)
            labels = labels.to(device).float()

            with autocast(device_type="cuda"):
                logits = model(input_ids, attention_mask)
                alphas = calculate_alphas(labels).to(device)
                per_sample_loss = criterion(alphas, logits)
                loss = torch.mean(per_sample_loss)

            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(val_loader):.4f}")
    return total_loss / len(val_loader)


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