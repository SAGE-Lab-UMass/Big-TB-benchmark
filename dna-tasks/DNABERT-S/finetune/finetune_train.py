import torch
from torch import nn
from torch.amp import autocast, GradScaler
import tqdm
import os

from utils.data_utils import prepare_multigene_input_fast

def train_with_acc_gradients(model, train_loader, val_loader, tokenizer, optimizer, scheduler, criterion, args):
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        grad_accum_steps = args.grad_accum_steps  # New argument you define outside (effective_batch_size // actual_batch_size)
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            input_ids, attention_mask, labels = prepare_multigene_input_fast(batch, tokenizer, args.max_length)

            logits_all_genes = []
            for i in range(len(input_ids)):
                logits = model(input_ids=input_ids[i].cuda(), attention_mask=attention_mask[i].cuda())
                logits_all_genes.append(logits)

            logits_avg = torch.stack(logits_all_genes, dim=0).mean(0)
            labels = labels.cuda().float()

            # Compute loss
            loss = criterion(logits_avg, labels)
            loss = loss / grad_accum_steps  # Normalize loss

            loss.backward()

            # Update only every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps  # Rescale back to original loss per batch

        print(f"Epoch {epoch} Training Loss: {running_loss / len(train_loader):.4f}")

        if (epoch + 1) % args.val_every == 0:
            evaluate(model, val_loader, tokenizer, args)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "dnabert_finetuned.pth"))
    print("Model saved!")



def train_no_acc_gradients(model, train_loader, val_loader, tokenizer, optimizer, scheduler, criterion, args):
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0


        for batch in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            input_ids, attention_mask, labels = prepare_multigene_input_fast(batch, tokenizer, args.max_length)

            logits_all_genes = []
            for i in range(len(input_ids)):
                logits = model(input_ids[i].cuda(), attention_mask[i].cuda())
                logits_all_genes.append(logits)

            print("logits_all_genes shape: ", logits_all_genes[0].shape)
            print("labels shape: ", labels.shape)

            logits_avg = torch.stack(logits_all_genes, dim=0).mean(0)
            print("logits_avg shape: ", logits_avg.shape)
            labels = labels.cuda().float()

            loss = criterion(logits_avg, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        print(f"Epoch {epoch} Training Loss: {running_loss / len(train_loader):.4f}")

        if (epoch + 1) % args.val_every == 0:
            evaluate(model, val_loader, tokenizer, args)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "dnabert_finetuned.pth"))
    print("Model saved!")



def evaluate(model, val_loader, tokenizer, args):
    device = next(model.parameters()).device
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    running_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Validating"):
            input_ids, attention_mask, labels = prepare_multigene_input_fast(batch, tokenizer, args.max_length)

            logits_all_genes = []
            for i in range(len(input_ids)):
                logits = model(input_ids[i].to(device), attention_mask[i].to(device))
                logits_all_genes.append(logits)

            logits_avg = torch.stack(logits_all_genes, dim=0).mean(0)
            labels = labels.to(device).float()

            loss = criterion(logits_avg, labels)
            running_val_loss += loss.item()

    print(f"Validation Loss: {running_val_loss / len(val_loader):.4f}")


def train_with_acc_gradients_mixed_precision(model, train_loader, val_loader, tokenizer, optimizer, scheduler, criterion, args):
    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        grad_accum_steps = args.grad_accum_steps
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            input_ids, attention_mask, labels = prepare_multigene_input_fast(batch, tokenizer, args.max_length)

            logits_all_genes = []
            for i in range(len(input_ids)):
                with autocast(device_type="cuda"):  # Mixed precision context
                    logits = model(input_ids=input_ids[i].cuda(), attention_mask=attention_mask[i].cuda())
                logits_all_genes.append(logits)
            
            logits_avg = torch.stack(logits_all_genes, dim=0).mean(0)
            labels = labels.cuda().float()

            # Compute loss
            with autocast(device_type="cuda"):
                loss = criterion(logits_avg, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()  # Scaled gradient for mixed precision

            # Gradient accumulation
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps

        print(f"Epoch {epoch} Training Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        if (epoch + 1) % args.val_every == 0:
            evaluate(model, val_loader, tokenizer, args)

    # Save only DNABERT weights
    save_dnabert_weights_only(model, args.output_dir)

def save_dnabert_weights_only(model, output_dir):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract only DNABERT weights
    dnabert_weights = model.base_model.state_dict()
    
    # Save only DNABERT weights
    torch.save(dnabert_weights, os.path.join(output_dir, "dnabert_only_finetuned.pth"))
    print("DNABERT weights saved (without FC layer)!")