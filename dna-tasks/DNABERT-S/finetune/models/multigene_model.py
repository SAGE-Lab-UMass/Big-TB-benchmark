import torch
import torch.nn as nn
from torch.amp import autocast
import tqdm
import os

class DNABERTClassifier(nn.Module):
    def __init__(self, base_model, hidden_dim=768, num_drugs=11, num_genes=12):
        super(DNABERTClassifier, self).__init__()
        self.base_model = base_model
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim * num_genes, num_drugs)  # Adjusted for concatenated genes
        self.sigmoid = nn.Sigmoid()  # Sigmoid for multi-label classification


    def forward(self, input_ids, attention_mask):
        # input_ids and attention_mask should be (batch_size, num_genes, seq_len)
        batch_size, num_genes, seq_len = input_ids.size()

        gene_embeddings = []

        for i in range(num_genes):
            with autocast(device_type="cuda"):
                outputs = self.base_model(
                    input_ids=input_ids[:, i, :], 
                    attention_mask=attention_mask[:, i, :]
                )
                sequence_output = outputs[0]  # (batch_size, seq_len, hidden_dim)

                # Mean-pooling (directly in GPU for memory efficiency)
                attention_mask_expanded = attention_mask[:, i, :].unsqueeze(-1)
                masked_output = sequence_output * attention_mask_expanded
                mean_pooled = masked_output.sum(1) / attention_mask_expanded.sum(1)

                gene_embeddings.append(mean_pooled)

        # Concatenate all gene embeddings (batch_size, num_genes * hidden_dim) directly
        concatenated_embedding = torch.cat(gene_embeddings, dim=1)

        # Dropout and Classification
        concatenated_embedding = self.dropout(concatenated_embedding)
        logits = self.fc(concatenated_embedding)
        logits = self.sigmoid(logits)  # Apply sigmoid for multi-label classification

        return logits