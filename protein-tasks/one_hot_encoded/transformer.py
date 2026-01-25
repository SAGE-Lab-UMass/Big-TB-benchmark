import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import shap 
from functools import reduce
import random
from sklearn.model_selection import train_test_split


# Mapping for 20 standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

        lengths = [len(seq) for seq in sequences]
        min_len = min(lengths)
        max_len = max(lengths)

        if max_len - min_len > 2:
            raise ValueError(f"Sequences vary too much in length! Found lengths: {set(lengths)}")

        self.seq_len = max_len  # allow minor difference (pad shorter sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        onehot = np.zeros((20, self.seq_len), dtype=np.float32)
        for i, aa in enumerate(seq):
            if i >= self.seq_len:  # (safety) but should never happen
                break
            if aa in AA_TO_INDEX:
                onehot[AA_TO_INDEX[aa], i] = 1.0

        return torch.tensor(onehot), torch.tensor(label, dtype=torch.float32)


class ProteinTransformer(nn.Module):
    def __init__(self, input_dim=20, d_model=128, nhead=4, num_layers=2,
                 max_len=6000):                      # upper bound
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # pre-compute positional encoding once
        pe = self._make_pe(d_model, max_len)
        self.register_buffer("pos_enc", pe)          # NOT a parameter

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.5),                        # small reg
            nn.Linear(d_model, 1)                  # logits
        )

    @staticmethod
    def _make_pe(d_model, length, device="cpu"):
        pos = torch.arange(length).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(1e4)/d_model))
        pe  = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe                                  # (L,d_model)

    def forward(self, x):                          # x (B,C,L)
        x = x.permute(0, 2, 1)                     # (B,L,C)
        x = self.embedding(x)                      # (B,L,d_model)
        L = x.size(1)
        x = x + self.pos_enc[:L, :].unsqueeze(0)   # add PE
        x = self.encoder(x)                        # (B,L,d_model)
        x = x.permute(0, 2, 1)                     # (B,d_model,L)
        return self.classifier(x).squeeze(-1)      # logits
