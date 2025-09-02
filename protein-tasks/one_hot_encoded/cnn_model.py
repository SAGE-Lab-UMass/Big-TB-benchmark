# cnn_model.py
import torch, torch.nn as nn, torch.nn.functional as F

class ProteinCNN1x1(nn.Module):
    def __init__(self, seq_len: int, in_dim: int = 20, stem_out: int = 64):
        super().__init__()
        self.stem  = nn.Conv1d(in_dim, stem_out, 1)
        self.conv1 = nn.Conv1d(stem_out, 64, 12, padding=6)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(3)
        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, seq_len)
            flat  = self._forward_feat(dummy).numel()
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

    def _forward_feat(self, x):
        x = self.stem(x)
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x)); x = self.pool2(x)
        return x

    def forward(self, x):            # x: (B,C,L)
        x = self._forward_feat(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)
