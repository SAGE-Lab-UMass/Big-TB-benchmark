import torch
import torch.nn as nn, torch.nn.functional as F, math


class MDCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MDCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # 12 -> 6

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)  # 6 -> 3

        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 3, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.relu6 = nn.ReLU()

        self.fc3 = nn.Linear(256, 11)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
    

class MDMLP(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MDMLP, self).__init__()

        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.LazyLinear(512)  # learns in_features = 9216
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 11)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Where to save the best model.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MeanDNABERTCNN(nn.Module):
    def __init__(self, seq_len, in_dim=1, num_classes=1, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # (B, 32, 1)

        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, seq_len if seq_len is not None else 512)
            flat = self._forward_feat(dummy).numel()

        self.fc1 = nn.Linear(flat, 16)
        self.fc_out = nn.Linear(16, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def _forward_feat(self, x):
        x = self.conv1(x)           # (B, 1, sum_L) → (B, 32, sum_L)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.global_avg_pool(x)  # → (B, 32, 1)
        return x

    def forward(self, x):
        # Input: (B, 1, sum(L))
        x = self._forward_feat(x)
        x = torch.flatten(x, 1)  # → (B, 32)
        x = self.dropout(F.relu(self.fc1(x)))  # → (B, 16)
        x = self.fc_out(x)                     # → (B, num_classes)
        return x.squeeze(-1) if self.num_classes == 1 else x
    
    
class DNABERTCNN(nn.Module):
    def __init__(self, seq_len, in_dim=768, stem_out=64):
        super().__init__()
        self.inp_project = nn.Conv1d(in_dim, stem_out, 1)

        self.conv1 = nn.Conv1d(stem_out, 64, kernel_size=12, padding=6)
        self.pool1 = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(1)

        # flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, seq_len)
            flat = self._forward_feat(dummy).numel()
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

    def _forward_feat(self, x):
        x = self.inp_project(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        return x

    def forward(self, x):
        x = self._forward_feat(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)   # logits
    
    
class PooledDNABERTCNN(nn.Module):
    def __init__(self, in_dim=768, stem_out=64):
        super().__init__()
        self.inp_project = nn.Conv1d(in_dim, stem_out, kernel_size=1)

        self.conv1 = nn.Conv1d(stem_out, 64, kernel_size=12, padding=6)
        self.pool1 = nn.MaxPool1d(3)

        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(3)

        # After convs, we will apply global average pooling: (B, 32, L') → (B, 32)
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

    def _forward_feat(self, x):
        x = self.inp_project(x)         # (B, stem_out, L)
        x = F.relu(self.conv1(x))       # (B, 64, L)
        x = self.pool1(x)               # (B, 64, L/3)
        x = F.relu(self.conv2(x))       # (B, 32, L/3)
        x = F.relu(self.conv3(x))       # (B, 32, L/3)
        x = self.pool2(x)               # (B, 32, L/9)
        return x

    def forward(self, x):
        x = self._forward_feat(x)       # (B, 32, L')
        x = x.mean(dim=-1)              # Global average pool → (B, 32)
        x = F.relu(self.fc1(x))         # (B, 256)
        x = F.relu(self.fc2(x))         # (B, 256)
        return self.fc_out(x).squeeze(-1)  # (B,)
