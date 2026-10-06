import torch
import torch.nn as nn
import torch.nn as nn, torch.nn.functional as F, math

    
class MDDNABERTCNN(nn.Module):
    def __init__(self, num_classes=11, dropout_rate=0.1):
        super(MDDNABERTCNN, self).__init__()
        
        # Efficient Convolutional Layer
        self.conv1 = nn.Conv1d(768, 32, kernel_size=3, padding=1)  # Reduced channels (32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # (12 -> 6)

        # Global Average Pooling for efficient feature reduction
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # (batch, 32, 6) -> (batch, 32, 1)
        self.flatten = nn.Flatten()  # (batch, 32, 1) -> (batch, 32)
        
        # Compact MLP for classification
        self.fc1 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Global average pooling and flatten
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        
        # Compact MLP Block
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu_fc(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    
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

    
class MDMLP(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(MDMLP, self).__init__()

        # Global Average Pooling to reduce (768, 12) -> (768)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # Reduced MLP with fewer parameters
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 11)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling
        x = self.global_avg_pool(x)  # (batch, 768, 12) -> (batch, 768, 1)
        x = self.flatten(x)          # (batch, 768, 1) -> (batch, 768)

        # MLP
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
    

class DNABERTMLP(nn.Module):
    """
    Simple fully connected model compatible with DNABERTCNN input layout.
    Expects (B, in_dim, seq_len) and performs global pooling before MLP.
    """
    def __init__(self, seq_len, in_dim=768, p_drop=0.2):
        super().__init__()

        # compute flattened dimension after pooling for safety
        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, seq_len)
            flat_dim = dummy.mean(dim=-1).numel()  # (B, in_dim)

        # MLP layers
        self.fc1 = nn.Linear(flat_dim, 256)
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p_drop)
        self.fc_out = nn.Linear(256, 1)

    def _forward_feat(self, x):
        # Global average pool over sequence length
        x = x.mean(dim=-1)             # (B, in_dim)
        x = F.relu(self.fc1(x))        # (B, hidden_dim)
        x = self.norm(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_feat(x)
        return self.fc_out(x).squeeze(-1)  # (B,)
    

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
