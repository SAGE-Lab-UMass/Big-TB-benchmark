import torch
import torch.nn as nn

    
class MDCNN(nn.Module):
    def __init__(self, num_classes=11, dropout_rate=0.1):
        super(MDCNN, self).__init__()
        
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
    
    
class SmallTransformer(nn.Module):
    def __init__(self, input_dim=768, seq_len=12, num_heads=4, hidden_dim=256, num_layers=2, num_classes=11):
        super(SmallTransformer, self).__init__()

        # Embedding Layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        x = self.transformer(x)  # (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)        # Global average pooling over sequence length
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    
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
    
    
class BigMDMLP(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(BigMDMLP, self).__init__()

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
