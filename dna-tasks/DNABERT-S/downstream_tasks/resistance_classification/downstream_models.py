import torch
import torch.nn as nn


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
