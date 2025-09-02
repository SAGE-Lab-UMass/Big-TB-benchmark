# ## Step 4: CNN Model


# ──────────────────────────────────────────────────────────────
# 2.  model inspired by MDCNN: I have used conv1D since we are not stacking multi gene embeddings. we are concatenating
# ──────────────────────────────────────────────────────────────
import torch.nn as nn, torch.nn.functional as F, math
import torch

# ──────────────────────────────────────────────────────────────
# ProteinCNN1x1 
# • Input :  (batch , C=in_dim , L_total)
#            C = 320   for raw tokens
#                =  K  for PCA-compressed channels
# • We CONCAT different genes *along the length axis*,
#   so we keep a **single 1-D convolutional stack**.
# ──────────────────────────────────────────────────────────────

class ProteinCNN1x1(nn.Module):
    def __init__(self, seq_len, in_dim=320, stem_out=64):
        """
        seq_len :  total padded length after gene-concat
        in_dim  :  #channels (320 or PCA-K)
        stem_out:  how many channels right after the 1×1 stem
        """
        super().__init__()
        # stem 1×1 conv – just re-mix the 320 (or K) channels
        self.stem = nn.Conv1d(in_dim, stem_out, 1)
        
        # ───── shallow CNN stack ───── 
        self.conv1 = nn.Conv1d(stem_out, 64, kernel_size=12, padding=6) # big receptive field
        self.pool1 = nn.MaxPool1d(3)                                    # shrink length ×3
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(3)                                    # shrink again

        # flatten dim - figure out flatten size (once)
        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, seq_len) # fake batch=1
            flat = self._forward_feat(dummy).numel() # total features

        # ───── dense head ─────
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)       # final logit

    # ----------------------------------------------------------
    # internal helper: CNN part only
    # ----------------------------------------------------------
    
    def _forward_feat(self, x):
        x = self.stem(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        return x
    # ----------------------------------------------------------
    # full forward pass
    # ----------------------------------------------------------
    def forward(self, x):
        """
        x shape  :  (batch , C=in_dim , L_total)
        returns  :  raw logits  (batch,)   – use torch.sigmoid later
        """
        x = self._forward_feat(x)  # CNN feature map
        x = torch.flatten(x, 1)   # keep batch dim, flatten rest
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)   # (batch,)   logits

