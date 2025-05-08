import torch
import torch.nn as nn

class DNABERTClassifier(nn.Module):
    def __init__(self, base_model, hidden_dim=768, num_drugs=11):
        super(DNABERTClassifier, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, num_drugs)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        masked_output = sequence_output * attention_mask_expanded
        mean_pooled = masked_output.sum(1) / attention_mask_expanded.sum(1)

        mean_pooled = self.dropout(mean_pooled)
        logits = self.fc(mean_pooled)
        return logits
    