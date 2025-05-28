import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiWeightedBCE(nn.Module):
    def __init__(self):
        super(MaskedMultiWeightedBCE, self).__init__()
        self.eps = 1e-06

    def forward(self, alpha, y_pred):
        """
        Calculates the masked weighted binary cross-entropy in multi-classification

        Parameters
        ----------
        alpha: Tensor
            A tensor of target y values weighted by the proportion of strains with resistance data
            for any given drug.
        y_pred: Tensor
            Model-predicted y values.

        Returns
        -------
        scalar value of the masked weighted BCE.
        """

        batch_size = alpha.size(0)
    
        # Ensure predictions are within the range (0, 1)
        y_pred = torch.clamp(y_pred, min=self.eps, max=1.0 - self.eps)
        
        # Create masks based on the value of `alpha`
        y_true_ = torch.gt(alpha, 0).float()  # Equivalent of K.cast(K.greater(alpha, 0.), K.floatx())
        mask = torch.ne(alpha, 0).float()     # Equivalent of K.cast(K.not_equal(alpha, 0.), K.floatx())
        
        # Count the number of non-missing values
        num_not_missing = torch.sum(mask, dim=-1)

        # if torch.any(num_not_missing == 0):
        #     print("At least one drug has no resistance data.")
        
        # Take the absolute value of `alpha`
        alpha = torch.abs(alpha)
        
        # Compute the binary cross-entropy
        bce = - alpha * y_true_ * torch.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * torch.log(1.0 - y_pred)

        # Apply the mask
        masked_bce = bce * mask     # Shape: (batch_size, num_drugs)

        # Return the mean masked BCE across the batch
        mean_bce_loss = torch.sum(masked_bce, dim=-1) / (num_not_missing + self.eps)

        # print("Alpha min/max:", alpha.min().item(), alpha.max().item())
        # print("y_pred min/max:", y_pred.min().item(), y_pred.max().item())

        return mean_bce_loss


class MaskedWeightedAccuracy(nn.Module):
    def __init__(self):
        super(MaskedWeightedAccuracy, self).__init__()
        pass

    def forward(self, alpha, y_pred):
        """
        Calculates the masked weighted accuracy of a model's predictions.

        Parameters
        ----------
        alpha: Tensor
            An element from the alpha matrix, a matrix of target y values weighted
            by the proportion of strains with resistance data for any given drug.
        y_pred: Tensor
            Model-predicted y values.

        Returns
        -------
        Scalar value of the masked weighted accuracy.
        """
        # Create a mask to identify where `alpha` is not equal to 0
        mask = torch.ne(alpha, 0).float()
        
        # Calculate the total number of non-zero elements in `alpha`
        total = torch.sum(mask)
        
        # Convert `alpha` to binary values (1 where `alpha > 0`, 0 where `alpha <= 0`)
        y_true_ = torch.gt(alpha, 0).float()
        
        # Calculate the number of correct predictions (where predicted values match the true values)
        correct = torch.sum((y_true_ == torch.round(y_pred)).float() * mask)
        
        # Return the masked weighted accuracy
        masked_weighted_acc = correct / total if total > 0 else torch.tensor(0.0, device=alpha.device)

        return masked_weighted_acc
    
    
class ThresholdValue(nn.Module):
    def __init__(self):
        super(ThresholdValue, self).__init__()

    def forward(self, y_true, y_pred):
        """
        Compute optimal threshold based on max sensitivity + specificity.
        Assumes y_true ∈ {0 (resistant), 1 (sensitive)}.
        Smaller y_pred means more resistant.
        """
        # Ensure inputs are torch tensors on CPU
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)

        y_true = y_true.cpu().flatten()
        y_pred = y_pred.cpu().flatten()

        thresholds = torch.linspace(0, 1, 101)
        num_sensitive = (y_true == 1).sum().item()
        num_resistant = (y_true == 0).sum().item()

        fpr_list = []
        tpr_list = []

        for threshold in thresholds:
            predicted_resistant = y_pred < threshold
            fp = ((predicted_resistant == 1) & (y_true == 1)).sum().item()
            tp = ((predicted_resistant == 1) & (y_true == 0)).sum().item()

            fpr = fp / num_sensitive if num_sensitive > 0 else 0
            tpr = tp / num_resistant if num_resistant > 0 else 0

            fpr_list.append(fpr)
            tpr_list.append(tpr)

        fpr_tensor = torch.tensor(fpr_list)
        tpr_tensor = torch.tensor(tpr_list)

        sens_spec_sum = (1 - fpr_tensor) + tpr_tensor
        best_ind = torch.argmax(sens_spec_sum).item()

        return {
            'threshold': thresholds[best_ind].item(),
            'spec': (1 - fpr_tensor[best_ind]).item(),
            'sens': tpr_tensor[best_ind].item()
        }
