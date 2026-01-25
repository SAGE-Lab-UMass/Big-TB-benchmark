import os
import numpy as np

def set_parameters():
    parameters = {"C": [0.0001, 0.001, 0.01, 0.1, 1.]}
    return parameters

def get_threshold_val(y_true, y_pred):
    """
    Compute the optimal threshold for prediction  based on the max sum of specificity and Sensitivity

    NB that we encoded R as 0, S as 1, so smaller predictions indicate higher chance of resistance

    We count falsely predicted resistance as a false positive, falsely predicted sensitivity as a false negative

    Parameters
    ----------
    y_true: np.array
        Actual labels for isolates
    y_pred: np.array
        Predicted labels for isolates

    Returns
    -------
    dict of str -> float with entries:
        sens: sensitivity at chosen threshold
        spec: specificity at chosen threshold
        threshold: chosen threshold value
    """

    # Compute number resistant and sensitive
    num_samples = y_pred.shape[0]
    num_sensitive = np.sum(y_true==1)
    num_resistant = np.sum(y_true==0)

    # Test thresholds from 0.01 to 0.99
    thresholds = np.linspace(0, 1, 101)

    fpr_ = []
    tpr_ = []

    for threshold in thresholds:

        fp_ = 0 # number of false positives
        tp_ = 0 # number of true positives

        for i in range(num_samples):
            # If y is predicted resistant
            if (y_pred[i] < threshold):
                # If actually sensitive, false positive
                if (y_true[i] == 1): fp_ += 1
                # If actually resistant, true positive
                if (y_true[i] == 0): tp_ += 1

        # Compute FPR and TPR
        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    valid_inds = np.arange(101)
    # Sensitivity = TPR, Specificity = 1-FPR
    sens_spec_sum = (1 - fpr_) + tpr_

    # get index of highest sum(s) of sens and spec
    best_sens_spec_sum = np.max(sens_spec_sum[valid_inds])
    best_inds = np.where(best_sens_spec_sum == sens_spec_sum[valid_inds])

    # Determine if one or multiple best
    if best_inds[0].shape[0] == 1:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))
    else:
        # If multiple best, take the last one (arbitrary)
        best_sens_spec_ind = np.array(np.squeeze(best_inds))[-1]

    return {'threshold': np.squeeze(thresholds[valid_inds][best_sens_spec_ind]),
            'spec': 1 - fpr_[valid_inds][best_sens_spec_ind],
            'sens': tpr_[valid_inds][best_sens_spec_ind]}

def create_output_dir(output_dir, drug):
    """
    Create output directory if it does not exist

    Parameters
    ----------
    output_dir: str
        Path to output directory

    drug: str
        Name of the drug to create a subdirectory for

    Returns
    -------
    str
        Path to the created output directory
    """
    # Create the base output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the drug-specific subdirectory
    output_path = os.path.join(output_dir, drug)
    saved_models_path = os.path.join(output_path, "saved_models")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(saved_models_path, exist_ok=True)
    
    return output_path, saved_models_path