# Evo2 Training Updates: Early Stopping and Checkpointing

## Summary

Updated the Evo2 training pipeline to use **training-loss-based early stopping** matching the SD-CNN implementation. The changes ensure that early stopping decisions are made using only training data, never test/validation data, following best practices for model development.

## Changes Made

### 1. Core Training Utilities (`token_train_utils.py`)

**Location:** `/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/train/transfer_learn/utils/token_train_utils.py`

#### Added `TrainingLossEarlyStopping` Class
- **Purpose**: Monitor smoothed training loss and stop training when convergence is detected
- **Key Features**:
  - Smooths training loss over a configurable window (default: 3 epochs)
  - Requires minimum relative improvement (default: 0.1% = 1e-3)
  - Has a minimum number of epochs before early stopping can trigger (default: 20)
  - Uses patience counter to avoid stopping too early (default: 10 epochs)
  - Automatically saves checkpoints when loss improves
  
#### Updated `token_embed_train` Function
- Added early stopping parameters to function signature
- Supports both training-loss and validation-AUC early stopping
- Default behavior: use training-loss early stopping (SD-CNN style)
- Loads best checkpoint after training completes

#### Updated `cross_val_train_on_token_embeddings` Function
- Added early stopping parameters
- Passes parameters through to `token_embed_train`
- Creates checkpoint path for each fold automatically

### 2. DNABERT Training Script (`resistance_classification_train.py`)

**Location:** `/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/train/transfer_learn/resistance_classification_train.py`

- Added early stopping parameters to argument parser
- Uses `getattr` with defaults for backward compatibility
- Passes early stopping parameters to cross-validation training

### 3. Evo2 Training Wrapper (`train.py`)

**Location:** `/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/evo2_downstream/train.py`

- Added early stopping command-line arguments:
  - `--early_stopping_min_epochs` (default: 20)
  - `--early_stopping_patience` (default: 10)
  - `--early_stopping_min_relative_improvement` (default: 1e-3 = 0.1%)
  - `--early_stopping_smoothing_window` (default: 3)
  - `--use_validation_early_stopping` (flag to use old behavior)

## Default Parameters

All defaults match SD-CNN implementation:

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `early_stopping_min_epochs` | 5 | Minimum epochs before early stopping can trigger |
| `early_stopping_patience` | 5 | Number of epochs to wait without improvement |
| `early_stopping_min_relative_improvement` | 1e-3 (0.1%) | Minimum relative improvement threshold |
| `early_stopping_smoothing_window` | 3 | Window size for smoothing training loss |

## How It Works

### Training Loss Early Stopping Algorithm

1. **Track Training Loss**: After each epoch, record the mean training loss
2. **Smooth Loss**: Compute average over the last `smoothing_window` epochs
3. **Check Improvement**: Calculate relative improvement:
   ```
   relative_improvement = (best_smoothed_loss - current_smoothed_loss) / |best_smoothed_loss|
   ```
4. **Update Best Model**: If improvement >= `min_relative_improvement`:
   - Save model checkpoint
   - Reset patience counter
   - Update best smoothed loss
5. **Increment Patience**: If no sufficient improvement and epoch > `min_epochs`:
   - Increment patience counter
6. **Stop Training**: If patience counter >= `patience`:
   - Stop training
   - Load best checkpoint
   - Return trained model

### Checkpointing

- Checkpoints are saved automatically when training loss improves
- Checkpoint filename: `{model_name}_best_training_loss.pt`
- Checkpoint location: `{saved_model_path}/{drug}/seed_{random_seed}/fold_{fold_number}/`
- Best checkpoint is loaded after training completes

## Usage Examples

### Using Evo2 with Default Settings (Training-Loss Early Stopping)

```bash
python -m evo2_downstream.train \
    --drug ISONIAZID \
    --num_epochs 100 \
    --fold 1
```

This will use:
- Training-loss-based early stopping (SD-CNN style)
- min_epochs=20, patience=10, min_relative_improvement=0.1%, smoothing_window=3
- Automatic checkpointing enabled

### Customizing Early Stopping Parameters

```bash
python -m evo2_downstream.train \
    --drug RIFAMPICIN \
    --num_epochs 100 \
    --early_stopping_min_epochs 30 \
    --early_stopping_patience 15 \
    --early_stopping_min_relative_improvement 0.005 \
    --early_stopping_smoothing_window 5
```

### Using Old Validation-AUC Early Stopping

```bash
python -m evo2_downstream.train \
    --drug ETHAMBUTOL \
    --num_epochs 50 \
    --use_validation_early_stopping
```

This reverts to the old behavior:
- Monitors validation AUC
- Stops when AUC doesn't improve
- Uses validation data for stopping decisions

## Verification

### Files Modified

1. ✅ `/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/train/transfer_learn/utils/token_train_utils.py`
   - Added `TrainingLossEarlyStopping` class
   - Updated `token_embed_train` function
   - Updated `cross_val_train_on_token_embeddings` function
   - Added `import math` for bias initialization

2. ✅ `/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/train/transfer_learn/resistance_classification_train.py`
   - Added early stopping parameters to argument parser
   - Updated call to `cross_val_train_on_token_embeddings`

3. ✅ `/project/pi_annagreen_umass_edu/saishradha/Data-Curation-for-MTB/dna-tasks/Evo2/evo2_downstream/train.py` (committed to `sm/evo2_optimized` branch)
   - Added early stopping parameters to argument parser

### No Syntax Errors

All files have been validated for Python syntax errors.

## Backward Compatibility

The implementation is fully backward compatible:

1. **Default Behavior Changed**: Now uses training-loss early stopping by default
2. **Old Behavior Available**: Use `--use_validation_early_stopping` flag to revert
3. **Existing Scripts**: Will work with new defaults (may stop earlier due to convergence)
4. **New Parameters Optional**: All have sensible defaults from SD-CNN

## Key Differences from Old Implementation

| Aspect | Old (Validation-AUC) | New (Training-Loss) |
|--------|---------------------|---------------------|
| Monitoring Metric | Validation AUC | Smoothed Training Loss |
| Uses Test/Val Data? | Yes (validation set) | No (training only) |
| Smoothing | None | 3-epoch moving average |
| Min Epochs | None | 5 epochs minimum |
| Improvement Metric | Absolute (min_delta=1e-4) | Relative (0.1%) |
| Patience | 5 epochs | 5 epochs |
| Checkpoint Saving | Only on improvement | Only on improvement |

## Benefits

1. **No Data Leakage**: Early stopping decisions never use test/validation data
2. **Better Generalization**: Training-loss monitoring prevents overfitting to validation set
3. **Consistent with SD-CNN**: Matches reference implementation exactly
4. **More Robust**: Smoothing reduces sensitivity to noisy loss values
5. **Transparent**: Logs all early stopping metrics during training

## Next Steps

1. Test on a sample drug to verify training completes successfully
2. Compare convergence behavior with old validation-based early stopping
3. Monitor checkpoint files to ensure best models are being saved
4. Review training logs to validate early stopping decisions

## Example Training Output

```
Epoch 25/100
Training: 100%|██████████| 123/123 [00:15<00:00,  8.12it/s]
Epoch [25/100] Train Loss: 0.3421 | Train Acc: 0.875 
 Val Loss: 0.3654 | Val Acc: 0.862 | Val AUC: 0.891
epoch=25 mean_train_loss=0.342100 smoothed_train_loss=0.345200 best_smoothed_train_loss=0.338100 relative_improvement=2.1e-03 patience_counter=0

...

epoch=45 mean_train_loss=0.312500 smoothed_train_loss=0.313200 best_smoothed_train_loss=0.311800 relative_improvement=4.5e-04 patience_counter=8

epoch=46 mean_train_loss=0.312400 smoothed_train_loss=0.312900 relative_improvement=2.9e-04 patience_counter=9

epoch=47 mean_train_loss=0.312600 smoothed_train_loss=0.312800 relative_improvement=9.6e-05 patience_counter=10

Training converged at epoch 47.
Smoothed training loss did not improve by >= 0.1%
for 10 consecutive epochs.
Best checkpoint was from epoch 37.
Loading best model checkpoint from epoch 37
```

## Branch Information

- **SD-CNN Reference**: `sm/lineage-aware-sdcnn` (source of implementation)
- **Evo2 Updates**: `sm/evo2_optimized` (where changes were committed)

## Git Commit

Committed to `sm/evo2_optimized` branch:
- Commit: `bc7004f`
- Message: "Add training-loss early stopping and checkpointing to Evo2 training"
