# Task 2 Report: Custom Neural Network on UCI HAR

## Objective
Task 2 requires building a custom neural network, reporting the architecture and trainable parameter count, plotting train/validation loss across epochs, and applying early stopping based on validation loss.

## Experimental Setup
- Dataset: `data/UCI HAR Dataset`
- Input features: 561
- Classes: 6
- Data split used in this run:
  - Train: 5881 samples
  - Validation: 1471 samples
  - Test: 2947 samples
- Device: CPU
- Early stopping monitor: `val_loss`

## Model Architecture
The custom model is a feed-forward MLP implemented in PyTorch:

1. `Linear(561 -> 256)`
2. `ReLU`
3. `Dropout(0.2)`
4. `Linear(256 -> 128)`
5. `ReLU`
6. `Dropout(0.2)`
7. `Linear(128 -> 6)`

## Trainable Parameters
Per-layer trainable parameters:
- `net.0.weight`: 143,616
- `net.0.bias`: 256
- `net.3.weight`: 32,768
- `net.3.bias`: 128
- `net.6.weight`: 768
- `net.6.bias`: 6

**Total trainable parameters: 177,542**

## Loss-Curve Interpretation (Train vs Validation)
From `results/task2_history.csv` and `results/task2_loss_curve.png`:
- Training loss decreases steadily from `0.6009` at epoch 1 to about `0.0213` by epoch 19.
- Validation loss drops quickly in the early stage and reaches the minimum at epoch 11 (`val_loss = 0.0518`).
- After epoch 11, validation loss fluctuates and does not improve consistently, while training loss keeps decreasing.

Interpretation: the model learns useful patterns quickly, and after the best epoch it starts showing mild overfitting behavior. Early stopping helps prevent unnecessary extra training.

## Early Stopping Summary
- Early stopping criterion: stop when `val_loss` does not improve for `patience = 8` epochs.
- Best epoch: **11**
- Best validation loss: **0.0518**
- Training stopped at epoch: **19**
- `stopped_early`: **true**

## Final Task 2 Performance
- Test loss: **0.1766**
- Test accuracy: **0.9454**

## One-Sentence Comparison with Task 1
Compared with the best Task 1 model (sklearn MLP accuracy `0.9511`), this Task 2 custom network (`0.9454`) is slightly lower by about `0.0057`, but still performs strongly on UCI HAR.

## Artifacts
- Metrics: `results/task2_metrics.json`
- History: `results/task2_history.csv`
- Loss plot: `results/task2_loss_curve.png`
- Model summary: `results/task2_model_summary.md`
- Script: `task2_custom_nn.py`

