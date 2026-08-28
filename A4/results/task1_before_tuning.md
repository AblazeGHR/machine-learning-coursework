# Task 1 Results Before Tuning

## Setup

- Dataset: `data/UCI HAR Dataset`
- Split: official train/test split from UCI HAR
- Train shape: `(7352, 561)`
- Test shape: `(2947, 561)`

## Baseline Results

| Model | Accuracy |
|---|---:|
| RandomForest (`task1_rf_baseline.py`) | 0.9287 |
| MLP (`task1_mlp_baseline.py`) | 0.9511 |

## Quick Comparison

- Absolute gain of MLP over RandomForest: `0.0224` (2.24 percentage points)
- Current winner before tuning: **MLP**

## Raw Logs

- `results/step1_baseline/rf_stdout.txt`
- `results/step1_baseline/mlp_stdout.txt`

