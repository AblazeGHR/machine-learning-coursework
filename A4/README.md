# A4 - UCI HAR Task 1 (Step-by-step)

## Step 1: Random Forest baseline

This step trains a `RandomForestClassifier` using the official UCI HAR train/test split.

### Expected data path

- `data/UCI HAR Dataset/train/X_train.txt`
- `data/UCI HAR Dataset/train/y_train.txt`
- `data/UCI HAR Dataset/test/X_test.txt`
- `data/UCI HAR Dataset/test/y_test.txt`

### Run

```bash
python task1_rf_baseline.py
```

Optional:

```bash
python task1_rf_baseline.py --n-estimators 500 --random-state 42
```

## Step 2: sklearn MLP baseline

This step trains a `Pipeline(StandardScaler + MLPClassifier)` on the same official UCI HAR train/test split.

### Run

```bash
python task1_mlp_baseline.py
```

Optional:

```bash
python task1_mlp_baseline.py --max-iter 300 --random-state 42
```

### Compare Step 1 vs Step 2

Run both scripts and compare `RandomForest accuracy` with `MLP accuracy`.

```bash
python task1_rf_baseline.py
python task1_mlp_baseline.py
```

## Step 3: Quick tuning (small search)

Run a small, fast parameter search for both models and save the outputs:

```bash
python task1_quick_tuning.py
```

Generated files:

- `results/task1_tuning_runs.csv`
- `results/task1_after_tuning.md`

## Task 2: Custom Neural Network (PyTorch)

This step trains a custom MLP with `val_loss` early stopping and outputs model/curve artifacts.

### Run (CPU-first)

```bash
python task2_custom_nn.py --force-cpu
```

Optional faster/longer run:

```bash
python task2_custom_nn.py --force-cpu --max-epochs 120 --patience 15
```

Generated files:

- `results/task2_history.csv`
- `results/task2_metrics.json`
- `results/task2_model_summary.md`
- `results/task2_loss_curve.png`

## Task 3: Compare predictions (Task 1 vs Task 2)

Run a unified comparison over Task 1 models and the Task 2 custom network:

```bash
python task3_compare_models.py --force-cpu
```

Generated files:

- `results/task3_comparison.csv`
- `results/task3_pairwise_agreement.csv`
- `results/task3_predictions.csv`
- `results/task3_classification_reports.txt`
- `report/task3_report.md`
