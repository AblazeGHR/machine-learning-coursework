# Task 1 Report (One-Page, Concise)

## Objective
This task compares two classifiers on the UCI Human Activity Recognition (HAR) dataset:
1) a `RandomForestClassifier`, and 2) an `MLPClassifier` from scikit-learn.
The goal is to identify which model performs better on activity prediction accuracy.

## Experimental Setup
- Dataset: `UCI HAR Dataset`
- Split: official predefined split (`train`/`test`)
- Train size: `(7352, 561)`
- Test size: `(2947, 561)`
- Models were evaluated on the same test set for a fair comparison.

Baseline configurations:
- Random Forest: `n_estimators=300`, `random_state=42`
- MLP: `Pipeline(StandardScaler + MLPClassifier(hidden_layer_sizes=(128,64), early_stopping=True, random_state=42))`

## Results Before Tuning
| Model | Test Accuracy |
|---|---:|
| Random Forest | 0.9287 |
| MLP | 0.9511 |

Key observation:
- MLP outperformed Random Forest by `0.0224` (2.24 percentage points).

## Quick Tuning (Small Search)
A small, fast parameter search was performed for both models.

- Random Forest candidates: `rf_n300`, `rf_n600`, `rf_n600_balanced`
- MLP candidates: `mlp_128_64`, `mlp_256_128`, `mlp_256_128_64`

Best tuned results:
- Best Random Forest: `rf_n300`, accuracy `0.9287`
- Best MLP: `mlp_128_64`, accuracy `0.9511`

Interpretation:
- In this quick search, tuning did not produce a meaningful improvement beyond baseline for either model.
- Larger MLP/RF settings increased training time but did not improve test accuracy.

## Final Comparison and Conclusion
On this task and split, **MLP is the better-performing model**.

- Final best Random Forest accuracy: `0.9287`
- Final best MLP accuracy: `0.9511`
- Final gap (MLP - RF): `+0.0224`

Therefore, for Task 1, the scikit-learn MLP provides higher predictive accuracy than Random Forest on UCI HAR under the tested configurations.

## Reproducibility Artifacts
- Baseline summary: `results/task1_before_tuning.md`
- Tuning summary: `results/task1_after_tuning.md`
- Full tuning table: `results/task1_tuning_runs.csv`
- Scripts: `task1_rf_baseline.py`, `task1_mlp_baseline.py`, `task1_quick_tuning.py`

