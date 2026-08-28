# Task 3 Report: Prediction Comparison (Task 1 vs Task 2)

## Overall Metrics

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Task1_sklearn_MLP | 0.9511 | 0.9510 |
| Task2_CustomNN | 0.9454 | 0.9456 |
| Task1_RandomForest | 0.9287 | 0.9271 |

## Key Finding

- Best model in this Task 3 run: **Task1_sklearn_MLP**.
- Accuracy margin over the second best model: **0.0058**.

## Prediction Agreement

- Pairwise agreement details are saved in `results\task3_pairwise_agreement.csv`.

## Conclusion

Task 2 custom NN can be compared directly with Task 1 models by test-set predictions; in this run, the ranking above shows which model predicts most accurately under the same dataset protocol.

## Artifacts

- `results/task3_comparison.csv`
- `results/task3_pairwise_agreement.csv`
- `results/task3_predictions.csv`
- `results/task3_classification_reports.txt`
