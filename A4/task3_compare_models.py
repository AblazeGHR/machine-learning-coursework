import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from task2_custom_nn import HARMLP, build_loaders, evaluate, standardize_from_train, train_one_epoch


def load_task1_data(root: Path):
    train_dir = root / "train"
    test_dir = root / "test"

    x_train = np.loadtxt(train_dir / "X_train.txt")
    y_train = np.loadtxt(train_dir / "y_train.txt").astype(int)
    x_test = np.loadtxt(test_dir / "X_test.txt")
    y_test = np.loadtxt(test_dir / "y_test.txt").astype(int)
    return x_train, y_train, x_test, y_test


def train_rf_predict(x_train, y_train, x_test, random_state: int):
    model = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def train_sklearn_mlp_predict(x_train, y_train, x_test, random_state: int):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    batch_size=128,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def train_task2_predict(
    x_train_full,
    y_train_full,
    x_test,
    random_state: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    val_size: float,
    force_cpu: bool,
):
    # Task 2 uses labels 0..5 internally.
    y_train_full_zero = y_train_full - 1

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full_zero,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full_zero,
    )

    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_train, x_val, x_test = standardize_from_train(x_train, x_val, x_test)

    # Dummy labels for loader compatibility; not used for test metric in this function.
    y_test_dummy = np.zeros(x_test.shape[0], dtype=np.int64)

    train_loader, val_loader, test_loader = build_loaders(
        x_train,
        y_train.astype(np.int64),
        x_val,
        y_val.astype(np.int64),
        x_test,
        y_test_dummy,
        batch_size,
    )

    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = HARMLP(input_dim=x_train.shape[1], num_classes=6).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    for _ in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _ = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_zero = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds_zero.append(logits.argmax(dim=1).cpu().numpy())

    pred_zero = np.concatenate(preds_zero)
    return pred_zero + 1


def save_comparison_table(rows, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)


def save_pairwise(pred_dict, out_path: Path):
    names = list(pred_dict.keys())
    rows = []
    for a, b in combinations(names, 2):
        pa = pred_dict[a]
        pb = pred_dict[b]
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "agreement_rate": float(np.mean(pa == pb)),
                "cohen_kappa": float(cohen_kappa_score(pa, pb)),
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_a", "model_b", "agreement_rate", "cohen_kappa"])
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(y_true, pred_dict, out_path: Path):
    names = list(pred_dict.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", *names])
        for i in range(len(y_true)):
            writer.writerow([int(y_true[i]), *[int(pred_dict[name][i]) for name in names]])


def build_task3_report(rows, pairwise_path: Path, output_path: Path):
    sorted_rows = sorted(rows, key=lambda x: x["accuracy"], reverse=True)
    best = sorted_rows[0]
    second = sorted_rows[1]
    gap = best["accuracy"] - second["accuracy"]

    lines = [
        "# Task 3 Report: Prediction Comparison (Task 1 vs Task 2)",
        "",
        "## Overall Metrics",
        "",
        "| Model | Accuracy | Macro F1 |",
        "|---|---:|---:|",
    ]

    for row in sorted_rows:
        lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")

    lines.extend(
        [
            "",
            "## Key Finding",
            "",
            f"- Best model in this Task 3 run: **{best['model']}**.",
            f"- Accuracy margin over the second best model: **{gap:.4f}**.",
            "",
            "## Prediction Agreement",
            "",
            f"- Pairwise agreement details are saved in `{pairwise_path}`.",
            "",
            "## Conclusion",
            "",
            "Task 2 custom NN can be compared directly with Task 1 models by test-set predictions; in this run, the ranking above shows which model predicts most accurately under the same dataset protocol.",
            "",
            "## Artifacts",
            "",
            "- `results/task3_comparison.csv`",
            "- `results/task3_pairwise_agreement.csv`",
            "- `results/task3_predictions.csv`",
            "- `results/task3_classification_reports.txt`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Task 3: compare predictions from Task 1 and Task 2")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "UCI HAR Dataset")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--report-path", type=Path, default=Path("report") / "task3_report.md")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_task1_data(args.data_root)

    rf_pred = train_rf_predict(x_train, y_train, x_test, args.random_state)
    sk_mlp_pred = train_sklearn_mlp_predict(x_train, y_train, x_test, args.random_state)
    task2_pred = train_task2_predict(
        x_train_full=x_train,
        y_train_full=y_train,
        x_test=x_test,
        random_state=args.random_state,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_size=args.val_size,
        force_cpu=args.force_cpu,
    )

    pred_dict = {
        "Task1_RandomForest": rf_pred,
        "Task1_sklearn_MLP": sk_mlp_pred,
        "Task2_CustomNN": task2_pred,
    }

    rows = []
    report_lines = []
    for name, pred in pred_dict.items():
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        rows.append({"model": name, "accuracy": acc, "macro_f1": macro_f1})

        report_lines.append(f"=== {name} ===")
        report_lines.append(classification_report(y_test, pred, digits=4))
        report_lines.append("")

    comparison_csv = args.results_dir / "task3_comparison.csv"
    pairwise_csv = args.results_dir / "task3_pairwise_agreement.csv"
    pred_csv = args.results_dir / "task3_predictions.csv"
    report_txt = args.results_dir / "task3_classification_reports.txt"

    save_comparison_table(rows, comparison_csv)
    save_pairwise(pred_dict, pairwise_csv)
    save_predictions(y_test, pred_dict, pred_csv)
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")

    build_task3_report(rows, pairwise_csv, args.report_path)

    print(f"Saved: {comparison_csv}")
    print(f"Saved: {pairwise_csv}")
    print(f"Saved: {pred_csv}")
    print(f"Saved: {report_txt}")
    print(f"Saved: {args.report_path}")


if __name__ == "__main__":
    main()

