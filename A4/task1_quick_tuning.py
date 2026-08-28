import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_har_dataset(root: Path):
    train_dir = root / "train"
    test_dir = root / "test"

    x_train = np.loadtxt(train_dir / "X_train.txt")
    y_train = np.loadtxt(train_dir / "y_train.txt").astype(int)
    x_test = np.loadtxt(test_dir / "X_test.txt")
    y_test = np.loadtxt(test_dir / "y_test.txt").astype(int)
    return x_train, y_train, x_test, y_test


def run_rf(config, x_train, y_train, x_test, y_test):
    start = time.perf_counter()
    model = RandomForestClassifier(**config)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    elapsed = time.perf_counter() - start
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "seconds": elapsed,
    }


def run_mlp(config, x_train, y_train, x_test, y_test):
    start = time.perf_counter()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**config)),
        ]
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    elapsed = time.perf_counter() - start
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "seconds": elapsed,
    }


def main():
    data_root = Path("data") / "UCI HAR Dataset"
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "task1_tuning_runs.csv"
    md_path = results_dir / "task1_after_tuning.md"

    baseline = {
        "RandomForest": 0.9287,
        "MLP": 0.9511,
    }

    x_train, y_train, x_test, y_test = load_har_dataset(data_root)

    rf_configs = [
        {
            "run_id": "rf_n300",
            "params": {
                "n_estimators": 300,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        {
            "run_id": "rf_n600",
            "params": {
                "n_estimators": 600,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        {
            "run_id": "rf_n600_balanced",
            "params": {
                "n_estimators": 600,
                "class_weight": "balanced_subsample",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    ]

    mlp_configs = [
        {
            "run_id": "mlp_128_64",
            "params": {
                "hidden_layer_sizes": (128, 64),
                "activation": "relu",
                "solver": "adam",
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "batch_size": 128,
                "max_iter": 200,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "random_state": 42,
            },
        },
        {
            "run_id": "mlp_256_128",
            "params": {
                "hidden_layer_sizes": (256, 128),
                "activation": "relu",
                "solver": "adam",
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "batch_size": 128,
                "max_iter": 300,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 12,
                "random_state": 42,
            },
        },
        {
            "run_id": "mlp_256_128_64",
            "params": {
                "hidden_layer_sizes": (256, 128, 64),
                "activation": "relu",
                "solver": "adam",
                "alpha": 5e-5,
                "learning_rate_init": 8e-4,
                "batch_size": 128,
                "max_iter": 400,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 15,
                "random_state": 42,
            },
        },
    ]

    rows = []

    for cfg in rf_configs:
        metrics = run_rf(cfg["params"], x_train, y_train, x_test, y_test)
        rows.append(
            {
                "model": "RandomForest",
                "run_id": cfg["run_id"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "seconds": metrics["seconds"],
                "delta_vs_baseline": metrics["accuracy"] - baseline["RandomForest"],
                "params": json.dumps(cfg["params"], ensure_ascii=True),
            }
        )

    for cfg in mlp_configs:
        metrics = run_mlp(cfg["params"], x_train, y_train, x_test, y_test)
        serializable_params = dict(cfg["params"])
        serializable_params["hidden_layer_sizes"] = list(serializable_params["hidden_layer_sizes"])
        rows.append(
            {
                "model": "MLP",
                "run_id": cfg["run_id"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "seconds": metrics["seconds"],
                "delta_vs_baseline": metrics["accuracy"] - baseline["MLP"],
                "params": json.dumps(serializable_params, ensure_ascii=True),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["accuracy"], reverse=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "run_id",
                "accuracy",
                "macro_f1",
                "seconds",
                "delta_vs_baseline",
                "params",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    best_rf = max((r for r in rows if r["model"] == "RandomForest"), key=lambda r: r["accuracy"])
    best_mlp = max((r for r in rows if r["model"] == "MLP"), key=lambda r: r["accuracy"])

    report = [
        "# Task 1 Results After Quick Tuning",
        "",
        "## Baselines",
        "",
        f"- RandomForest baseline accuracy: {baseline['RandomForest']:.4f}",
        f"- MLP baseline accuracy: {baseline['MLP']:.4f}",
        "",
        "## Best Tuned Runs",
        "",
        f"- RandomForest best: `{best_rf['run_id']}` | accuracy `{best_rf['accuracy']:.4f}` | delta `{best_rf['delta_vs_baseline']:+.4f}`",
        f"- MLP best: `{best_mlp['run_id']}` | accuracy `{best_mlp['accuracy']:.4f}` | delta `{best_mlp['delta_vs_baseline']:+.4f}`",
        "",
        "## Artifacts",
        "",
        "- `results/task1_tuning_runs.csv`",
        "- `results/task1_before_tuning.md`",
    ]
    md_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Saved tuning table: {csv_path}")
    print(f"Saved tuning summary: {md_path}")
    print("\nTop 6 runs by accuracy:")
    for r in rows_sorted[:6]:
        print(
            f"{r['model']:12s} {r['run_id']:20s} "
            f"acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f} "
            f"delta={r['delta_vs_baseline']:+.4f} time={r['seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()

