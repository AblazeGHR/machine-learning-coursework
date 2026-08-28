import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
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


def main():
    parser = argparse.ArgumentParser(description="Task 1 Step 2: sklearn MLP baseline on UCI HAR")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data") / "UCI HAR Dataset",
        help="Path to UCI HAR Dataset directory",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_har_dataset(args.data_root)

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
                    max_iter=args.max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Data root: {args.data_root}")
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(f"MLP accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()

