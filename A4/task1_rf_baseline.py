import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_har_dataset(root: Path):
    train_dir = root / "train"
    test_dir = root / "test"

    x_train = np.loadtxt(train_dir / "X_train.txt")
    y_train = np.loadtxt(train_dir / "y_train.txt").astype(int)
    x_test = np.loadtxt(test_dir / "X_test.txt")
    y_test = np.loadtxt(test_dir / "y_test.txt").astype(int)

    return x_train, y_train, x_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Task 1 Step 1: Random Forest baseline on UCI HAR")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data") / "UCI HAR Dataset",
        help="Path to UCI HAR Dataset directory",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_har_dataset(args.data_root)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Data root: {args.data_root}")
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(f"RandomForest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()

