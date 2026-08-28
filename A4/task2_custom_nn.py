import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class HARMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EarlyStoppingState:
    best_val_loss: float = math.inf
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    stopped_early: bool = False


def load_har_dataset(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = root / "train"
    test_dir = root / "test"

    x_train = np.loadtxt(train_dir / "X_train.txt").astype(np.float32)
    y_train = np.loadtxt(train_dir / "y_train.txt").astype(np.int64) - 1
    x_test = np.loadtxt(test_dir / "X_test.txt").astype(np.float32)
    y_test = np.loadtxt(test_dir / "y_test.txt").astype(np.int64) - 1
    return x_train, y_train, x_test, y_test


def standardize_from_train(
    train_x: np.ndarray, *other_arrays: np.ndarray
) -> Tuple[np.ndarray, ...]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0

    scaled = [(train_x - mean) / std]
    for arr in other_arrays:
        scaled.append((arr - mean) / std)
    return tuple(scaled)


def build_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_model_summary(model: nn.Module, out_path: Path) -> None:
    lines = ["# Task 2 Model Summary", "", "## Architecture", ""]
    for i, module in enumerate(model.net):
        lines.append(f"- Layer {i}: `{module}`")
    lines.append("")
    lines.append("## Trainable Parameters")
    lines.append("")

    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
            lines.append(f"- `{name}`: {count}")
    lines.append("")
    lines.append(f"- **Total trainable parameters**: {total}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += x_batch.size(0)
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        total_samples += x_batch.size(0)
    return total_loss / total_samples


def save_history_csv(history, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)


def save_loss_plot(history, out_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Task 2: Train/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Task 2 custom neural network on UCI HAR (CPU-first)")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "UCI HAR Dataset")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    x_train_full, y_train_full, x_test, y_test = load_har_dataset(args.data_root)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_train_full,
    )

    x_train, x_val, x_test = standardize_from_train(x_train, x_val, x_test)

    train_loader, val_loader, test_loader = build_loaders(
        x_train, y_train, x_val, y_val, x_test, y_test, args.batch_size
    )

    input_dim = x_train.shape[1]
    num_classes = int(y_train_full.max() + 1)

    model = HARMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_summary_path = args.results_dir / "task2_model_summary.md"
    write_model_summary(model, model_summary_path)

    early = EarlyStoppingState()
    best_state_dict = None
    history = []

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < early.best_val_loss - 1e-6:
            early.best_val_loss = val_loss
            early.best_epoch = epoch
            early.epochs_without_improvement = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            early.epochs_without_improvement += 1
            if early.epochs_without_improvement >= args.patience:
                early.stopped_early = True
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Best epoch: {early.best_epoch}, best val_loss: {early.best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    history_path = args.results_dir / "task2_history.csv"
    metrics_path = args.results_dir / "task2_metrics.json"
    plot_path = args.results_dir / "task2_loss_curve.png"

    save_history_csv(history, history_path)
    save_loss_plot(history, plot_path)

    metrics = {
        "device": str(device),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "best_epoch": int(early.best_epoch),
        "best_val_loss": float(early.best_val_loss),
        "stopped_early": bool(early.stopped_early),
        "epochs_ran": int(len(history)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "total_trainable_params": int(model_parameter_count(model)),
        "history_csv": str(history_path),
        "loss_curve_png": str(plot_path),
        "model_summary_md": str(model_summary_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved files:")
    print(f"- {history_path}")
    print(f"- {metrics_path}")
    print(f"- {plot_path}")
    print(f"- {model_summary_path}")


if __name__ == "__main__":
    main()

