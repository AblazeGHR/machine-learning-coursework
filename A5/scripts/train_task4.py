from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10 import get_cifar10_loaders
from src.models import CNNAutoencoder
from src.utils.plot import save_image_grid


def parse_dims(text: str) -> list[int]:
    dims = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not dims:
        raise ValueError("At least one latent dimension is required.")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 4 CNN Autoencoder sweep on CIFAR-10.")
    parser.add_argument("--root", type=str, default="data", help="Path to CIFAR-10 data directory.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per latent dimension.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--dims", type=str, default="1,2,4,8,16", help="Comma-separated latent dimensions to sweep.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda; defaults to auto-detect.")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer=None,
    device: torch.device | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    iterator = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    for images, _ in iterator:
        images = images.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        recon, _ = model(images)
        loss = criterion(recon, images)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        iterator.set_postfix(loss=loss.item())

    return total_loss / max(total_samples, 1)


def save_reconstruction_preview(model: nn.Module, loader, device: torch.device, output_path: Path) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        recon, _ = model(images)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired = torch.cat([images[:8], recon[:8]], dim=0)
    labels = ["orig"] * 8 + ["recon"] * 8
    save_image_grid(
        paired,
        labels,
        output_path=output_path,
        title=f"Task 4: reconstruction preview (latent_dim={model.latent_dim})",
        nrow=8,
        normalize_input=False,
    )


def save_loss_curve(history: list[dict], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    test_losses = [item["test_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="train_loss")
    ax.plot(epochs, test_losses, marker="o", label="test_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(output_path: Path, rows: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["latent_dim", "final_train_loss", "final_test_loss", "epochs", "batch_size", "lr", "seed"])
        writer.writeheader()
        writer.writerows(rows)


def plot_test_error(output_path: Path, rows: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dims = [row["latent_dim"] for row in rows]
    test_losses = [row["final_test_loss"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, test_losses, marker="o")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Final test loss")
    ax.set_title("Task 4: Test error vs latent dimension")
    ax.set_xticks(dims)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_grid(output_path: Path, histories: dict[int, list[dict]]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dims = list(histories.keys())
    ncols = 2
    nrows = int(np.ceil(len(dims) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)

    for idx, dim in enumerate(dims):
        ax = axes[idx // ncols][idx % ncols]
        history = histories[dim]
        epochs = [item["epoch"] for item in history]
        train_losses = [item["train_loss"] for item in history]
        test_losses = [item["test_loss"] for item in history]
        ax.plot(epochs, train_losses, marker="o", label="train_loss")
        ax.plot(epochs, test_losses, marker="o", label="test_loss")
        ax.set_title(f"latent_dim={dim}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for idx in range(len(dims), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Task 4: Loss curves across latent dimensions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dims = parse_dims(args.dims)

    train_loader, test_loader = get_cifar10_loaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=0,
        download=args.download,
        normalize=False,
        pin_memory=(device.type == "cuda"),
    )

    output_root = PROJECT_ROOT / "reports" / "task4"
    figures_dir = output_root / "figures"
    data_dir = output_root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "root": args.root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dims": dims,
        "device": str(device),
        "seed": args.seed,
    }
    (data_dir / "task4_sweep_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict] = []
    histories: dict[int, list[dict]] = {}

    for dim in dims:
        print(f"=== latent_dim={dim} ===")
        model = CNNAutoencoder(latent_dim=dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        history = []
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
            test_loss = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
            history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
            print(f"latent_dim={dim} epoch={epoch} train_loss={train_loss:.6f} test_loss={test_loss:.6f}")

        histories[dim] = history
        final_train_loss = history[-1]["train_loss"]
        final_test_loss = history[-1]["test_loss"]
        rows.append(
            {
                "latent_dim": dim,
                "final_train_loss": final_train_loss,
                "final_test_loss": final_test_loss,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
            }
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "latent_dim": dim,
                "history": history,
                "args": {**config, "latent_dim": dim},
            },
            data_dir / f"task4_latent_{dim}.pt",
        )
        (data_dir / f"task4_latent_{dim}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_loss_curve(history, figures_dir / f"task4_latent_{dim}_loss_curve.png", title=f"Task 4: Loss curve (latent_dim={dim})")

    write_csv(data_dir / "task4_sweep_results.csv", rows)
    (data_dir / "task4_sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    plot_test_error(figures_dir / "task4_test_error_vs_latent_dim.png", rows)
    plot_loss_grid(figures_dir / "task4_loss_curves.png", histories)

    best_row = min(rows, key=lambda item: item["final_test_loss"])
    best_dim = int(best_row["latent_dim"])
    best_checkpoint = data_dir / f"task4_latent_{best_dim}.pt"
    best_model = CNNAutoencoder(latent_dim=best_dim).to(device)
    checkpoint = torch.load(best_checkpoint, map_location=device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    save_reconstruction_preview(best_model, test_loader, device, figures_dir / f"task4_latent_{best_dim}_reconstruction_preview.png")

    baseline_dim = 4
    if baseline_dim in dims:
        baseline_model = CNNAutoencoder(latent_dim=baseline_dim).to(device)
        checkpoint = torch.load(data_dir / f"task4_latent_{baseline_dim}.pt", map_location=device)
        baseline_model.load_state_dict(checkpoint["model_state_dict"])
        save_reconstruction_preview(baseline_model, test_loader, device, figures_dir / f"task4_latent_{baseline_dim}_reconstruction_preview.png")

    print(f"best_latent_dim={best_dim}")
    print(f"best_test_loss={best_row['final_test_loss']:.6f}")
    print(f"saved_data_dir={data_dir}")
    print(f"saved_figures_dir={figures_dir}")


if __name__ == "__main__":
    main()

