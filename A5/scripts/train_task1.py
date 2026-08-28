from __future__ import annotations

import argparse
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
from src.models import MLPAutoencoder
from src.utils.plot import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 1 MLP Autoencoder on CIFAR-10.")
    parser.add_argument("--root", type=str, default="data", help="Path to CIFAR-10 data directory.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=4, help="Latent dimension.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda; defaults to auto-detect.")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit training batches for a quick smoke test.")
    parser.add_argument("--max-test-batches", type=int, default=None, help="Limit test batches for a quick smoke test.")
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
    max_batches: int | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    iterator = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    for batch_idx, (images, _) in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

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
        title="Task 1: original vs reconstruction",
        nrow=8,
        normalize_input=False,
    )


def save_loss_curve(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    test_losses = [item["test_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="train_loss")
    ax.plot(epochs, test_losses, marker="o", label="test_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Task 1: Loss curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_loader, test_loader = get_cifar10_loaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=0,
        download=args.download,
        normalize=False,
        pin_memory=(device.type == "cuda"),
    )

    model = MLPAutoencoder(latent_dim=args.latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_root = PROJECT_ROOT / "reports" / "task1"
    figures_dir = output_root / "figures"
    data_dir = output_root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
        )
        test_loss = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer=None,
            device=device,
            max_batches=args.max_test_batches,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} test_loss={test_loss:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "history": history,
        "args": vars(args),
    }
    ckpt_path = data_dir / "task1_mlp_autoencoder.pt"
    torch.save(checkpoint, ckpt_path)

    history_path = data_dir / "task1_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    preview_path = figures_dir / "task1_reconstruction_preview.png"
    save_reconstruction_preview(model, test_loader, device, preview_path)

    loss_curve_path = figures_dir / "task1_loss_curve.png"
    save_loss_curve(history, loss_curve_path)

    print(f"saved_checkpoint: {ckpt_path}")
    print(f"saved_history: {history_path}")
    print(f"saved_preview: {preview_path}")
    print(f"saved_loss_curve: {loss_curve_path}")


if __name__ == "__main__":
    main()

