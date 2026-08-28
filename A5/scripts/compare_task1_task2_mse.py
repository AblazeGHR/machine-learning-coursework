from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.cifar10 import get_cifar10_datasets
from src.models import MLPAutoencoder, CNNAutoencoder


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def extract_per_image_mse(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    per_image = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            recon, _ = model(images)
            # per-element squared error then mean over channels/height/width per image
            se = (recon - images) ** 2
            # flatten per image and mean per image
            per_elem_mean = se.view(se.size(0), -1).mean(dim=1).cpu().numpy()
            per_image.append(per_elem_mean)
    return np.concatenate(per_image, axis=0)


if __name__ == "__main__":
    device = resolve_device()
    # load test dataset without normalization (same as training scripts)
    _, test_dataset = get_cifar10_datasets(root="data", download=False, normalize=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # checkpoints
    ckpt1 = PROJECT_ROOT / "reports" / "task1" / "data" / "task1_mlp_autoencoder.pt"
    ckpt2 = PROJECT_ROOT / "reports" / "task2" / "data" / "task2_cnn_autoencoder.pt"

    model1 = MLPAutoencoder(latent_dim=4).to(device)
    model2 = CNNAutoencoder(latent_dim=4).to(device)
    load_checkpoint(model1, ckpt1, device)
    load_checkpoint(model2, ckpt2, device)

    per_mse1 = extract_per_image_mse(model1, test_loader, device)
    per_mse2 = extract_per_image_mse(model2, test_loader, device)

    assert per_mse1.shape == per_mse2.shape, "Per-image arrays must match in length"
    n = per_mse1.shape[0]

    diff = per_mse2 - per_mse1  # CNN - MLP

    # summary stats
    summary = {
        "n": int(n),
        "mlp_mean_mse": float(per_mse1.mean()),
        "mlp_median_mse": float(np.median(per_mse1)),
        "mlp_std_mse": float(per_mse1.std()),
        "cnn_mean_mse": float(per_mse2.mean()),
        "cnn_median_mse": float(np.median(per_mse2)),
        "cnn_std_mse": float(per_mse2.std()),
        "diff_mean": float(diff.mean()),
        "diff_median": float(np.median(diff)),
        "diff_std": float(diff.std()),
        "cnn_better_count": int(np.sum(diff < 0)),
        "mlp_better_count": int(np.sum(diff > 0)),
        "equal_count": int(np.sum(np.isclose(diff, 0.0, atol=1e-12))),
    }

    out_dir = PROJECT_ROOT / "reports" / "compare"
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # save arrays
    np.savez_compressed(out_dir / "task1_task2_per_image_mse.npz", mlp=per_mse1, cnn=per_mse2, diff=diff)

    # histograms
    plt.figure(figsize=(8, 5))
    plt.hist(per_mse1, bins=100, alpha=0.5, label="MLP per-image MSE")
    plt.hist(per_mse2, bins=100, alpha=0.5, label="CNN per-image MSE")
    plt.legend()
    plt.xlabel("per-image MSE")
    plt.ylabel("count")
    plt.title("Per-image MSE distribution: MLP vs CNN (test set)")
    plt.tight_layout()
    plt.savefig(figs_dir / "per_image_mse_histogram.png", dpi=200)
    plt.close()

    # diff histogram
    plt.figure(figsize=(8, 4))
    plt.hist(diff, bins=100, alpha=0.8)
    plt.axvline(0.0, color="k", linestyle="--")
    plt.xlabel("CNN MSE - MLP MSE (per-image)")
    plt.ylabel("count")
    plt.title("Per-image MSE difference (CNN - MLP)")
    plt.tight_layout()
    plt.savefig(figs_dir / "per_image_mse_diff_histogram.png", dpi=200)
    plt.close()

    # save summary
    import json
    (out_dir / "task1_task2_mse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # print summary
    print("Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"saved_arrays: {out_dir / 'task1_task2_per_image_mse.npz'}")
    print(f"saved_figures: {figs_dir}")
    print(f"saved_summary: {out_dir / 'task1_task2_mse_summary.json'}")

