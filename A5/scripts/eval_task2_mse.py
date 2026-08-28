from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.cifar10 import get_cifar10_datasets
from src.models import CNNAutoencoder


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: CNNAutoencoder, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


if __name__ == "__main__":
    data_root = "data"
    _, test_dataset = get_cifar10_datasets(root=data_root, download=False, normalize=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    checkpoint_path = PROJECT_ROOT / "reports" / "task2" / "data" / "task2_cnn_autoencoder.pt"
    device = resolve_device()

    model = CNNAutoencoder(latent_dim=4).to(device)
    load_checkpoint(model, checkpoint_path, device)

    criterion = nn.MSELoss(reduction="mean")
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            recon, _ = model(images)
            loss = criterion(recon, images)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

    mse = total_loss / max(total_samples, 1)
    print(f"recomputed_task2_test_mse={mse:.12f}")

