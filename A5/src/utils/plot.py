from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from src.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def save_image_grid(
    images: torch.Tensor,
    labels: Iterable[str],
    output_path: str | Path,
    title: str = "CIFAR-10 preview",
    nrow: int = 8,
    normalize_input: bool = True,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(labels)

    display_images = denormalize(images) if normalize_input else images
    grid = make_grid(display_images[: len(labels)], nrow=nrow, padding=2)
    grid = grid.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(grid)
    ax.set_title(title)
    ax.axis("off")

    caption = " | ".join(labels)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

