from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10 import CIFAR10_CLASSES, get_cifar10_datasets
from src.models import MLPAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Task 1 embeddings with a 2D scatter plot.")
    parser.add_argument("--root", type=str, default="data", help="Path to CIFAR-10 data directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "task1" / "data" / "task1_mlp_autoencoder.pt"),
        help="Path to the Task 1 checkpoint.",
    )
    parser.add_argument("--num-samples", type=int, default=800, help="Number of random test samples to visualize.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size used for embedding extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda; defaults to auto-detect.")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def load_checkpoint(model: MLPAutoencoder, checkpoint_path: Path, device: torch.device) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def extract_embeddings(model: MLPAutoencoder, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            embeddings = model.encode(images).cpu().numpy()
            all_embeddings.append(embeddings)
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def plot_embedding_scatter(
    output_path: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(11, 9))
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if not np.any(mask):
            continue
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18,
            alpha=0.75,
            color=colors[class_idx % len(colors)],
            label=class_name,
            edgecolors="none",
        )

    ax.set_title("Task 1: 800 CIFAR-10 samples projected to 2D (PCA of 4D latent space)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rng = set_seed(args.seed)

    _, test_dataset = get_cifar10_datasets(root=args.root, download=args.download, normalize=False)
    sample_size = min(args.num_samples, len(test_dataset))
    sample_indices = rng.choice(len(test_dataset), size=sample_size, replace=False)
    sample_indices = np.sort(sample_indices)

    subset = Subset(test_dataset, sample_indices.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MLPAutoencoder(latent_dim=4).to(device)
    checkpoint_path = Path(args.checkpoint)
    load_checkpoint(model, checkpoint_path, device)

    embeddings_4d, labels = extract_embeddings(model, loader, device)
    pca = PCA(n_components=2, random_state=args.seed)
    coordinates_2d = pca.fit_transform(embeddings_4d)

    output_root = PROJECT_ROOT / "reports" / "task1"
    figures_dir = output_root / "figures"
    data_dir = output_root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    scatter_path = figures_dir / "task1_embedding_scatter.png"
    plot_embedding_scatter(scatter_path, coordinates_2d, labels, CIFAR10_CLASSES)

    np.savez_compressed(
        data_dir / "task1_embedding_scatter.npz",
        sample_indices=sample_indices,
        labels=labels,
        embeddings_4d=embeddings_4d,
        coordinates_2d=coordinates_2d,
        class_names=np.array(CIFAR10_CLASSES, dtype=object),
        pca_components=pca.components_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
    )

    metadata_path = data_dir / "task1_embedding_scatter.json"
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "dataset_root": args.root,
                "split": "test",
                "num_samples": int(sample_size),
                "seed": int(args.seed),
                "projection": "PCA",
                "embedding_dim": 4,
                "output_figure": str(scatter_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"num_samples={sample_size}")
    print(f"embedding_dim={embeddings_4d.shape[1]}")
    print(f"pca_explained_variance_ratio={pca.explained_variance_ratio_.tolist()}")
    print(f"saved_figure={scatter_path}")
    print(f"saved_data={data_dir / 'task1_embedding_scatter.npz'}")
    print(f"saved_metadata={metadata_path}")


if __name__ == "__main__":
    main()

