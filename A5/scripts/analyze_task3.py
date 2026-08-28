from __future__ import annotations

import csv
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10 import CIFAR10_CLASSES, get_cifar10_datasets
from src.models import CNNAutoencoder


def load_checkpoint(model: CNNAutoencoder, checkpoint_path: Path, device: torch.device) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def extract_embeddings(model: CNNAutoencoder, loader: DataLoader, device: torch.device):
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


def compute_class_centers(embeddings: np.ndarray, labels: np.ndarray):
    centers = {}
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        class_embeddings = embeddings[labels == class_idx]
        centers[class_name] = class_embeddings.mean(axis=0)
    return centers


def compute_distance_matrix(centers: dict[str, np.ndarray]):
    class_names = list(CIFAR10_CLASSES)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
    for i, name_i in enumerate(class_names):
        for j, name_j in enumerate(class_names):
            diff = centers[name_i] - centers[name_j]
            matrix[i, j] = float(np.linalg.norm(diff))
    return class_names, matrix


def summarize_distances(class_names: list[str], matrix: np.ndarray):
    pairs = []
    for i, j in combinations(range(len(class_names)), 2):
        pairs.append((float(matrix[i, j]), class_names[i], class_names[j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    nearest_pair = pairs_sorted[0]
    farthest_pair = pairs_sorted[-1]

    per_class = {}
    for i, class_name in enumerate(class_names):
        others = [(float(matrix[i, j]), class_names[j]) for j in range(len(class_names)) if j != i]
        others_sorted = sorted(others, key=lambda x: x[0])
        per_class[class_name] = {
            "nearest": {"class": others_sorted[0][1], "distance": others_sorted[0][0]},
            "farthest": {"class": others_sorted[-1][1], "distance": others_sorted[-1][0]},
        }

    return {
        "nearest_pair": {"distance": nearest_pair[0], "class_a": nearest_pair[1], "class_b": nearest_pair[2]},
        "farthest_pair": {"distance": farthest_pair[0], "class_a": farthest_pair[1], "class_b": farthest_pair[2]},
        "per_class": per_class,
    }


def save_embeddings_npz(output_path: Path, embeddings: np.ndarray, labels: np.ndarray, class_names: list[str]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        labels=labels,
        class_names=np.array(class_names, dtype=object),
    )


def save_centers_json(output_path: Path, centers: dict[str, np.ndarray]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: center.tolist() for name, center in centers.items()}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_distance_matrix_csv(output_path: Path, class_names: list[str], matrix: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + [f"{value:.8f}" for value in row])


def save_summary_json(output_path: Path, summary: dict, class_names: list[str], matrix: np.ndarray, embeddings: np.ndarray, labels: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_classes": len(class_names),
        "class_names": class_names,
        "distance_matrix": matrix.tolist(),
        **summary,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_heatmap(output_path: Path, class_names: list[str], matrix: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title("Task 3: Class center distance matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_distance_bars(output_path: Path, class_names: list[str], matrix: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_distances = []
    for i in range(len(class_names)):
        others = [matrix[i, j] for j in range(len(class_names)) if j != i]
        mean_distances.append(float(np.mean(others)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, mean_distances)
    ax.set_ylabel("Mean distance to other class centers")
    ax.set_title("Task 3: Mean distance per class")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_results(class_names: list[str], matrix: np.ndarray, embeddings: np.ndarray, labels: np.ndarray) -> None:
    assert embeddings.shape[1] == 4, f"Expected 4D embeddings, got {embeddings.shape[1]}"
    assert matrix.shape == (10, 10), f"Expected 10x10 matrix, got {matrix.shape}"
    assert np.allclose(matrix, matrix.T), "Distance matrix must be symmetric"
    assert np.allclose(np.diag(matrix), 0.0), "Distance matrix diagonal must be zero"
    assert np.all(np.isfinite(matrix)), "Distance matrix contains non-finite values"
    assert len(set(labels.tolist())) == 10, "All 10 classes must appear in the test set"
    assert len(class_names) == 10, "Expected 10 class names"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task2_checkpoint = PROJECT_ROOT / "reports" / "task2" / "data" / "task2_cnn_autoencoder.pt"
    output_root = PROJECT_ROOT / "reports" / "task3"
    data_dir = output_root / "data"
    figures_dir = output_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _, test_dataset = get_cifar10_datasets(root="data", download=False, normalize=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    model = CNNAutoencoder(latent_dim=4).to(device)
    load_checkpoint(model, task2_checkpoint, device)

    embeddings, labels = extract_embeddings(model, test_loader, device)
    centers = compute_class_centers(embeddings, labels)
    class_names, matrix = compute_distance_matrix(centers)
    summary = summarize_distances(class_names, matrix)

    validate_results(class_names, matrix, embeddings, labels)

    save_embeddings_npz(data_dir / "task3_embeddings.npz", embeddings, labels, class_names)
    save_centers_json(data_dir / "task3_class_centers.json", centers)
    save_distance_matrix_csv(data_dir / "task3_distance_matrix.csv", class_names, matrix)
    save_summary_json(data_dir / "task3_distance_summary.json", summary, class_names, matrix, embeddings, labels)
    (data_dir / "task3_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(task2_checkpoint),
                "dataset_root": "data",
                "split": "test",
                "batch_size": 256,
                "latent_dim": 4,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_heatmap(figures_dir / "task3_distance_heatmap.png", class_names, matrix)
    plot_mean_distance_bars(figures_dir / "task3_center_distance_bar.png", class_names, matrix)

    print(f"num_samples={embeddings.shape[0]}")
    print(f"embedding_dim={embeddings.shape[1]}")
    print(f"nearest_pair={summary['nearest_pair']}")
    print(f"farthest_pair={summary['farthest_pair']}")
    print(f"saved_data_dir={data_dir}")
    print(f"saved_figures_dir={figures_dir}")


if __name__ == "__main__":
    main()

