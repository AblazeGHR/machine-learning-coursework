from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10 import get_cifar10_loaders, label_names
from src.utils.plot import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CIFAR-10 loading and visualization.")
    parser.add_argument("--root", type=str, default="data", help="Path to CIFAR-10 data directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for preview.")
    parser.add_argument("--output", type=str, default="outputs/cifar10_batch_preview.png", help="Preview image output path.")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_loader, test_loader = get_cifar10_loaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=0,
        download=args.download,
        normalize=True,
        pin_memory=False,
    )

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    train_images, train_labels = train_batch
    test_images, test_labels = test_batch

    train_names = label_names(train_labels[: min(16, len(train_labels))])
    test_names = label_names(test_labels[: min(16, len(test_labels))])

    print(f"train_images: {tuple(train_images.shape)}")
    print(f"test_images: {tuple(test_images.shape)}")
    print(f"train_labels: {train_labels[:16].tolist()}")
    print(f"train_class_names: {train_names}")
    print(f"test_labels: {test_labels[:16].tolist()}")
    print(f"test_class_names: {test_names}")

    save_image_grid(
        train_images[:16],
        train_names,
        output_path=args.output,
        title="CIFAR-10 sample batch",
        nrow=8,
        normalize_input=True,
    )
    print(f"saved_preview: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

