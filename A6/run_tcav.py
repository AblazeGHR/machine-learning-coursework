import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        b = self.features(x)
        return self.classifier(b)

    def forward_with_bottleneck(self, x):
        b = self.features(x)
        logits = self.classifier(b)
        return logits, b


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, test_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Train epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        acc = evaluate_accuracy(model, test_loader, device)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, test_acc={acc:.4f}")


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def draw_line(img, x0, y0, x1, y1, value=1.0, thickness=1):
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, steps)
    ys = np.linspace(y0, y1, steps)
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                xx, yy = xi + dx, yi + dy
                if 0 <= xx < img.shape[1] and 0 <= yy < img.shape[0]:
                    img[yy, xx] = max(img[yy, xx], value)


def generate_x_image(size=28, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    img = rng.normal(0.02, 0.02, (size, size)).clip(0.0, 0.2)
    margin = rng.integers(5, 8)
    thickness = int(rng.integers(1, 2))
    draw_line(img, margin, margin, size - margin - 1, size - margin - 1, value=1.0, thickness=thickness)
    draw_line(img, size - margin - 1, margin, margin, size - margin - 1, value=1.0, thickness=thickness)
    return img.clip(0.0, 1.0).astype(np.float32)


def generate_non_x_image(size=28, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    img = rng.normal(0.02, 0.02, (size, size)).clip(0.0, 0.2)
    margin = int(rng.integers(4, 8))
    thickness = int(rng.integers(1, 2))

    choice = int(rng.integers(0, 3))
    if choice == 0:
        y = int(rng.integers(margin, size - margin))
        draw_line(img, margin, y, size - margin - 1, y, value=1.0, thickness=thickness)
    elif choice == 1:
        x = int(rng.integers(margin, size - margin))
        draw_line(img, x, margin, x, size - margin - 1, value=1.0, thickness=thickness)
    else:
        x0 = int(rng.integers(margin, size // 2))
        y0 = int(rng.integers(margin, size - margin))
        x1 = int(rng.integers(size // 2, size - margin))
        y1 = int(rng.integers(margin, size - margin))
        draw_line(img, x0, y0, x1, y1, value=1.0, thickness=thickness)

    return img.clip(0.0, 1.0).astype(np.float32)


def make_concept_batch(n, generator, seed):
    rng = np.random.default_rng(seed)
    data = [generator(rng=rng) for _ in range(n)]
    arr = np.stack(data, axis=0)
    return torch.from_numpy(arr).unsqueeze(1)


def extract_activations(model, images, device, batch_size=64):
    model.eval()
    acts = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = images[i : i + batch_size].to(device)
            _, b = model.forward_with_bottleneck(x)
            acts.append(b.reshape(b.size(0), -1).cpu().numpy())
    return np.concatenate(acts, axis=0)


def train_cav(concept_acts, random_acts, seed):
    X = np.vstack([concept_acts, random_acts])
    y = np.array([1] * len(concept_acts) + [0] * len(random_acts))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LinearSVC(random_state=seed, dual=False, max_iter=20000)
    clf.fit(Xs, y)

    cav = clf.coef_[0]
    # Align CAV orientation so it points from random set toward concept set.
    concept_proj = Xs[: len(concept_acts)] @ cav
    random_proj = Xs[len(concept_acts) :] @ cav
    if concept_proj.mean() < random_proj.mean():
        cav = -cav

    cav = cav / (np.linalg.norm(cav) + 1e-12)
    return scaler, cav, clf.score(Xs, y)


def collect_class_images(model, loader, cls, max_n, device):
    model.eval()
    selected = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            mask = (labels == cls) & (preds == cls)
            if mask.any():
                selected.append(images[mask].cpu())
            if sum(x.size(0) for x in selected) >= max_n:
                break
    if not selected:
        raise RuntimeError(f"No correctly classified samples for class {cls}.")
    data = torch.cat(selected, dim=0)[:max_n]
    return data


def directional_derivatives(model, images, target_class, scaler, cav, device, batch_size=32):
    model.eval()
    dots_all = []
    for i in range(0, len(images), batch_size):
        x = images[i : i + batch_size].to(device)
        x.requires_grad_(True)
        logits, b = model.forward_with_bottleneck(x)
        b.retain_grad()

        target = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        target.backward()

        grads = b.grad.detach().cpu().numpy().reshape(b.size(0), -1)
        grads = scaler.transform(grads)
        dots = grads @ cav
        dots_all.append(dots)

    dots_all = np.concatenate(dots_all, axis=0)
    score = float(np.mean(dots_all > 0.0))
    return score, dots_all


def save_concept_figure(concept_images, random_images, save_path):
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i in range(8):
        axes[0, i].imshow(concept_images[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(random_images[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_title("X concept")
    axes[1, 0].set_title("Random")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_score_figure(scores, save_path):
    labels = ["Class 8", "Class 3"]
    means = [float(np.mean(scores[8])), float(np.mean(scores[3]))]
    stds = [float(np.std(scores[8], ddof=1)), float(np.std(scores[3], ddof=1))]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, means, yerr=stds, capsize=6, color=["#1f77b4", "#ff7f0e"])
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("TCAV score")
    ax.set_title('TCAV for concept "X-shape"')
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_dot_hist_figure(dots8, dots3, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dots8, bins=25, alpha=0.6, label="Class 8", color="#1f77b4")
    ax.hist(dots3, bins=25, alpha=0.6, label="Class 3", color="#ff7f0e")
    ax.axvline(0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Directional derivative (grad · CAV)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Directional derivative distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def ci95(values):
    arr = np.array(values)
    m = float(arr.mean())
    if len(arr) <= 1:
        return m, m
    s = float(arr.std(ddof=1))
    half = 1.96 * s / math.sqrt(len(arr))
    return m - half, m + half


def build_report(metrics, paths):
    lines = []
    lines.append("# TCAV Report: X-shape concept for predicting digit 8")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Dataset: MNIST")
    lines.append("- Model: 2-layer CNN")
    lines.append(f"- Test accuracy: {metrics['test_acc']:.4f}")
    lines.append(f"- CAV runs: {metrics['runs']}")
    lines.append(f"- Concept set size: {metrics['concept_size']}")
    lines.append(f"- Eval samples per class: {metrics['eval_size']}")
    lines.append("")
    lines.append("## TCAV results")
    lines.append(f"- Class 8 score: mean={metrics['score8_mean']:.4f}, std={metrics['score8_std']:.4f}, 95% CI=[{metrics['score8_ci'][0]:.4f}, {metrics['score8_ci'][1]:.4f}]")
    lines.append(f"- Class 3 score: mean={metrics['score3_mean']:.4f}, std={metrics['score3_std']:.4f}, 95% CI=[{metrics['score3_ci'][0]:.4f}, {metrics['score3_ci'][1]:.4f}]")
    lines.append(f"- Score gap (8 - 3): {metrics['score_gap']:.4f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- A TCAV score above 0.5 indicates the concept tends to increase the class logit.")
    lines.append("- If class 8 score is consistently higher than class 3 score, the model relies more on the X-shape concept for class 8.")
    lines.append("- This analysis is concept-quality dependent; generated concept images may not cover all visual variants.")
    lines.append("")
    lines.append("## Figures")
    lines.append(f"- Concept examples: {paths['concept_fig']}")
    lines.append(f"- TCAV scores: {paths['score_fig']}")
    lines.append(f"- Directional derivatives: {paths['hist_fig']}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train MNIST CNN and run TCAV for X-shape concept")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--concept-size", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Fast smoke run")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.concept_size = 120
        args.eval_size = 120
        args.runs = 4

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path("outputs/run_tcav")
    model_dir = out_dir / "models"
    fig_dir = out_dir / "figures"
    report_dir = out_dir / "report"
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    train_model(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr)
    test_acc = evaluate_accuracy(model, test_loader, device)

    model_path = model_dir / "mnist_cnn.pt"
    torch.save(model.state_dict(), model_path)

    eval_imgs_8 = collect_class_images(model, test_loader, cls=8, max_n=args.eval_size, device=device)
    eval_imgs_3 = collect_class_images(model, test_loader, cls=3, max_n=args.eval_size, device=device)

    scores = {8: [], 3: []}
    last_dots = {8: None, 3: None}
    concept_preview = make_concept_batch(8, generate_x_image, seed=args.seed)
    random_preview = make_concept_batch(8, generate_non_x_image, seed=args.seed + 1)

    for run in range(args.runs):
        concept_imgs = make_concept_batch(args.concept_size, generate_x_image, seed=args.seed + 100 * run)
        random_imgs = make_concept_batch(args.concept_size, generate_non_x_image, seed=args.seed + 100 * run + 1)

        concept_acts = extract_activations(model, concept_imgs, device)
        random_acts = extract_activations(model, random_imgs, device)

        scaler, cav, cav_train_acc = train_cav(concept_acts, random_acts, seed=args.seed + run)

        score8, dots8 = directional_derivatives(model, eval_imgs_8, target_class=8, scaler=scaler, cav=cav, device=device)
        score3, dots3 = directional_derivatives(model, eval_imgs_3, target_class=3, scaler=scaler, cav=cav, device=device)

        scores[8].append(score8)
        scores[3].append(score3)
        last_dots[8] = dots8
        last_dots[3] = dots3

        print(
            f"Run {run + 1}/{args.runs}: CAV train acc={cav_train_acc:.3f}, "
            f"TCAV8={score8:.3f}, TCAV3={score3:.3f}"
        )

    concept_fig = fig_dir / "concept_samples.png"
    score_fig = fig_dir / "tcav_scores.png"
    hist_fig = fig_dir / "directional_derivative_hist.png"

    save_concept_figure(concept_preview.numpy(), random_preview.numpy(), concept_fig)
    save_score_figure(scores, score_fig)
    save_dot_hist_figure(last_dots[8], last_dots[3], hist_fig)

    score8_mean = float(np.mean(scores[8]))
    score8_std = float(np.std(scores[8], ddof=1)) if len(scores[8]) > 1 else 0.0
    score3_mean = float(np.mean(scores[3]))
    score3_std = float(np.std(scores[3], ddof=1)) if len(scores[3]) > 1 else 0.0
    score8_ci = ci95(scores[8])
    score3_ci = ci95(scores[3])

    metrics = {
        "test_acc": test_acc,
        "runs": args.runs,
        "concept_size": args.concept_size,
        "eval_size": args.eval_size,
        "score8_mean": score8_mean,
        "score8_std": score8_std,
        "score8_ci": score8_ci,
        "score3_mean": score3_mean,
        "score3_std": score3_std,
        "score3_ci": score3_ci,
        "score_gap": score8_mean - score3_mean,
    }

    report_text = build_report(
        metrics,
        {
            "concept_fig": str(concept_fig).replace("\\", "/"),
            "score_fig": str(score_fig).replace("\\", "/"),
            "hist_fig": str(hist_fig).replace("\\", "/"),
        },
    )

    report_file = report_dir / "tcav_report.md"
    report_file.write_text(report_text, encoding="utf-8")

    summary_file = report_dir / "summary.json"
    summary_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nExperiment complete.")
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
