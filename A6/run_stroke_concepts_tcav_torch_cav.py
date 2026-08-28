import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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


def draw_circle(img, cx, cy, r, value=1.0, thickness=1):
    for t in np.linspace(0.0, 2.0 * math.pi, 220):
        x = int(round(cx + r * math.cos(t)))
        y = int(round(cy + r * math.sin(t)))
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < img.shape[1] and 0 <= yy < img.shape[0]:
                    img[yy, xx] = max(img[yy, xx], value)


def base_canvas(size=28, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0.02, 0.02, (size, size)).clip(0.0, 0.2)


def concept_x(size=28, rng=None):
    img = base_canvas(size=size, rng=rng)
    m = int(rng.integers(5, 8))
    draw_line(img, m, m, size - m - 1, size - m - 1, thickness=1)
    draw_line(img, size - m - 1, m, m, size - m - 1, thickness=1)
    return img.clip(0.0, 1.0).astype(np.float32)


def concept_vertical(size=28, rng=None):
    img = base_canvas(size=size, rng=rng)
    x = int(rng.integers(11, 17))
    m = int(rng.integers(4, 7))
    draw_line(img, x, m, x, size - m - 1, thickness=1)
    return img.clip(0.0, 1.0).astype(np.float32)


def concept_horizontal(size=28, rng=None):
    img = base_canvas(size=size, rng=rng)
    y = int(rng.integers(11, 17))
    m = int(rng.integers(4, 7))
    draw_line(img, m, y, size - m - 1, y, thickness=1)
    return img.clip(0.0, 1.0).astype(np.float32)


def concept_slash(size=28, rng=None):
    img = base_canvas(size=size, rng=rng)
    m = int(rng.integers(4, 8))
    draw_line(img, m, size - m - 1, size - m - 1, m, thickness=1)
    return img.clip(0.0, 1.0).astype(np.float32)


def concept_backslash(size=28, rng=None):
    img = base_canvas(size=size, rng=rng)
    m = int(rng.integers(4, 8))
    draw_line(img, m, m, size - m - 1, size - m - 1, thickness=1)
    return img.clip(0.0, 1.0).astype(np.float32)


def concept_loop(size=28, rng=None):
    img = base_canvas(size=size, rng=rng)
    cx = int(rng.integers(11, 17))
    cy = int(rng.integers(11, 17))
    r = int(rng.integers(6, 9))
    draw_circle(img, cx, cy, r, thickness=1)
    return img.clip(0.0, 1.0).astype(np.float32)


def random_non_concept(size=28, rng=None):
    img = base_canvas(size=size, rng=rng)
    mode = int(rng.integers(0, 3))
    m = int(rng.integers(4, 8))
    if mode == 0:
        x = int(rng.integers(m, size - m))
        draw_line(img, x, m, x, size - m - 1, thickness=1)
    elif mode == 1:
        y = int(rng.integers(m, size - m))
        draw_line(img, m, y, size - m - 1, y, thickness=1)
    else:
        x0 = int(rng.integers(m, size // 2))
        y0 = int(rng.integers(m, size - m))
        x1 = int(rng.integers(size // 2, size - m))
        y1 = int(rng.integers(m, size - m))
        draw_line(img, x0, y0, x1, y1, thickness=1)
    return img.clip(0.0, 1.0).astype(np.float32)


def make_concept_batch(n, generator, seed):
    rng = np.random.default_rng(seed)
    arr = np.stack([generator(rng=rng) for _ in range(n)], axis=0)
    return torch.from_numpy(arr).unsqueeze(1)


def extract_activations(model, images, device, batch_size=128):
    model.eval()
    acts = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = images[i : i + batch_size].to(device)
            _, b = model.forward_with_bottleneck(x)
            acts.append(b.reshape(b.size(0), -1).cpu().numpy())
    return np.concatenate(acts, axis=0)


def train_cav_torch(concept_acts, random_acts, seed, device, epochs=40, lr=1e-2, batch_size=256):
    # inputs are numpy arrays (N, D)
    X = np.vstack([concept_acts, random_acts]).astype(np.float32)
    y = np.array([1] * len(concept_acts) + [0] * len(random_acts), dtype=np.float32)

    # standardize in numpy
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    Xs = (X - mean) / std

    # torch dataset
    tx = torch.from_numpy(Xs)
    ty = torch.from_numpy(y).unsqueeze(1)
    dataset = TensorDataset(tx, ty)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    D = Xs.shape[1]
    torch.manual_seed(seed)
    model = nn.Linear(D, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # train
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    # get weight vector
    w = model.weight.detach().cpu().numpy().ravel()

    # alignment check using scaled projections
    concept_proj = Xs[: len(concept_acts)] @ w
    random_proj = Xs[len(concept_acts) :] @ w
    if concept_proj.mean() < random_proj.mean():
        w = -w

    # normalize
    w = w / (np.linalg.norm(w) + 1e-12)

    # approximate train acc
    logits = tx.numpy() @ w
    preds = (logits > 0).astype(int)
    acc = float((preds == y).mean())

    scaler = (mean.ravel(), std.ravel())
    return scaler, w, acc


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
    return torch.cat(selected, dim=0)[:max_n]


def directional_derivatives(model, images, target_class, scaler, cav, device, batch_size=64):
    model.eval()
    mean, std = scaler
    dots_all = []
    for i in range(0, len(images), batch_size):
        x = images[i : i + batch_size].to(device)
        x.requires_grad_(True)
        logits, b = model.forward_with_bottleneck(x)
        b.retain_grad()

        objective = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        objective.backward()

        grads = b.grad.detach().cpu().numpy().reshape(b.size(0), -1)
        grads = (grads - mean) / std
        dots_all.append(grads @ cav)

    dots = np.concatenate(dots_all, axis=0)
    score = float(np.mean(dots > 0.0))
    return score


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def one_sided_pvals_against_half(scores):
    arr = np.array(scores, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0

    if std < 1e-12:
        if mean > 0.5:
            return mean, std, 0.0, 1.0
        if mean < 0.5:
            return mean, std, 1.0, 0.0
        return mean, std, 1.0, 1.0

    z = (mean - 0.5) / (std / math.sqrt(n))
    p_gt = max(0.0, min(1.0, 1.0 - normal_cdf(z)))
    p_lt = max(0.0, min(1.0, normal_cdf(z)))
    return mean, std, p_gt, p_lt


def save_bar_figure(results, save_path):
    concepts = [r["concept"] for r in results]
    means = [r["score8_mean"] for r in results]
    stds = [r["score8_std"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(concepts, means, yerr=stds, capsize=5, color="#4c78a8")
    ax.axhline(0.5, linestyle="--", linewidth=1, color="black")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("TCAV score for class 8")
    ax.set_title("Concept dependence test on class 8")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_gap_figure(results, save_path):
    concepts = [r["concept"] for r in results]
    gaps = [r["gap_8_minus_3"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(concepts, gaps, color="#f58518")
    ax.axhline(0.0, linestyle="--", linewidth=1, color="black")
    ax.set_ylabel("TCAV gap (class8 - class3)")
    ax.set_title("Class-specificity of candidate concepts")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def build_report(results, test_acc, args, bar_fig, gap_fig):
    lines = []
    lines.append("# Stroke Concept Significance Report (MNIST + TCAV, PyTorch CAV)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Test accuracy: {test_acc:.4f}")
    lines.append(f"- CAV runs per concept: {args.runs}")
    lines.append(f"- Concept size: {args.concept_size}")
    lines.append(f"- Eval size per class: {args.eval_size}")
    lines.append("")
    lines.append("## Significance rule")
    lines.append("- Positive dependency: one-sided p(mean > 0.5) < 0.05 and mean > 0.5")
    lines.append("- Negative dependency: one-sided p(mean < 0.5) < 0.05 and mean < 0.5")
    lines.append("")
    lines.append("## Results by concept")

    for r in results:
        pos_tag = "YES" if r["positive_dependency"] else "NO"
        neg_tag = "YES" if r["negative_dependency"] else "NO"
        lines.append(
            "- "
            + f"{r['concept']}: mean8={r['score8_mean']:.4f}, std8={r['score8_std']:.4f}, "
            + f"mean3={r['score3_mean']:.4f}, gap={r['gap_8_minus_3']:.4f}, "
            + f"p_gt_0.5={r['p_value_gt_0_5']:.4g}, p_lt_0.5={r['p_value_lt_0_5']:.4g}, "
            + f"positive={pos_tag}, negative={neg_tag}"
        )

    lines.append("")
    lines.append("## Figures")
    lines.append(f"- {bar_fig}")
    lines.append(f"- {gap_fig}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Test multiple stroke concepts for significant TCAV (PyTorch CAV)")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--concept-size", type=int, default=320)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="outputs/run_tcav/models/mnist_cnn.pt")
    parser.add_argument("--cav-epochs", type=int, default=40)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("outputs/stroke_concepts_pytorch")
    fig_dir = out_dir / "figures"
    report_dir = out_dir / "report"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    model_file = Path(args.model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Please run run_tcav.py first to create outputs/models/mnist_cnn.pt"
        )
    state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)

    test_acc = evaluate_accuracy(model, test_loader, device)

    eval_imgs_8 = collect_class_images(model, test_loader, cls=8, max_n=args.eval_size, device=device)
    eval_imgs_3 = collect_class_images(model, test_loader, cls=3, max_n=args.eval_size, device=device)

    concept_generators = {
        "x_shape": concept_x,
        "vertical": concept_vertical,
        "horizontal": concept_horizontal,
        "slash": concept_slash,
        "backslash": concept_backslash,
        "loop": concept_loop,
    }

    results = []
    for idx, (name, gen) in enumerate(concept_generators.items()):
        scores8 = []
        scores3 = []
        cav_accs = []

        for run in tqdm(range(args.runs), desc=f"Concept {name}"):
            seed_base = args.seed + idx * 10000 + run * 100
            concept_imgs = make_concept_batch(args.concept_size, gen, seed=seed_base)
            random_imgs = make_concept_batch(args.concept_size, random_non_concept, seed=seed_base + 1)

            concept_acts = extract_activations(model, concept_imgs, device)
            random_acts = extract_activations(model, random_imgs, device)
            scaler, cav, cav_acc = train_cav_torch(
                concept_acts, random_acts, seed=seed_base + 2, device=device, epochs=args.cav_epochs
            )

            score8 = directional_derivatives(model, eval_imgs_8, target_class=8, scaler=scaler, cav=cav, device=device)
            score3 = directional_derivatives(model, eval_imgs_3, target_class=3, scaler=scaler, cav=cav, device=device)

            scores8.append(score8)
            scores3.append(score3)
            cav_accs.append(cav_acc)

        m8, s8, p_gt, p_lt = one_sided_pvals_against_half(scores8)
        m3 = float(np.mean(scores3))
        s3 = float(np.std(scores3, ddof=1)) if len(scores3) > 1 else 0.0

        results.append(
            {
                "concept": name,
                "cav_train_acc_mean": float(np.mean(cav_accs)),
                "score8_mean": m8,
                "score8_std": s8,
                "score3_mean": m3,
                "score3_std": s3,
                "gap_8_minus_3": m8 - m3,
                "p_value_gt_0_5": p_gt,
                "p_value_lt_0_5": p_lt,
                "positive_dependency": bool((m8 > 0.5) and (p_gt < 0.05)),
                "negative_dependency": bool((m8 < 0.5) and (p_lt < 0.05)),
            }
        )

        print(f"{name}: mean8={m8:.3f}, mean3={m3:.3f}, p_gt_0.5={p_gt:.3g}, p_lt_0.5={p_lt:.3g}")

    results.sort(key=lambda x: x["score8_mean"], reverse=True)

    bar_fig = fig_dir / "stroke_concepts_tcav8_pytorch.png"
    gap_fig = fig_dir / "stroke_concepts_gap_8_minus_3_pytorch.png"
    save_bar_figure(results, bar_fig)
    save_gap_figure(results, gap_fig)

    summary_path = report_dir / "stroke_concepts_summary_pytorch.json"
    summary_path.write_text(
        json.dumps(
            {
                "test_acc": test_acc,
                "runs": args.runs,
                "concept_size": args.concept_size,
                "eval_size": args.eval_size,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_text = build_report(
        results,
        test_acc,
        args,
        str(bar_fig).replace("\\", "/"),
        str(gap_fig).replace("\\", "/"),
    )
    report_path = report_dir / "stroke_concepts_report_pytorch.md"
    report_path.write_text(report_text, encoding="utf-8")

    print("Done.")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
