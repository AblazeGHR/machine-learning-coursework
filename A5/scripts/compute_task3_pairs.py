from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
npz_path = PROJECT_ROOT / "reports" / "task3" / "data" / "task3_embeddings.npz"
centers_json = PROJECT_ROOT / "reports" / "task3" / "data" / "task3_class_centers.json"

if npz_path.exists():
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    class_names = data["class_names"].tolist()
else:
    # fallback: load precomputed centers
    if centers_json.exists():
        payload = json.loads(centers_json.read_text(encoding="utf-8"))
        class_names = list(payload.keys())
        centers = np.stack([np.array(payload[name]) for name in class_names], axis=0)
        # compute pairwise distances from centers
        n = centers.shape[0]
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        # ignore diagonal when searching for farthest/nearest: set diag to -inf for max search and +inf for min search
        mask = np.eye(n, dtype=bool)
        dists_for_max = dists.copy()
        dists_for_max[mask] = -np.inf
        dists_for_min = dists.copy()
        dists_for_min[mask] = np.inf
        imax = np.unravel_index(np.argmax(dists_for_max), dists_for_max.shape)
        imin = np.unravel_index(np.argmin(dists_for_min), dists_for_min.shape)
        far_pair = (class_names[imax[0]], class_names[imax[1]], float(dists[imax]))
        near_pair = (class_names[imin[0]], class_names[imin[1]], float(dists[imin]))
        print(json.dumps({"farthest_pair": far_pair, "nearest_pair": near_pair}, ensure_ascii=False))
        sys.exit(0)
    else:
        raise FileNotFoundError("Neither embeddings npz nor centers json found")

# compute class centers from embeddings and labels
class_names = [str(x) for x in class_names]
num_classes = len(class_names)
centers = np.zeros((num_classes, embeddings.shape[1]), dtype=float)
for i, name in enumerate(class_names):
    mask = (labels == i)
    if np.sum(mask) == 0:
        centers[i] = np.zeros(embeddings.shape[1])
    else:
        centers[i] = embeddings[mask].mean(axis=0)

dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
# pairwise distances
dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
# ignore diagonal when searching
mask = np.eye(num_classes, dtype=bool)
dists_for_max = dists.copy()
dists_for_max[mask] = -np.inf
dists_for_min = dists.copy()
dists_for_min[mask] = np.inf
imax = np.unravel_index(np.argmax(dists_for_max), dists_for_max.shape)
imin = np.unravel_index(np.argmin(dists_for_min), dists_for_min.shape)

farthest = (class_names[imax[0]], class_names[imax[1]], float(dists[imax]))
nearest = (class_names[imin[0]], class_names[imin[1]], float(dists[imin]))

# print as JSON for machine-readability
print(json.dumps({"farthest_pair": farthest, "nearest_pair": nearest}, ensure_ascii=False))

