from __future__ import annotations

import json
from pathlib import Path
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "reports" / "task4" / "data"
FIG_DIR = PROJECT_ROOT / "reports" / "task4" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# dimensions user requested (from screenshot)
requested_dims = [2, 4, 8, 16, 32]

# find available history files matching pattern
history_files = list(DATA_DIR.glob("task4_latent_*_history.json"))
# also accept files like task4_latent_1_history.json (we'll parse numeric)
pattern = re.compile(r"task4_latent_(\d+)_history\.json")

available = {}
for p in history_files:
    m = pattern.search(p.name)
    if not m:
        continue
    d = int(m.group(1))
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    # payload expected to be a list of epoch dicts with test_loss
    if isinstance(payload, list) and len(payload) > 0:
        last = payload[-1]
        test_loss = last.get("test_loss")
        if test_loss is None:
            # try to find any test_loss fields
            for e in reversed(payload):
                if "test_loss" in e:
                    test_loss = e["test_loss"]
                    break
        if test_loss is not None:
            available[d] = float(test_loss)

# Also check sweep results csv/json if present
sweep_csv = DATA_DIR / "task4_sweep_results.csv"
if sweep_csv.exists():
    try:
        with sweep_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "latent_dim" in row and "final_test_loss" in row:
                    try:
                        d = int(row["latent_dim"])
                        available.setdefault(d, float(row["final_test_loss"]))
                    except Exception:
                        pass
    except Exception:
        pass

# prepare output for dims that are both requested and available
present_dims = [d for d in requested_dims if d in available]
missing_dims = [d for d in requested_dims if d not in available]
# if none of requested are present, fall back to all available sorted
if len(present_dims) == 0:
    dims_sorted = sorted(available.keys())
else:
    dims_sorted = sorted(present_dims)

mse_list = [available[d] for d in dims_sorted]

# Save summary json and csv
summary = {"dims": dims_sorted, "test_mse": mse_list, "missing_requested_dims": missing_dims}
(DATA_DIR / "task4_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
with (DATA_DIR / "task4_sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["latent_dim", "final_test_loss"])
    for d in dims_sorted:
        writer.writerow([d, available[d]])

# plot line chart
plt.figure(figsize=(6, 4))
plt.plot(dims_sorted, mse_list, marker="o")
plt.xlabel("latent dimension")
plt.ylabel("final test MSE")
plt.title("Task 4: Test MSE vs Latent Dimension")
plt.grid(True, alpha=0.3)
plt.xticks(dims_sorted)
plt.tight_layout()
fig_path = FIG_DIR / "task4_test_error_vs_latent_dim.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()

# print concise output
print("dims:", dims_sorted)
print("mse:", mse_list)
if missing_dims:
    print("missing_requested_dims:", missing_dims)
print("summary_saved:", DATA_DIR / "task4_sweep_summary.json")
print("csv_saved:", DATA_DIR / "task4_sweep_summary.csv")
print("figure_saved:", fig_path)

