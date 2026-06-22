import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT / "code" / "experiments" / "results" / "cross_task"
out_dir = ROOT / "figures_arxiv"
out_dir.mkdir(exist_ok=True)

# ── colour map ──
task_colors = {
    "CartPole":   "#E74C3C",
    "MountainCar": "#3498DB",
    "Acrobot":    "#2ECC71",
    "FlappyBird": "#F39C12",
    "Snake":      "#9B59B6",
}

tasks = list(task_colors.keys())

# ── load CartPole seed0 ──
cp = np.load(data_dir / "CartPole_seed0.npz")
best_hist  = cp["best_hist"]
mean_hist  = cp["mean_hist"]
gens       = np.arange(len(best_hist))

# ── create figure ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── (a) CartPole seed 0 ──
ax = axes[0]
ax.plot(gens, best_hist, color="#2980B9", linewidth=1.8, label="Best Fitness")
ax.plot(gens, mean_hist, color="#85C1E9", linewidth=1.2, linestyle="--",
        alpha=0.5, label="Mean Fitness")
ax.axhline(y=490, color="gray", linestyle=":", linewidth=1.2, label="Solve Threshold (490)")
ax.set_xlabel("Generation", fontsize=12)
ax.set_ylabel("Best Fitness", fontsize=12)
ax.set_title("CartPole-v1 Evolution (Seed 0)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

# ── (b) multi-task comparison ──
ax = axes[1]
for task in tasks:
    fpath = data_dir / f"{task}_seed0.npz"
    d = np.load(fpath)
    bh = d["best_hist"]
    ax.plot(np.arange(len(bh)), bh, color=task_colors[task],
            linewidth=1.6, label=task)

ax.set_xlabel("Generation", fontsize=12)
ax.set_ylabel("Best Fitness", fontsize=12)
ax.set_title("Multi-Task Evolution Comparison (Seed 0)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()

# ── save ──
for ext in ("png", "pdf"):
    out_path = out_dir / f"fig01_evolution_curve.{ext}"
    fig.savefig(out_path, dpi=300 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")

plt.close()
