"""Generate Fig. 11 boundary-scan panels from processed CSV files."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
BOUNDARY_DIR = ROOT / "code" / "experiments" / "sec4_adaptation" / "results" / "boundary_scan"
CARTPOLE_CSV = BOUNDARY_DIR / "cartpole_boundary.csv"
FLAPPY_CSV = BOUNDARY_DIR / "flappy_boundary.csv"


def zone_color(value: float) -> str:
    if value >= 490:
        return "#2ca02c"
    if value >= 200:
        return "#ff7f0e"
    return "#d62728"


def build_cartpole_matrix(cp: pd.DataFrame) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
    gravities = sorted(cp["gravity"].unique())
    lengths = sorted(cp["length"].unique())
    mean_mat = np.zeros((len(gravities), len(lengths)))
    std_mat = np.zeros((len(gravities), len(lengths)))
    for i, gravity in enumerate(reversed(gravities)):
        for j, length in enumerate(lengths):
            row = cp[(cp["gravity"] == gravity) & (cp["length"] == length)]
            mean_mat[i, j] = row["mean"].values[0]
            std_mat[i, j] = row["std"].values[0]
    return gravities, lengths, mean_mat, std_mat


def draw_cartpole(ax: plt.Axes, cp: pd.DataFrame, title: str) -> None:
    gravities, lengths, mean_mat, std_mat = build_cartpole_matrix(cp)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=100, vmax=500)
    image = ax.imshow(mean_mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for i in range(len(gravities)):
        for j in range(len(lengths)):
            value = mean_mat[i, j]
            sd = std_mat[i, j]
            text_color = "white" if value < 250 or value > 420 else "black"
            ax.text(j, i, f"{value:.0f}\n+/-{sd:.0f}", ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=3, edgecolor=zone_color(value), facecolor="none", zorder=5))

    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([f"{length}" for length in lengths], fontsize=11)
    ax.set_yticks(range(len(gravities)))
    ax.set_yticklabels([f"{gravity}" for gravity in reversed(gravities)], fontsize=11)
    ax.set_xlabel("Pole length (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gravity (m/s^2)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean steps survived", fontsize=11)
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="k", label="Comfort (>=490)"),
        Patch(facecolor="#ff7f0e", edgecolor="k", label="Boundary (200-489)"),
        Patch(facecolor="#d62728", edgecolor="k", label="Failure (<200)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9, edgecolor="gray")


def draw_flappy(ax: plt.Axes, flappy: pd.DataFrame, title: str) -> None:
    speeds = flappy["pipe_speed"].values
    means = flappy["mean"].values
    stds = flappy["std"].values

    ax.axhspan(40, 60, color="#2ca02c", alpha=0.10)
    ax.axhspan(15, 40, color="#ff7f0e", alpha=0.10)
    ax.axhspan(0, 15, color="#d62728", alpha=0.10)
    ax.axhline(y=40, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(y=15, color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(speeds[-1] + 0.3, 50, "Comfort", fontsize=9, color="#2ca02c", fontweight="bold", va="center")
    ax.text(speeds[-1] + 0.3, 27.5, "Boundary", fontsize=9, color="#ff7f0e", fontweight="bold", va="center")
    ax.text(speeds[-1] + 0.3, 7, "Failure", fontsize=9, color="#d62728", fontweight="bold", va="center")
    ax.errorbar(speeds, means, yerr=stds, fmt="o-", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, markerfacecolor="#1f77b4", markeredgecolor="white", markeredgewidth=1.5, linewidth=2.0, zorder=10)
    for speed, mean, sd in zip(speeds, means, stds):
        offset = 8 if mean > 20 else -12
        ax.annotate(f"{mean:.1f}+/-{sd:.1f}", xy=(speed, mean), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#333333")
    ax.set_xlabel("Pipe speed", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(speeds)
    ax.set_xticklabels([f"{speed:.0f}" for speed in speeds], fontsize=11)
    ax.set_ylim(0, max(means + stds) * 1.2)
    ax.set_xlim(speeds[0] - 0.5, speeds[-1] + 1.5)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    cp = pd.read_csv(CARTPOLE_CSV)
    flappy = pd.read_csv(FLAPPY_CSV)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig_a, ax_a = plt.subplots(1, 1, figsize=(7, 6))
    draw_cartpole(ax_a, cp, "CartPole parameter boundary scan")
    fig_a.tight_layout()
    fig_a.savefig(FIG_DIR / "fig11a_boundary_cartpole.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    fig_a.savefig(FIG_DIR / "fig11a_boundary_cartpole.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(1, 1, figsize=(7, 5))
    draw_flappy(ax_b, flappy, "FlappyBird pipe-speed boundary")
    fig_b.tight_layout()
    fig_b.savefig(FIG_DIR / "fig11b_boundary_flappy.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    fig_b.savefig(FIG_DIR / "fig11b_boundary_flappy.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig_b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    draw_cartpole(ax1, cp, "(a) CartPole parameter boundary scan")
    draw_flappy(ax2, flappy, "(b) FlappyBird pipe-speed boundary")
    fig.tight_layout(w_pad=3)
    fig.savefig(FIG_DIR / "fig11_boundary_scan.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(FIG_DIR / "fig11_boundary_scan.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved boundary-scan figures to {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
