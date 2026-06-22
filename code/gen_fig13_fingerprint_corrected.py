"""Generate corrected Fig. 13 for cross-task structural fingerprints.

This script does not overwrite the original Fig. 13. It fixes the task
metadata used for feature extraction and task-distance computation:

- FlappyBird: 4D state -> 8 input neurons.
- Snake: 8D state -> 16 input neurons.

It also uses edge directions consistent with the SNN convention
W[i, j] = source i -> target j when computing layer-specific features.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from itertools import permutations
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CROSS_DIR = ROOT / "code" / "experiments" / "results" / "cross_task"
OUT_DIR = ROOT / "code" / "experiments" / "results" / "fingerprint_analysis_corrected"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


TASK_META = {
    "CartPole": {
        "n_input": 8,
        "n_output": 2,
        "state_dim": 4,
        "action_dim": 2,
        "time_scale": 500,
        "control_type": "balance",
        "network_size": 128,
    },
    "MountainCar": {
        "n_input": 4,
        "n_output": 3,
        "state_dim": 2,
        "action_dim": 3,
        "time_scale": 200,
        "control_type": "propulsion",
        "network_size": 64,
    },
    "Acrobot": {
        "n_input": 12,
        "n_output": 3,
        "state_dim": 6,
        "action_dim": 3,
        "time_scale": 500,
        "control_type": "swing-up",
        "network_size": 128,
    },
    "FlappyBird": {
        "n_input": 8,
        "n_output": 2,
        "state_dim": 4,
        "action_dim": 2,
        "time_scale": 9999,
        "control_type": "avoidance",
        "network_size": 64,
    },
    "Snake": {
        "n_input": 16,
        "n_output": 4,
        "state_dim": 8,
        "action_dim": 4,
        "time_scale": 9999,
        "control_type": "navigation",
        "network_size": 80,
    },
}

TASK_ORDER = ["CartPole", "MountainCar", "Acrobot", "FlappyBird", "Snake"]
PLOT_ORDER = ["Acrobot", "CartPole", "FlappyBird", "MountainCar", "Snake"]
TASK_COLORS = {
    "CartPole": "#E74C3C",
    "MountainCar": "#3498DB",
    "Acrobot": "#2ECC71",
    "FlappyBird": "#F39C12",
    "Snake": "#9B59B6",
}
CONTROL_TYPE_ENCODE = {
    "balance": 0,
    "propulsion": 1,
    "swing-up": 2,
    "avoidance": 3,
    "navigation": 4,
}


def directed_clustering_coeff(adj: np.ndarray) -> float:
    n = adj.shape[0]
    adj_bin = (adj > 0).astype(int)
    total = 0.0
    count = 0
    for i in range(n):
        neighbors = np.where(adj_bin[i] | adj_bin[:, i])[0]
        if len(neighbors) < 2:
            continue
        for j in neighbors:
            for k in neighbors:
                if j >= k:
                    continue
                tri = (
                    adj_bin[i, j]
                    + adj_bin[j, i]
                    + adj_bin[i, k]
                    + adj_bin[k, i]
                    + adj_bin[j, k]
                    + adj_bin[k, j]
                )
                if tri >= 3:
                    total += 1.0
        count += len(neighbors) * (len(neighbors) - 1) // 2
    return float(total / max(count, 1))


def characteristic_path_length(adj: np.ndarray) -> float:
    n = adj.shape[0]
    adj_bin = (adj > 0).astype(int)
    total_len = 0
    total_pairs = 0
    for src in range(n):
        dist = np.full(n, -1)
        dist[src] = 0
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in np.where(adj_bin[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        valid = dist[dist > 0]
        if len(valid) > 0:
            total_len += valid.sum()
            total_pairs += len(valid)
    return float(total_len / max(total_pairs, 1))


def modularity_ei(adj: np.ndarray, types: np.ndarray) -> float:
    n = adj.shape[0]
    m = adj.sum()
    if m == 0:
        return 0.0
    communities = (types == -1).astype(int)
    q = 0.0
    for i in range(n):
        for j in range(n):
            if communities[i] == communities[j]:
                k_i_out = adj[i].sum()
                k_j_in = adj[:, j].sum()
                q += adj[i, j] - k_i_out * k_j_in / m
    return float(q / m)


def degree_entropy(adj: np.ndarray) -> float:
    degrees = adj.sum(axis=0) + adj.sum(axis=1)
    total = degrees.sum()
    if total <= 0:
        return 0.0
    probs = degrees / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def assortativity(adj: np.ndarray, types: np.ndarray) -> float:
    same_type_edges = 0
    total_edges = 0
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                total_edges += 1
                same_type_edges += int(types[i] == types[j])
    if total_edges == 0:
        return 0.0
    p_same = same_type_edges / total_edges
    p_e = (types == 1).sum() / n
    p_i = 1 - p_e
    p_random = p_e * p_e + p_i * p_i
    return float((p_same - p_random) / max(1 - p_random, 1e-6))


def information_transfer_efficiency(adj: np.ndarray) -> float:
    n = adj.shape[0]
    adj_bin = (adj > 0).astype(int)
    eff = 0.0
    count = 0
    for src in range(n):
        dist = np.full(n, -1)
        dist[src] = 0
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in np.where(adj_bin[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d > 0:
                eff += 1.0 / d
                count += 1
    return float(eff / max(count, 1))


def load_networks() -> list[dict]:
    networks: list[dict] = []
    for path in sorted(CROSS_DIR.glob("*.npz")):
        data = np.load(path)
        task = str(data["task"])
        if task not in TASK_META:
            continue
        meta = TASK_META[task]
        networks.append(
            {
                "name": path.stem,
                "task": task,
                "seed": int(data["seed"]),
                "adj": data["adj"],
                "types": data["types"],
                "fitness": float(data["best_fitness"]),
                "n_input": meta["n_input"],
                "n_output": meta["n_output"],
            }
        )
    return networks


def extract_features(net: dict) -> dict[str, float]:
    adj = net["adj"]
    types = net["types"]
    n = adj.shape[0]
    n_in = int(net["n_input"])
    n_out = int(net["n_output"])
    hidden = np.arange(n_in, n - n_out)
    input_idx = np.arange(n_in)
    output_idx = np.arange(n - n_out, n)

    exc = types == 1
    inh = types == -1
    total_conn = adj.sum()

    f: dict[str, float] = {}
    f["density"] = float(total_conn / (n * (n - 1))) if n > 1 else 0.0
    f["clustering"] = directed_clustering_coeff(adj)
    f["path_length"] = characteristic_path_length(adj) / n
    f["modularity_ei"] = modularity_ei(adj, types)
    f["degree_entropy"] = degree_entropy(adj) / np.log2(n) if n > 1 else 0.0
    f["assortativity"] = assortativity(adj, types)

    f["ei_ratio"] = float(exc.sum() / n)
    f["ee_ratio"] = float(adj[exc][:, exc].sum() / max(total_conn, 1))
    f["ei_ratio_conn"] = float(adj[exc][:, inh].sum() / max(total_conn, 1))
    f["ie_ratio"] = float(adj[inh][:, exc].sum() / max(total_conn, 1))
    f["ii_ratio"] = float(adj[inh][:, inh].sum() / max(total_conn, 1))
    f["exc_indeg"] = float(adj[:, exc].sum(axis=0).mean() / max(n, 1))
    f["inh_indeg"] = float(adj[:, inh].sum(axis=0).mean() / max(n, 1))

    if len(hidden) > 0:
        # Rows are sources and columns are targets under W[i,j].
        f["in_to_hidden"] = float(
            adj[np.ix_(input_idx, hidden)].sum() / max(len(input_idx) * len(hidden), 1)
        )
        f["hidden_to_out"] = float(
            adj[np.ix_(hidden, output_idx)].sum() / max(len(hidden) * len(output_idx), 1)
        )
        f["hidden_internal"] = float(
            adj[np.ix_(hidden, hidden)].sum() / max(len(hidden) ** 2, 1)
        )
    else:
        f["in_to_hidden"] = 0.0
        f["hidden_to_out"] = 0.0
        f["hidden_internal"] = 0.0

    f["info_efficiency"] = information_transfer_efficiency(adj)
    f["inv_path_len"] = 1.0 / max(f["path_length"] * n, 1e-6)
    f["sparsity"] = 1.0 - f["density"]
    return f


def build_feature_matrix(networks: list[dict]):
    feature_dicts = [extract_features(n) for n in networks]
    feature_names = sorted(feature_dicts[0].keys())
    x = np.array([[fd[k] for k in feature_names] for fd in feature_dicts])
    y = np.array([n["task"] for n in networks])
    return x, y, feature_names


def loo_knn_predictions(x: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    preds = []
    for i in range(len(x)):
        train_x = np.delete(x, i, axis=0)
        train_y = np.delete(y, i, axis=0)
        dists = np.linalg.norm(train_x - x[i], axis=1)
        nearest = np.argsort(dists)[:k]
        votes = Counter(train_y[nearest])
        preds.append(votes.most_common(1)[0][0])
    return np.array(preds)


def task_distance_matrix(tasks: list[str]) -> np.ndarray:
    numeric = ["state_dim", "action_dim", "time_scale", "network_size"]
    vecs = []
    for task in tasks:
        meta = TASK_META[task]
        vec = [meta[k] for k in numeric]
        vec.append(CONTROL_TYPE_ENCODE[meta["control_type"]])
        vecs.append(vec)
    arr = np.array(vecs, dtype=float)
    arr = (arr - arr.mean(axis=0)) / np.where(arr.std(axis=0) == 0, 1.0, arr.std(axis=0))
    return squareform(pdist(arr, metric="euclidean"))


def mantel_exact(struct_dist: np.ndarray, task_dist: np.ndarray) -> tuple[float, float]:
    triu = np.triu_indices_from(struct_dist, k=1)
    s = struct_dist[triu]
    t_obs = task_dist[triu]
    r_obs, _ = pearsonr(s, t_obs)
    count = 0
    total = 0
    n = struct_dist.shape[0]
    for perm in permutations(range(n)):
        p = np.array(perm)
        t_perm = task_dist[p][:, p][triu]
        r_perm, _ = pearsonr(s, t_perm)
        count += int(abs(r_perm) >= abs(r_obs) - 1e-12)
        total += 1
    return float(r_obs), float(count / total)


def row_normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray, tasks: list[str]):
    cm = np.zeros((len(tasks), len(tasks)), dtype=int)
    idx = {t: i for i, t in enumerate(tasks)}
    for yt, yp in zip(y_true, y_pred):
        cm[idx[yt], idx[yp]] += 1
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return cm, cm_norm


def plot_figure(
    x_std: np.ndarray,
    y: np.ndarray,
    networks: list[dict],
    feature_names: list[str],
    importances: np.ndarray,
    struct_dist: np.ndarray,
    task_dist: np.ndarray,
    mantel_r: float,
    mantel_p: float,
    y_pred: np.ndarray,
    acc: float,
) -> None:
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init="pca")
    x_tsne = tsne.fit_transform(x_std)

    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    for task in TASK_ORDER:
        mask = y == task
        ax_a.scatter(
            x_tsne[mask, 0],
            x_tsne[mask, 1],
            s=85,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            color=TASK_COLORS[task],
            label=task,
        )
        for i in np.where(mask)[0]:
            ax_a.annotate(str(networks[i]["seed"]), (x_tsne[i, 0], x_tsne[i, 1]), fontsize=7, alpha=0.55)
    ax_a.set_title("(a) Cross-Task Structural Fingerprint (t-SNE)", fontsize=15, fontweight="bold")
    ax_a.set_xlabel("t-SNE 1")
    ax_a.set_ylabel("t-SNE 2")
    ax_a.grid(True, alpha=0.2)
    ax_a.legend(title="Task", frameon=True)

    ax_b_outer = fig.add_subplot(gs[0, 1])
    ax_b_outer.axis("off")
    pos = ax_b_outer.get_position()
    ax_b1 = fig.add_axes([pos.x0, pos.y0 + 0.11, pos.width * 0.46, pos.height * 0.62])
    ax_b2 = fig.add_axes([pos.x0 + pos.width * 0.54, pos.y0 + 0.11, pos.width * 0.46, pos.height * 0.62])
    for ax, mat, title in [
        (ax_b1, struct_dist, "Structural Distance"),
        (ax_b2, task_dist, "Task Metadata Distance"),
    ]:
        im = ax.imshow(mat, cmap="YlOrRd")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(PLOT_ORDER)))
        ax.set_yticks(range(len(PLOT_ORDER)))
        ax.set_xticklabels(PLOT_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(PLOT_ORDER, fontsize=8)
        for i in range(len(PLOT_ORDER)):
            for j in range(len(PLOT_ORDER)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax_b_outer.text(
        0.5,
        0.03,
        f"Mantel test: r = {mantel_r:.3f}, p = {mantel_p:.3f}",
        transform=ax_b_outer.transAxes,
        ha="center",
        fontsize=11,
        style="italic",
    )
    ax_b_outer.text(0.0, 0.96, "(b)", transform=ax_b_outer.transAxes, fontsize=15, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 0])
    order = np.argsort(importances)[::-1][:10]
    labels = [feature_names[i].replace("_", " ").title() for i in order][::-1]
    vals = importances[order][::-1]
    ax_c.barh(labels, vals, color="#5DADE2")
    for i, v in enumerate(vals):
        ax_c.text(v + max(vals) * 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax_c.set_xlabel("Importance (Mean Decrease in Impurity)")
    ax_c.set_title("(c) Feature Importance (Random Forest)", fontsize=15, fontweight="bold")
    ax_c.grid(axis="x", alpha=0.2)

    ax_d = fig.add_subplot(gs[1, 1])
    cm, cm_norm = row_normalized_confusion(y, y_pred, PLOT_ORDER)
    im = ax_d.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax_d.set_xticks(range(len(PLOT_ORDER)))
    ax_d.set_yticks(range(len(PLOT_ORDER)))
    ax_d.set_xticklabels(PLOT_ORDER, rotation=45, ha="right")
    ax_d.set_yticklabels(PLOT_ORDER)
    ax_d.set_xlabel("Predicted Task")
    ax_d.set_ylabel("True Task")
    ax_d.set_title(f"(d) LOO-KNN Confusion Matrix ({acc:.1%})", fontsize=15, fontweight="bold")
    for i in range(len(PLOT_ORDER)):
        for j in range(len(PLOT_ORDER)):
            txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax_d.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=9,
            )
    fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04, label="Row-Normalized Proportion")

    fig.suptitle("Cross-Task Structural Fingerprint Analysis", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        for stem in ("fig13_fingerprint", "fig13_fingerprint_corrected"):
            out = FIG_DIR / f"{stem}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"Saved: {out}")
    plt.close(fig)


def main() -> None:
    networks = load_networks()
    if len(networks) != 50:
        raise RuntimeError(f"Expected 50 networks, got {len(networks)}")

    counts = Counter(n["task"] for n in networks)
    print("Loaded networks:", dict(sorted(counts.items())))

    x_raw, y, feature_names = build_feature_matrix(networks)
    x_std = StandardScaler().fit_transform(x_raw)
    y_pred = loo_knn_predictions(x_std, y, k=3)
    acc = float(np.mean(y_pred == y))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(x_std, y)
    importances = rf.feature_importances_

    tasks = PLOT_ORDER
    task_fingerprints = np.array([x_std[y == task].mean(axis=0) for task in tasks])
    struct_dist = squareform(pdist(task_fingerprints, metric="euclidean"))
    task_dist = task_distance_matrix(tasks)
    mantel_r, mantel_p = mantel_exact(struct_dist, task_dist)

    plot_figure(
        x_std,
        y,
        networks,
        feature_names,
        importances,
        struct_dist,
        task_dist,
        mantel_r,
        mantel_p,
        y_pred,
        acc,
    )

    cm, cm_norm = row_normalized_confusion(y, y_pred, tasks)
    top_features = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in np.argsort(importances)[::-1]
    ]
    report = {
        "corrections": {
            "FlappyBird": {"state_dim": 4, "n_input": 8},
            "Snake": {"state_dim": 8, "n_input": 16},
            "layer_edges": "rows are sources and columns are targets under W[i,j]",
        },
        "n_networks": len(networks),
        "task_counts": dict(sorted(counts.items())),
        "feature_count": len(feature_names),
        "loo_knn_k": 3,
        "loo_knn_accuracy": acc,
        "random_baseline": 1 / len(tasks),
        "mantel": {"r": mantel_r, "p_exact_label_permutation": mantel_p},
        "tasks_order": tasks,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": cm_norm.tolist(),
        "top_features": top_features,
        "feature_names": feature_names,
    }
    with (OUT_DIR / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with (OUT_DIR / "report.txt").open("w", encoding="utf-8") as f:
        f.write("Corrected cross-task structural fingerprint analysis\n")
        f.write("=" * 58 + "\n")
        f.write("Corrections: FlappyBird n_input=8/state_dim=4; Snake n_input=16/state_dim=8.\n")
        f.write("Layer features use W[i,j] as source i -> target j.\n\n")
        f.write(f"Networks: {len(networks)}\n")
        f.write(f"Feature count: {len(feature_names)}\n")
        f.write(f"LOO-KNN accuracy: {acc:.3f} ({acc/(1/len(tasks)):.2f}x random)\n")
        f.write(f"Mantel exact label-permutation: r={mantel_r:.3f}, p={mantel_p:.3f}\n\n")
        f.write("Confusion matrix rows=true, cols=predicted; task order: " + ", ".join(tasks) + "\n")
        f.write(str(cm) + "\n\n")
        f.write("Top 10 features:\n")
        for row in top_features[:10]:
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    print(f"Saved: {OUT_DIR / 'report.json'}")
    print(f"Saved: {OUT_DIR / 'report.txt'}")
    print(f"LOO-KNN accuracy: {acc:.3f}")
    print(f"Mantel exact: r={mantel_r:.3f}, p={mantel_p:.3f}")
    print("Top 5 features:")
    for row in top_features[:5]:
        print(f"  {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
