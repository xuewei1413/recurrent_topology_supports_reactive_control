"""
v5 M3 模块：E→K 压缩实验（计划 §2.3.4）

验证任务所需的传播修正有效维度：
取 E 条件的 ~1400 维 edge_gains，用 k-means（按边端点位置特征）压缩到
K=1/2/4/8 个 gain 组，组内取均值 gain，不重新优化直接评估性能保留率。

若 K 组压缩即可保留大部分 E 的性能 → 有效调制维度 ≈ K，与 K* 判定互证。

产出：results_v5/gains/e_compression_{task}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol
from experiments_v2.core.scaffold import NeuronScaffold
from experiments_v2.core.encoder import ObservationEncoder
from experiments_v2.core.controller import UnifiedSNNController
from experiments_v2.core.weight_construction import E
from experiments_v2.tasks.adapter import make_env
from experiments_v2.mechanisms.run_causal_propagation_v5 import load_v3_topology

K_VALUES = [1, 2, 4, 8]
TEST_SEEDS = list(range(70000, 70100))  # 与 gains_staged 相同 test 区间


def kmeans_numpy(features, K, rng, n_iters=50, n_restarts=5):
    """纯 numpy k-means（避免 sklearn 依赖）。返回 labels。"""
    best_labels, best_inertia = None, np.inf
    n = len(features)
    for _ in range(n_restarts):
        idx = rng.choice(n, size=min(K, n), replace=False)
        centers = features[idx].copy()
        for _ in range(n_iters):
            d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            new_centers = centers.copy()
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    new_centers[k] = features[mask].mean(axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        inertia = ((features - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


def compress_e_gains(edge_gains, adjacency, K, rng):
    """按边端点位置 k-means 聚类，组内 gain 取均值。返回压缩后 edge_gains。"""
    active = np.argwhere(adjacency > 0)
    n_total = adjacency.shape[0]
    features = np.column_stack([
        active[:, 0] / n_total,   # 目标位置（归一化）
        active[:, 1] / n_total,   # 源位置
    ])
    labels = kmeans_numpy(features, K, rng)
    compressed = edge_gains.copy()
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            compressed[mask] = edge_gains[mask].mean()
    return compressed, labels


def run_e_compression(task_name, remote_root, n_topologies=20):
    task = TASK_SPECS[task_name]
    ctrl_proto = ControllerProtocol()
    encoder = ObservationEncoder.load(
        f"{remote_root}/results_v2/pilot/observation_stats/observation_stats_{task_name}.json"
    )

    out_path = Path(remote_root) / "results_v5" / "gains" / f"e_compression_{task_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {"task": task_name, "version": "v5", "k_values": K_VALUES,
                   "per_seed": []}
    done = {r["seed"] for r in results["per_seed"]}

    for seed_idx in range(min(n_topologies, 20)):
        evo_seed = 3000 + seed_idx
        if evo_seed in done:
            continue
        f0_adj = load_v3_topology(task_name, evo_seed, remote_root)
        staged_path = (f"{remote_root}/results_v3/gains_staged/{task_name}/"
                       f"gain_result_{task_name}_seed{evo_seed}.json")
        if f0_adj is None or not os.path.exists(staged_path):
            continue
        with open(staged_path) as f:
            e_rec = json.load(f).get("conditions", {}).get("E", {})
        e_gains = e_rec.get("gain_vector")
        e_test = (e_rec.get("test") or {}).get("raw_return_mean")
        if not e_gains or e_test is None:
            continue
        e_gains = np.array(e_gains, dtype=float)
        n_active = int(f0_adj.sum())
        if len(e_gains) > n_active:
            e_gains = e_gains[:n_active]
        elif len(e_gains) < n_active:
            e_gains = np.pad(e_gains, (0, n_active - len(e_gains)), constant_values=1.0)

        t0 = time.time()
        scaffold = NeuronScaffold.from_seed(seed=evo_seed, task=task, controller=ctrl_proto)
        env = make_env(task)

        # F0 参考（压缩 K→∞ 时应回到 E 本身）
        seed_rec = {"seed": evo_seed, "e_test": e_test, "compression": {}}
        for K in K_VALUES:
            rng = np.random.RandomState(evo_seed * 31 + K)
            compressed, labels = compress_e_gains(e_gains, f0_adj, K, rng)
            ctrl = UnifiedSNNController(scaffold, f0_adj, encoder, ctrl_proto,
                                        edge_gains=compressed, condition=E)
            res = ctrl.evaluate(env, TEST_SEEDS[:20], task.max_steps)
            seed_rec["compression"][str(K)] = {
                "test_mean": res["mean"],
                "retention_vs_E": (res["mean"] / e_test if e_test else None),
            }
            print(f"    [K={K}] test={res['mean']:.1f} (E={e_test:.1f})", flush=True)

        env.close()
        seed_rec["elapsed_sec"] = round(time.time() - t0, 1)
        results["per_seed"].append(seed_rec)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [v5 E-compress {task_name} seed={evo_seed}] ({seed_rec['elapsed_sec']}s)",
              flush=True)

    # 汇总保留率
    if results["per_seed"]:
        summary = {}
        for K in K_VALUES:
            rets = [r["compression"][str(K)]["test_mean"]
                    for r in results["per_seed"] if str(K) in r["compression"]]
            es = [r["e_test"] for r in results["per_seed"] if str(K) in r["compression"]]
            if rets:
                summary[str(K)] = {
                    "mean_test": float(np.mean(rets)),
                    "mean_E": float(np.mean(es)),
                    "mean_retention": float(np.mean(rets) / np.mean(es)) if np.mean(es) else None,
                    "n": len(rets),
                }
        results["summary"] = summary
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  {task_name} retention summary:", flush=True)
        for K in K_VALUES:
            s = summary.get(str(K))
            if s:
                print(f"    K={K}: {s['mean_test']:.1f} vs E {s['mean_E']:.1f} "
                      f"(retention {s['mean_retention']:.2f})", flush=True)

    print(f"  Saved {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--remote-root", type=str,
                        default="/home/lixuewei/tmp/struct_mod_v2")
    args = parser.parse_args()
    run_e_compression(args.task, args.remote_root)
