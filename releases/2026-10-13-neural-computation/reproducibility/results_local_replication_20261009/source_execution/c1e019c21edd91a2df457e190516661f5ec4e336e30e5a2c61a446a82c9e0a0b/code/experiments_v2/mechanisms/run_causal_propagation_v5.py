"""
v5 C3 核心对照：固定接口的匹配删除（替换 v4 版本，修复缺陷 F1）

所有结构干预严格固定 input→hidden 和 hidden→output 接口，
只在 HH 子图内操作，从而真正区分：
  1. 递归存在（有无 HH 边）        → F0 vs noHH
  2. 递归具体组织（HH 拓扑结构）    → F0 vs HH_rewired（同边数同度分布）
  3. 递归环的贡献                  → noHH vs HH_DAG
  4. 剂量-响应关系                 → HH_dose_25/50/75/100(=noHH)
  5. 旧版全网络匹配删除（仅参考）   → matched_full_random（不用于主结论）

主对比（设计文档 §4 冻结判据）：
  - noHH vs matched_IO_random：HH 边具体位置/组织是否有特殊贡献
  - noHH vs HH_DAG：递归环是否有贡献
  - F0 vs HH_rewired：度分布是否解释了大部分结构优势

证伪规则（计划 §六）：
  - 若 noHH ≈ matched_IO_random：接受"主要是删边效应"，降级 C3 主张
  - 若 HH_rewired ≈ F0：结构优势主要来自度分布
  - 若 HH_DAG ≈ noHH：递归环无特殊贡献

产出：results_v5/causal_propagation/causal_propagation_v5_{task}.json
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
_CODE_DIR = _HERE.parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol
from experiments_v2.core.scaffold import NeuronScaffold
from experiments_v2.core.encoder import ObservationEncoder
from experiments_v2.core.controller import UnifiedSNNController
from experiments_v2.core.weight_construction import F0
from experiments_v2.tasks.adapter import make_env
from experiments_v2.mechanisms.matched_connectivity_controls import (
    audit_condition,
    delete_hh_edges_only,
    hh_bounds,
    interface_counts,
    is_dag,
    make_hh_dag,
    rewire_hh_preserving_degree,
)

CONDITIONS = [
    "F0",
    "noHH",
    "HH_rewired",
    "HH_DAG",
    "HH_dose_25",
    "HH_dose_50",
    "HH_dose_75",
    "matched_IO_random",
    "matched_full_random",  # 旧版全网络匹配删除，仅作参考
]


def load_v3_topology(task_name, evo_seed, remote_root, topo_subdir="topology"):
    """加载 v3 冻结拓扑（兼容两种命名）。

    topo_subdir 默认 "topology"（v3 冻结队列 3000-3019）；
    c2 confirmatory 队列（3020-3034，prereg 2026-07-30）用 "topology_c2"。
    """
    for name in (
        f"v3_F0_topology_{task_name}_seed{evo_seed}.npy",
        f"F0_topology_{task_name}_seed{evo_seed}.npy",
    ):
        path = f"{remote_root}/results_v3/{topo_subdir}/{task_name}/{name}"
        if os.path.exists(path):
            return np.load(path)
    return None


def build_conditions(f0_adj, n_in, n_out, evo_seed):
    """构造全部干预条件的邻接矩阵（接口全部固定，matched_full_random 除外）。"""
    n_total = f0_adj.shape[0]
    h_start, h_end = hh_bounds(n_total, n_in, n_out)
    n_hh_edges = int(f0_adj[h_start:h_end, h_start:h_end].sum())

    conds = {"F0": f0_adj.copy()}
    conds["noHH"] = delete_hh_edges_only(
        f0_adj, n_in, n_out, n_hh_edges, np.random.RandomState(evo_seed * 11 + 1)
    )
    conds["HH_rewired"] = rewire_hh_preserving_degree(
        f0_adj, n_in, n_out, np.random.RandomState(evo_seed * 11 + 2)
    )
    conds["HH_DAG"] = make_hh_dag(
        f0_adj, n_in, n_out, np.random.RandomState(evo_seed * 11 + 3)
    )
    for dose in (25, 50, 75):
        conds[f"HH_dose_{dose}"] = delete_hh_edges_only(
            f0_adj, n_in, n_out, int(n_hh_edges * dose / 100),
            np.random.RandomState(evo_seed * 11 + 3 + dose),
        )
    # 核心匹配对照：仅限 HH 子图内删除同数量边。
    # 注意：删除数 = 全部 HH 边数时与 noHH 重合（计划 §2.1.2 已注明）。
    conds["matched_IO_random"] = delete_hh_edges_only(
        f0_adj, n_in, n_out, n_hh_edges, np.random.RandomState(evo_seed * 2000)
    )
    # 旧版对照（v4 缺陷 F1 的逻辑，仅保留作参考，不用于主结论）
    rng = np.random.RandomState(evo_seed * 1000)
    active = np.argwhere(f0_adj > 0)
    del_idx = rng.choice(len(active), size=min(n_hh_edges, len(active)), replace=False)
    adj_matched_full = f0_adj.copy()
    for idx in del_idx:
        r, c = active[idx]
        adj_matched_full[r, c] = 0.0
    conds["matched_full_random"] = adj_matched_full
    return conds, n_hh_edges


def evaluate_adj(task, scaffold, adj, encoder, ctrl_proto, seeds):
    """在给定邻接矩阵上评估平均回报。"""
    env = make_env(task)
    scores = []
    for seed in seeds:
        ctrl = UnifiedSNNController(scaffold, adj, encoder, ctrl_proto, condition=F0)
        result = ctrl.rollout(env, task.max_steps, seed=seed)
        scores.append(result["total_reward"])
    env.close()
    return float(np.mean(scores)), [float(s) for s in scores]


def run_causal_propagation_v5(task_name, remote_root, output_dir,
                              mechanism_seeds, n_topologies=20, seed_base=3000):
    task = TASK_SPECS[task_name]
    ctrl_proto = ControllerProtocol()
    encoder = ObservationEncoder.load(
        f"{remote_root}/results_v2/pilot/observation_stats/observation_stats_{task_name}.json"
    )
    n_in, n_out = task.n_input, task.n_output
    eval_seeds = mechanism_seeds[:20]

    out_path = Path(output_dir) / f"causal_propagation_v5_{task_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume：已有 seed 记录则跳过
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {"task": task_name, "version": "v5", "conditions": CONDITIONS,
                   "per_seed": [], "audits": []}
    done_seeds = {r["seed"] for r in results["per_seed"]}

    for seed_idx in range(min(n_topologies, 20)):
        evo_seed = seed_base + seed_idx
        if evo_seed in done_seeds:
            print(f"  [v5 {task_name} seed={evo_seed}] already done, skip", flush=True)
            continue
        f0_adj = load_v3_topology(task_name, evo_seed, remote_root)
        if f0_adj is None:
            print(f"  [v5 {task_name} seed={evo_seed}] topology missing, skip", flush=True)
            continue

        t0 = time.time()
        scaffold = NeuronScaffold.from_seed(seed=evo_seed, task=task, controller=ctrl_proto)
        conds, n_hh_edges = build_conditions(f0_adj, n_in, n_out, evo_seed)

        seed_result = {"seed": evo_seed, "n_hh_edges": n_hh_edges,
                       "n_total_edges": int(f0_adj.sum())}
        for cond_name, adj in conds.items():
            mean_score, per_episode = evaluate_adj(
                task, scaffold, adj, encoder, ctrl_proto, eval_seeds
            )
            seed_result[cond_name] = mean_score
            seed_result[f"{cond_name}__episodes"] = per_episode
            print(f"    [{cond_name}] {mean_score:.1f}", flush=True)

        # 增量审计
        seed_audit = {"seed": evo_seed}
        for cond_name, adj in conds.items():
            if cond_name == "F0":
                continue
            a = audit_condition(f0_adj, adj, n_in, n_out)
            if cond_name == "HH_DAG":
                h_start, h_end = hh_bounds(adj.shape[0], n_in, n_out)
                a["hh_is_dag"] = bool(is_dag(adj[h_start:h_end, h_start:h_end]))
            if cond_name == "HH_rewired":
                # 度保持验证
                h_start, h_end = hh_bounds(adj.shape[0], n_in, n_out)
                a["degree_preserved"] = bool(
                    np.array_equal(f0_adj[h_start:h_end, :].sum(axis=1),
                                   adj[h_start:h_end, :].sum(axis=1))
                    and np.array_equal(f0_adj[:, h_start:h_end].sum(axis=0),
                                       adj[:, h_start:h_end].sum(axis=0))
                )
            seed_audit[cond_name] = a

        seed_result["delta_noHH"] = seed_result["F0"] - seed_result["noHH"]
        seed_result["delta_HH_rewired"] = seed_result["F0"] - seed_result["HH_rewired"]
        seed_result["delta_HH_DAG"] = seed_result["F0"] - seed_result["HH_DAG"]
        seed_result["elapsed_sec"] = round(time.time() - t0, 1)

        results["per_seed"].append(seed_result)
        results["audits"].append(seed_audit)

        # 每个 seed 完成后立即落盘（崩溃可恢复）
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"  [v5 {task_name} seed={evo_seed}] F0={seed_result['F0']:.1f} "
              f"noHH={seed_result['noHH']:.1f} rewired={seed_result['HH_rewired']:.1f} "
              f"DAG={seed_result['HH_DAG']:.1f} ({seed_result['elapsed_sec']}s)",
              flush=True)

    print(f"  Saved {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--remote-root", type=str,
                        default="/home/lixuewei/tmp/struct_mod_v2")
    parser.add_argument("--output-dir", type=str,
                        default="/home/lixuewei/tmp/struct_mod_v2/results_v5/causal_propagation")
    parser.add_argument("--seed-base", type=int, default=3000,
                        help="evo_seed = seed_base + seed_idx (POMDP 用 5000)")
    args = parser.parse_args()

    mechanism_seeds = list(range(93000, 93020))
    run_causal_propagation_v5(args.task, args.remote_root, args.output_dir,
                              mechanism_seeds, seed_base=args.seed_base)
