"""E5 匹配删除实验（度+符号+模块位置匹配的骨架删除对照）。

prereg: prereg_route_v1.md「E5 匹配删除实验设计登记（2026-08-03 15:25）」。

仪器原语全部复用已冻结代码：
  - E2r-v2: stable_seed / edge_set_hash / eval_return / delete_edge_set
    （run_e2r_matched_lesion.py，SHA 冻结 0d998e3f）；
  - E1: edge_contributions + SKEL_FRAC（run_skeleton_v6.py，确定性无随机）；
  - eval 段 EVAL3=93040-93059（held-out，与 E2r-v2 同段）。

每 cell：F0 / del_skel 各测一次；R=20 个 rep（主分析 r<5），每 rep
骨架边在非骨架池中贪婪匹配（cost = 1000·符号 + 100·源模块 + 100·目标模块
+ |出度差| + |入度差|，模块 = module_labels_for_k(8)），另取同数随机非
骨架边集参照；rep 种子 stable_seed("e5_matched", tag, r)，边集哈希落盘。

用法：
  CP : python3 -u run_e5_matched_skeleton.py --cohort e1 --scaffold-seed 7000 --ga-seed 7100 --budget 20000
  Acr: python3 -u run_e5_matched_skeleton.py --cohort acr --seed 3000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol  # noqa: E402
from experiments_v2.core.scaffold import NeuronScaffold  # noqa: E402
from experiments_v2.core.encoder import ObservationEncoder  # noqa: E402
from experiments_v2.mechanisms.run_causal_propagation_v5 import (  # noqa: E402
    load_v3_topology)
from experiments_v2.mechanisms.run_e2r_matched_lesion import (  # noqa: E402
    stable_seed, edge_set_hash, eval_return, delete_edge_set, EVAL3_SEEDS)
from experiments_v2.mechanisms.run_skeleton_v6 import (  # noqa: E402
    edge_contributions, SKEL_FRAC)

R_MAIN = 20          # prereg 主分析用 r<5；全部 20 作稳定性层
K_MODULES = 8        # 与调制阶梯固定八模块划分一致


def edge_features_e5(edges, scaffold, f0_adj, mod):
    """每条 (s,d) 边：(源符号, 源模块, 目标模块, 源出度(HH), 目标入度(HH))。"""
    out_deg = (f0_adj != 0).sum(axis=0)   # adj[d,s]：列和=源出度
    in_deg = (f0_adj != 0).sum(axis=1)    # 行和=目标入度
    feats = {}
    for s, d in edges:
        feats[(int(s), int(d))] = (
            int(np.sign(scaffold.neuron_types[int(s)])),
            int(mod[int(s)]), int(mod[int(d)]),
            int(out_deg[int(s)]), int(in_deg[int(d)]))
    return feats


def greedy_match_e5(skel_edges, pool, feats, rng):
    """为每条骨架边在非骨架池中贪婪找最近匹配（符号>源模块>目标模块>度）。"""
    remaining = list(pool)
    rng.shuffle(remaining)
    hi_order = list(skel_edges)
    rng.shuffle(hi_order)
    matched, qualities = [], []
    for h in hi_order:
        fh = feats[h]
        best, best_cost = None, None
        for e in remaining:
            fe = feats[e]
            cost = (0 if fe[0] == fh[0] else 1000) \
                + (0 if fe[1] == fh[1] else 100) \
                + (0 if fe[2] == fh[2] else 100) \
                + abs(fe[3] - fh[3]) + abs(fe[4] - fh[4])
            if best_cost is None or cost < best_cost:
                best, best_cost = e, cost
        matched.append(best)
        remaining.remove(best)
        fe = feats[best]
        qualities.append({
            "sign_match": bool(fe[0] == fh[0]),
            "modpair_match": bool(fe[1] == fh[1] and fe[2] == fh[2]),
            "out_deg_diff": abs(fe[3] - fh[3]),
            "in_deg_diff": abs(fe[4] - fh[4])})
    return matched, qualities


def run_body(task, task_name, f0_adj, scaffold, enc, proto, tag, lineage,
             extra, out_dir):
    n_in, n_out = task.n_input, task.n_output
    t0 = time.time()
    edges, contrib = edge_contributions(f0_adj, n_in, n_out)
    n_del = max(1, int(round(SKEL_FRAC * len(edges))))
    order = np.argsort(-contrib)
    skel_edges = [tuple(map(int, e)) for e in edges[order[:n_del]]]
    skel_set = set(skel_edges)
    pool = [tuple(map(int, e)) for e in edges if tuple(map(int, e)) not in skel_set]
    mod = scaffold.module_labels_for_k(K_MODULES)
    feats = edge_features_e5(set(skel_edges) | set(pool), scaffold, f0_adj, mod)

    J_F0 = eval_return(f0_adj, scaffold, enc, proto, task, EVAL3_SEEDS)
    J_del_skel = eval_return(delete_edge_set(f0_adj, skel_edges),
                             scaffold, enc, proto, task, EVAL3_SEEDS)

    reps = []
    for r in range(R_MAIN):
        rep_seed = stable_seed("e5_matched", tag, r)
        rng = np.random.RandomState(rep_seed)
        m_edges, quals = greedy_match_e5(skel_edges, pool, feats, rng)
        r_idx = rng.choice(len(pool), size=n_del, replace=False)
        rand_edges = [pool[i] for i in r_idx]
        J_m = eval_return(delete_edge_set(f0_adj, m_edges),
                          scaffold, enc, proto, task, EVAL3_SEEDS)
        J_r = eval_return(delete_edge_set(f0_adj, rand_edges),
                          scaffold, enc, proto, task, EVAL3_SEEDS)
        reps.append({
            "rep": r, "rep_seed": rep_seed,
            "matched_hash": edge_set_hash(m_edges),
            "rand_hash": edge_set_hash(rand_edges),
            "sign_match_frac": float(np.mean([q["sign_match"] for q in quals])),
            "modpair_match_frac": float(np.mean([q["modpair_match"] for q in quals])),
            "out_deg_diff_med": float(np.median([q["out_deg_diff"] for q in quals])),
            "in_deg_diff_med": float(np.median([q["in_deg_diff"] for q in quals])),
            "J_del_matched": J_m, "J_del_rand": J_r})
        print(f"  [{tag}] rep {r}: matched={J_m:.2f} rand={J_r:.2f} "
              f"sign={reps[-1]['sign_match_frac']:.3f} "
              f"mod={reps[-1]['modpair_match_frac']:.3f}", flush=True)

    rec = {"version": "e5_matched_v1", "task": task_name, "tag": tag,
           "lineage": lineage, "eval_seeds": EVAL3_SEEDS,
           "skel_frac": SKEL_FRAC, "n_hh": int(len(edges)), "n_skel": n_del,
           "skel_hash": edge_set_hash(skel_edges),
           "k_modules": K_MODULES, "R": R_MAIN,
           "J_F0": J_F0, "J_del_skel": J_del_skel,
           "del_drop_skel": J_F0 - J_del_skel,
           "reps": reps, "elapsed_sec": time.time() - t0}
    rec.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"e5_{task_name}_{tag}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print(f"[{tag}] done: F0={J_F0:.2f} del_skel={J_del_skel:.2f} "
          f"drop={rec['del_drop_skel']:.2f} ({rec['elapsed_sec']:.0f}s) -> {out}",
          flush=True)


def run_e1_cell(scaffold_seed, ga_seed, budget, out_dir):
    task_name = "CartPole"
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/"
                   f"observation_stats_{task_name}.json"))
    tag = f"sf{scaffold_seed}_ga{ga_seed}_b{budget}"
    f0_adj = np.load(ROOT / "results_v3/e1_variance" / task_name / tag /
                     f"e1_F0_topo_{task_name}_{tag}.npy")
    scaffold = NeuronScaffold.from_seed(seed=scaffold_seed, task=task,
                                        controller=proto)
    run_body(task, task_name, f0_adj, scaffold, enc, proto, tag,
             f"sf{scaffold_seed}_ga{ga_seed}",
             {"scaffold_seed": scaffold_seed, "ga_seed": ga_seed,
              "budget": budget}, out_dir)


def run_acr_cell(seed, out_dir):
    task_name = "Acrobot"
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/"
                   f"observation_stats_{task_name}.json"))
    f0_adj = load_v3_topology(task_name, seed, str(ROOT), "topology")
    scaffold = NeuronScaffold.from_seed(seed=seed, task=task, controller=proto)
    run_body(task, task_name, f0_adj, scaffold, enc, proto, f"seed{seed}",
             f"seed{seed}", {"seed": seed}, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["e1", "acr"], required=True)
    ap.add_argument("--scaffold-seed", type=int)
    ap.add_argument("--ga-seed", type=int)
    ap.add_argument("--budget", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.cohort == "e1":
        run_e1_cell(args.scaffold_seed, args.ga_seed, args.budget,
                    ROOT / "results_v6/e5_matched_skeleton")
    else:
        run_acr_cell(args.seed, ROOT / "results_v6/e5_matched_skeleton_acr")
