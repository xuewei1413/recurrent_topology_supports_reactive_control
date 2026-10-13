"""骨架必要性 + 移植充分性实验（文档步骤 3-4）。

骨架：HH 边按"穿过它的 in→out 加权路径多重性"排序（与 proxy 同权
 w=(1,0.5,0.25)，a 步前缀 × b 步后缀，a+b≤2）。
  contribution(i→j) = Σ_{a+b≤2} w_{a+b+1} · pin_a[i] · pout_b[j]

条件（行为评估 EVAL_SEEDS=93000-93019）：
  必要性（从 F0 删除，计数匹配）：
    del_skel   : 删 top-20% 贡献 HH 边
    del_random : 删同数随机 HH 边（seeded）
    del_low    : 删同数最低贡献 HH 边（反向对照：应≈无损伤）
  充分性（两个互补条件 + 权重值对照）：
    skel_only  : 只保留骨架边+原始精确权重，删其余 80% HH 边
                 （≈F0 ⇒ 骨架+权重即完整因果网络；崩 ⇒ 非骨架边也必要）
    # 注：拓扑矩阵为二值（0/1），边权=掩码×源符号×模块增益 ⇒ 系统内无逐边
    # 连续权重可置换，"权重值"问题由 warmstart 实验在增益维回答。
    transplant : 从 RW-free 全预算定向 swaps（n_swaps=缺失骨架边数）
                 最大化恢复骨架原边——纯拓扑可移植性（权重值被重接打乱）
    transplant_ctrl : 同次数随机 swaps
    skel_plus_rand : 骨架原边固定 + 其余 80% 边度保持随机重接
                 （≈F0 ⇒ 背景中性，transplant 失败=恢复不全；
                   崩塌 ⇒ 错误背景主动干扰）
  ρ = (J_cond − J_RW) / (J_F0 − J_RW)，判据 ρ ≥ 0.5 且 ≫ ρ_ctrl。

任务：CartPole + Acrobot（route 四层通过的两任务），20 拓扑/任务。

产出：results_v6/skeleton/skeleton_v6_{task}_{seed}.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol  # noqa: E402
from experiments_v2.core.scaffold import NeuronScaffold  # noqa: E402
from experiments_v2.core.encoder import ObservationEncoder  # noqa: E402
from experiments_v2.mechanisms.run_causal_propagation_v5 import (  # noqa: E402
    load_v3_topology, evaluate_adj,
)
from experiments_v2.mechanisms.run_route_decisive_v6 import (  # noqa: E402
    random_rewire, hh_edge_list, EVAL_SEEDS, PROXY_W,
)

SKEL_FRAC = 0.20


def edge_contributions(adj, n_in, n_out):
    """每条 HH 边的加权路径贡献。返回 (edges, contrib)。"""
    n = adj.shape[0]
    A = sparse.csr_matrix(adj.astype(float))
    w1, w2, w3 = PROXY_W
    # 前缀路径 in→node（a 步）
    pin = [np.zeros(n), np.zeros(n), np.zeros(n)]
    pin[0][:n_in] = 1.0                       # a=0：自身即输入
    col_in = A[:, :n_in].sum(axis=1)
    pin[1] = np.asarray(col_in).ravel()       # a=1
    pin[2] = np.asarray((A @ col_in)).ravel()  # a=2
    # 后缀路径 node→out（b 步）
    pout = [np.zeros(n), np.zeros(n), np.zeros(n)]
    pout[0][n - n_out:] = 1.0
    row_out = A[n - n_out:, :].sum(axis=0)
    pout[1] = np.asarray(row_out).ravel()
    pout[2] = np.asarray(row_out @ A).ravel()
    edges = hh_edge_list(adj, n_in, n_out)
    contrib = (w1 * (pin[0][edges[:, 0]] * pout[0][edges[:, 1]])
               + w2 * (pin[0][edges[:, 0]] * pout[1][edges[:, 1]]
                       + pin[1][edges[:, 0]] * pout[0][edges[:, 1]])
               + w3 * (pin[0][edges[:, 0]] * pout[2][edges[:, 1]]
                       + pin[1][edges[:, 0]] * pout[1][edges[:, 1]]
                       + pin[2][edges[:, 0]] * pout[0][edges[:, 1]]))
    return edges, contrib


def delete_edges(adj, edges):
    child = adj.copy()
    for s, d in edges:
        child[d, s] = 0.0
    return child


def guided_transplant(rw_adj, skel_set, n_in, n_out, n_swaps, n_cand, rng):
    """度保持 swaps，目标=最大化骨架原边恢复数（overlap 贪心）。"""
    child = rw_adj.copy()
    edges = hh_edge_list(child, n_in, n_out)

    def overlap():
        return sum(1 for s, d in edges if (int(s), int(d)) in skel_set)

    done, attempts = 0, 0
    while done < n_swaps and attempts < n_swaps * 80 + 200:
        attempts += 1
        best, best_ov = None, None
        from experiments_v2.mechanisms.run_route_decisive_v6 import (
            propose_swap, apply_swap)
        for _ in range(n_cand):
            sw = propose_swap(child, edges, rng)
            if sw is None:
                continue
            apply_swap(child, edges, sw)
            ov = overlap()
            i1, i2, (a, b), (c, d) = sw
            apply_swap(child, edges, (i1, i2, (a, d), (c, b)))  # 撤销
            if best_ov is None or ov > best_ov:
                best, best_ov = sw, ov
        if best is None:
            continue
        apply_swap(child, edges, best)
        done += 1
    return child, done


def run(task_name, evo_seed, out_dir, topo_subdir="topology"):
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/observation_stats_{task_name}.json"))
    n_in, n_out = task.n_input, task.n_output
    f0_adj = load_v3_topology(task_name, evo_seed, str(ROOT), topo_subdir)
    if f0_adj is None:
        return
    scaffold = NeuronScaffold.from_seed(seed=evo_seed, task=task, controller=proto)
    t0 = time.time()

    edges, contrib = edge_contributions(f0_adj, n_in, n_out)
    n_del = max(1, int(round(SKEL_FRAC * len(edges))))
    order = np.argsort(-contrib)
    skel_edges = [tuple(e) for e in edges[order[:n_del]]]
    low_edges = [tuple(e) for e in edges[order[-n_del:]]]
    rng = np.random.RandomState(evo_seed * 211 + 5)
    rand_idx = rng.choice(len(edges), size=n_del, replace=False)
    rand_edges = [tuple(e) for e in edges[rand_idx]]
    skel_set = set(skel_edges)

    rec = {"version": "v6_skeleton", "task": task_name, "seed": evo_seed,
           "skel_frac": SKEL_FRAC, "n_skel": n_del, "n_hh": int(len(edges)),
           "eval_seeds": EVAL_SEEDS, "conditions": {}}

    def behav(adj):
        m, per = evaluate_adj(task, scaffold, adj, enc, proto, EVAL_SEEDS)
        return {"return_mean": m, "per_episode": per}

    conds = {"F0": f0_adj,
             "del_skel": delete_edges(f0_adj, skel_edges),
             "del_random": delete_edges(f0_adj, rand_edges),
             "del_low": delete_edges(f0_adj, low_edges)}
    # skel_only：只保留骨架 HH 边（权重原值），其余 HH 边全部清零
    non_skel = [tuple(e) for e in edges if tuple(e) not in skel_set]
    conds["skel_only"] = delete_edges(f0_adj, non_skel)
    # skel_plus_rand：骨架原边固定 + 其余 80% HH 边度保持随机重接
    # （与 skel_only 对照 ⇒ 判定 transplant 失败是因"背景干扰"还是"恢复不全"；
    #   propose_swap 要求目标空位 ⇒ 骨架位置天然不可被占用，全局度保持）
    from experiments_v2.mechanisms.run_route_decisive_v6 import (
        propose_swap, apply_swap)
    child = f0_adj.copy()
    bg_edges = np.array([list(e) for e in edges if tuple(e) not in skel_set])
    rng_bg = np.random.RandomState(evo_seed * 733 + 11)
    n_bg_sw = len(bg_edges) * 10
    done_bg, att_bg = 0, 0
    while done_bg < n_bg_sw and att_bg < n_bg_sw * 50 + 100:
        att_bg += 1
        sw = propose_swap(child, bg_edges, rng_bg)
        if sw is None:
            continue
        apply_swap(child, bg_edges, sw)
        done_bg += 1
    conds["skel_plus_rand"] = child
    # RW-free：与 C3 同协议的度保持全重接（seed 与 route 实验不冲突）
    rw_adj, _ = random_rewire(f0_adj, n_in, n_out, int(f0_adj.sum()) * 10,
                              np.random.RandomState(evo_seed * 100 + 50))
    conds["RW_free"] = rw_adj
    # 移植：全预算（缺失骨架边数次 swaps，每次最多恢复 2 条）
    missing = sum(1 for s, d in skel_edges if rw_adj[d, s] == 0)
    n_sw = max(1, missing)
    tr_adj, tr_done = guided_transplant(rw_adj, skel_set, n_in, n_out,
                                        n_sw, 40,
                                        np.random.RandomState(evo_seed * 59 + 1))
    conds["transplant"] = tr_adj
    # skel_partial：只保留 transplant 实际恢复的骨架边子集、无背景
    # （与 transplant 同边集 ⇒ 精确分离"恢复不全"vs"背景干扰"的贡献）
    child_p = delete_edges(f0_adj, non_skel)
    for s_, d_ in skel_edges:
        if tr_adj[d_, s_] == 0:
            child_p[d_, s_] = 0.0
    conds["skel_partial"] = child_p
    rc_adj, _ = random_rewire(rw_adj, n_in, n_out, n_sw,
                              np.random.RandomState(evo_seed * 59 + 2))
    conds["transplant_ctrl"] = rc_adj
    rec["transplant_swaps"] = tr_done
    rec["skel_edges_missing_in_RW"] = int(missing)
    # 移植后骨架边恢复审计
    rec["skel_edges_restored_transplant"] = int(
        sum(1 for s, d in skel_edges if tr_adj[d, s] > 0))
    rec["skel_edges_present_ctrl"] = int(
        sum(1 for s, d in skel_edges if rc_adj[d, s] > 0))

    for name, adj in conds.items():
        rec["conditions"][name] = behav(adj)
        print(f"[{task_name} {evo_seed}] {name}: "
              f"{rec['conditions'][name]['return_mean']:.1f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        op = out_dir / f"skeleton_v6_{task_name}_{evo_seed}.json"
        with open(op, "w") as f:
            json.dump(rec, f, indent=2)
    rec["minutes"] = (time.time() - t0) / 60
    op = out_dir / f"skeleton_v6_{task_name}_{evo_seed}.json"
    with open(op, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved {op} ({rec['minutes']:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--topo-subdir", default="topology",
                    help="results_v3 下的拓扑目录；c2 队列用 topology_c2")
    ap.add_argument("--out-dir", default="results_v6/skeleton",
                    help="ROOT 下的相对输出目录")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    op = out_dir / f"skeleton_v6_{args.task}_{args.seed}.json"
    if op.exists():
        with open(op) as f:
            d = json.load(f)
        if "skel_partial" in d.get("conditions", {}):
            print("done, skip", flush=True)
            return
    run(args.task, args.seed, out_dir, args.topo_subdir)


if __name__ == "__main__":
    main()
