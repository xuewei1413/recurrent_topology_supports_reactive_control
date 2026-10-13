"""损伤型路由指纹（lesion-response route fingerprint）——prereg 分类 A 的
仪器升级重测（不重跑搜索，只重测既有构造）。

动机（2026-07-30 route 主实验意外发现）：
  扰动响应张量 R[d,a,τ] 对冗余路径补偿天然钝感（摧毁一条并行路径，
  响应在几步内被其余路径补偿），而 k 步路径多重性（proxy）同时通过
  H2/H3。本仪器直接测"信号实际依赖哪些边组"：
    L[g] = mean_states Σ_{τ=1..5} |margin_{lesion g} − margin_base|   (acc 读出)
  F0 的 HH 边按端点位置 k-means 分 G 组；变体删除的是"组内仍存在的边"。
  S_lesion = cosine(L_variant, L_F0)（组空间一致，指纹可直接比）。

构造重建：route_decisive 未存邻接，但搜索全种子确定（preserve=seed*37+s、
  destroy=*91+s、rescue=*53+s、rescue_ctrl=*77+s），按同参数确定性重建。

模式：
  pilot   : F0 指纹 + 4 组半分 retest（冻结 SESOI_lesion）
  measure : 重建 12 条件 + F0，输出各条件 S_lesion 与指纹

产出：results_v6/lesion_route/lesion_route_v6_{task}_{seed}.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol  # noqa: E402
from experiments_v2.core.weight_construction import F0  # noqa: E402
from experiments_v2.core.scaffold import NeuronScaffold  # noqa: E402
from experiments_v2.core.encoder import ObservationEncoder  # noqa: E402
from experiments_v2.core.controller import UnifiedSNNController  # noqa: E402
from experiments_v2.mechanisms.run_causal_propagation_v5 import (  # noqa: E402
    load_v3_topology,
)
from experiments_v2.mechanisms.run_route_pilot_v6 import (  # noqa: E402
    collect_calibration, cosine,
)
from experiments_v2.mechanisms.run_route_decisive_v6 import (  # noqa: E402
    guided_rewire, random_rewire, hh_edge_list, proxy_vec, RESCUE_EDGE_FRAC,
)
from experiments_v2.run_e_compression_v5 import kmeans_numpy  # noqa: E402

G_GROUPS = 32
TAU_MAX = 5
H_FRAC = 0.20
N_CAND = 40
N_SEARCH_SEEDS = 3


def make_edge_groups(f0_adj, n_in, n_out, evo_seed):
    """F0 HH 边按 (src,dst) 归一化位置 k-means 分 G 组。返回 (edges, labels)。"""
    edges = hh_edge_list(f0_adj, n_in, n_out)      # (m,2) [src,dst]
    n = f0_adj.shape[0]
    feats = np.column_stack([edges[:, 0] / n, edges[:, 1] / n])
    rng = np.random.RandomState(evo_seed * 17 + 3)
    labels = kmeans_numpy(feats, G_GROUPS, rng)
    return edges, labels


def margin_vec(acc, n_out):
    a = acc[-n_out:]
    tot = a.sum()
    return a / tot if tot > 1e-12 else np.zeros(n_out)


def replay_margins(ctrl, adj, scaffold, enc, proto, episodes, states, lesion_edges):
    """对每个校准态：从 v 快照重放 τ=0..5，返回 (n_states, TAU+1, n_out) margin。

    lesion_edges: 要删除的 (src,dst) 列表（None = 基线）。
    """
    if lesion_edges is not None and len(lesion_edges):
        adj_l = adj.copy()
        for s, d in lesion_edges:
            adj_l[d, s] = 0.0
    else:
        adj_l = adj
    ctrl_l = UnifiedSNNController(scaffold, adj_l, enc, proto, condition=F0)
    clone = copy.deepcopy(ctrl_l)
    out = np.zeros((len(states), TAU_MAX + 1, ctrl_l.n_out))
    for si, (ei, t) in enumerate(states):
        ep = episodes[ei]
        v_snap = ep["snaps"][t][0]
        seq = ep["seq"]
        clone.v[:] = v_snap
        for k in range(TAU_MAX + 1):
            _, acc, _ = clone.forward(seq[t + k], return_trace=True)
            out[si, k] = margin_vec(acc, ctrl_l.n_out)
    return out


def lesion_fingerprint(adj, scaffold, enc, proto, episodes, states,
                       groups_on_f0, base_margins=None):
    """L[g] = mean_states Σ_τ ||margin_lesion_g − margin_base||_1。"""
    edges, labels = groups_on_f0
    if base_margins is None:
        base_margins = replay_margins(None, adj, scaffold, enc, proto,
                                      episodes, states, None)
    L = np.zeros(G_GROUPS)
    present = adj[edges[:, 1], edges[:, 0]] > 0     # 组内仍存在于本图的边
    for g in range(G_GROUPS):
        idx = np.where((labels == g) & present)[0]
        if len(idx) == 0:
            continue
        m_les = replay_margins(None, adj, scaffold, enc, proto,
                               episodes, states, [tuple(e) for e in edges[idx]])
        diff = np.abs(m_les[:, 1:, :] - base_margins[:, 1:, :])
        L[g] = float(diff.sum(axis=(1, 2)).mean())
    return L, base_margins


def reconstruct_variants(f0_adj, n_in, n_out, evo_seed):
    """按 route_decisive 同种子确定性重建 12 条件邻接。"""
    n_hh = len(hh_edge_list(f0_adj, n_in, n_out))
    h = max(2, int(round(H_FRAC * n_hh / 2)))
    n_rescue = max(1, int(round(RESCUE_EDGE_FRAC * n_hh / 2)))
    ref_vec = proxy_vec(f0_adj, n_in, n_out)
    variants = []
    for sseed in range(N_SEARCH_SEEDS):
        rng = np.random.RandomState(evo_seed * 37 + sseed)
        adj_p, _, _ = guided_rewire(f0_adj, ref_vec, n_in, n_out,
                                    h, N_CAND, rng, "preserve")
        rng = np.random.RandomState(evo_seed * 91 + sseed)
        adj_d, _, _ = guided_rewire(f0_adj, ref_vec, n_in, n_out,
                                    h, N_CAND, rng, "destroy")
        rng = np.random.RandomState(evo_seed * 53 + sseed)
        adj_r, _, _ = guided_rewire(adj_d, ref_vec, n_in, n_out,
                                    n_rescue, N_CAND, rng, "rescue")
        rng = np.random.RandomState(evo_seed * 77 + sseed)
        adj_rc, _ = random_rewire(adj_d, n_in, n_out, n_rescue, rng)
        variants.append({"search_seed": sseed, "preserved": adj_p,
                         "destroyed": adj_d, "rescue": adj_r,
                         "rescue_ctrl": adj_rc})
    return variants


def run(task_name, evo_seed, mode, n_states, out_dir):
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/observation_stats_{task_name}.json"))
    f0_adj = load_v3_topology(task_name, evo_seed, str(ROOT))
    if f0_adj is None:
        print(f"[{task_name} {evo_seed}] topology missing", flush=True)
        return
    scaffold = NeuronScaffold.from_seed(seed=evo_seed, task=task, controller=proto)
    t0 = time.time()

    from experiments_v2.tasks.adapter import make_env
    ctrl = UnifiedSNNController(scaffold, f0_adj, enc, proto, condition=F0)
    env = make_env(task)
    episodes, states = collect_calibration(ctrl, env, task, n_states,
                                           np.random.RandomState(2026))
    env.close()
    groups = make_edge_groups(f0_adj, task.n_input, task.n_output, evo_seed)

    rec = {"version": "v6_lesion_route", "task": task_name, "seed": evo_seed,
           "G": G_GROUPS, "n_states": len(states), "conditions": []}

    # F0 指纹 + retest
    rng_qc = np.random.RandomState(7)
    tr = []
    if mode == "pilot":
        L_full, _ = lesion_fingerprint(f0_adj, scaffold, enc, proto,
                                       episodes, states, groups)
        for rep in range(2):
            perm = rng_qc.permutation(len(states))
            h1 = [states[i] for i in perm[: len(states) // 2]]
            h2 = [states[i] for i in perm[len(states) // 2:]]
            L1, _ = lesion_fingerprint(f0_adj, scaffold, enc, proto,
                                       episodes, h1, groups)
            L2, _ = lesion_fingerprint(f0_adj, scaffold, enc, proto,
                                       episodes, h2, groups)
            c = cosine(L1, L2)
            if c is not None:
                tr.append(c)
        rec["F0_retest"] = tr
        rec["F0_retest_mean"] = float(np.mean(tr)) if tr else None
        rec["minutes"] = (time.time() - t0) / 60
        print(f"[pilot {task_name} {evo_seed}] lesion retest S={rec['F0_retest_mean']} "
              f"({rec['minutes']:.1f} min)", flush=True)
    else:
        L_F0, _ = lesion_fingerprint(f0_adj, scaffold, enc, proto,
                                     episodes, states, groups)
        rec["L_F0_norm"] = float(np.linalg.norm(L_F0))
        variants = reconstruct_variants(f0_adj, task.n_input, task.n_output,
                                        evo_seed)
        for v in variants:
            for cond in ("preserved", "destroyed", "rescue", "rescue_ctrl"):
                adj_v = v[cond]
                L_v, _ = lesion_fingerprint(adj_v, scaffold, enc, proto,
                                            episodes, states, groups)
                rec["conditions"].append({
                    "search_seed": v["search_seed"], "condition": cond,
                    "S_lesion": cosine(L_v, L_F0),
                })
            print(f"[{task_name} {evo_seed} s{v['search_seed']}] "
                  f"S_lesion p={rec['conditions'][-4]['S_lesion']:.3f} "
                  f"d={rec['conditions'][-3]['S_lesion']:.3f} "
                  f"r={rec['conditions'][-2]['S_lesion']:.3f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        rec["minutes"] = (time.time() - t0) / 60

    op = out_dir / (f"lesion_pilot_v6_{task_name}_{evo_seed}.json"
                    if mode == "pilot" else
                    f"lesion_route_v6_{task_name}_{evo_seed}.json")
    with open(op, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved {op}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["pilot", "measure"], default="measure")
    ap.add_argument("--n-states", type=int, default=200)
    args = ap.parse_args()
    out_dir = ROOT / "results_v6/lesion_route"
    out_dir.mkdir(parents=True, exist_ok=True)
    op = out_dir / (f"lesion_pilot_v6_{args.task}_{args.seed}.json"
                    if args.mode == "pilot" else
                    f"lesion_route_v6_{args.task}_{args.seed}.json")
    if op.exists():
        with open(op) as f:
            d = json.load(f)
        if args.mode == "pilot" and d.get("F0_retest"):
            print("done, skip", flush=True)
            return
        if args.mode == "measure" and len(d.get("conditions", [])) >= 12:
            print("done, skip", flush=True)
            return
    run(args.task, args.seed, args.mode, args.n_states, out_dir)


if __name__ == "__main__":
    main()
