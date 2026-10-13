"""损伤指纹 v2（prereg 修订案 v5 仪器）——修 v1 三缺陷 + 冗余家族超加性检验。

相对 v1 的修正：
  1. 全 HH 槽位空间固定分组：hidden×hidden 平面按索引均匀切 8×4=32 组，
     所有条件共用同一分组（v1 只对 F0 现存边 k-means，组随条件漂移、大小不等）。
  2. (g,τ) 分辨指纹：L[g,τ] = mean_states Σ_actions |margin_les − margin_base|，
     不再对 τ 求和；S = cosine(flatten(L_variant), flatten(L_F0))。
  3. 自有状态重放：每个变体用自己的 episode 校准态（v1 全用 F0 态）。
  4. 双组损伤超加性（冗余路径家族检验）：单组损伤 D_g = Σ_τ L[g,τ]，
     分层抽 16 对：8 对最低伤组（遮蔽区：单伤被补偿，超加性 I>0 为冗余签名）
     + 8 对最高伤组（天花板区对照）。I_pair = D_joint − D_1 − D_2。
     预测：preserved 遮蔽区 I 显著高于 destroyed（冗余家族被摧毁后补偿消失）。

模式（种子池仅 3000-3019 可用，队列内分层）：
  pilot   : seeds 3000-3004，F0 split-half retest + preserved/destroyed 分离
            → 冻结 SESOI_S_v2 与 SESOI_I_v2（pilot 拓扑仅作 discovery）
  measure : seeds 3005-3019（confirmatory 队列），F0 + 12 重建条件（同 v1 种子）

产出：results_v6/lesion_v2/lesion_v2_{task}_{seed}.json（pilot 前缀 lesion_v2p_）
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
from experiments_v2.mechanisms.run_lesion_fingerprint_v6 import (  # noqa: E402
    margin_vec, reconstruct_variants, TAU_MAX, H_FRAC, N_CAND, N_SEARCH_SEEDS,
)

G_SRC, G_DST = 8, 4
G_GROUPS = G_SRC * G_DST          # 32
N_PAIRS = 16
TOPK_PAIR = 8


def fixed_groups(adj, n_in, n_out):
    """全 HH 槽位 8×4 固定网格分组。返回 group_edges: list[G] of (m_g,2) 现存边。"""
    n = adj.shape[0]
    hid = np.arange(n_in, n - n_out)
    src_bins = np.array_split(hid, G_SRC)
    dst_bins = np.array_split(hid, G_DST)
    gid = np.zeros(n, int)
    for bi, b in enumerate(src_bins):
        gid[b] = bi
    edges = hh_edge_list(adj, n_in, n_out)          # (m,2) [src,dst] 现存边
    g_of_edge = gid[edges[:, 0]] * G_DST + gid[edges[:, 1]]
    return [edges[g_of_edge == g] for g in range(G_GROUPS)]


def replay_margins(adj, scaffold, enc, proto, episodes, states, lesion_edges):
    """从 v 快照重放 τ=0..TAU_MAX，返回 (n_states, TAU+1, n_out)。"""
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
        clone.v[:] = ep["snaps"][t][0]
        seq = ep["seq"]
        for k in range(TAU_MAX + 1):
            _, acc, _ = clone.forward(seq[t + k], return_trace=True)
            out[si, k] = margin_vec(acc, ctrl_l.n_out)
    return out


def fingerprint(adj, scaffold, enc, proto, episodes, states, group_edges,
                base_margins=None):
    """L[g,τ]（不对 τ 求和）+ 各组现存边数。返回 (L, base_margins, n_edges)。"""
    if base_margins is None:
        base_margins = replay_margins(adj, scaffold, enc, proto,
                                      episodes, states, None)
    L = np.zeros((G_GROUPS, TAU_MAX))
    n_edges = np.zeros(G_GROUPS, int)
    for g in range(G_GROUPS):
        eg = group_edges[g]
        n_edges[g] = len(eg)
        if len(eg) == 0:
            continue
        m_les = replay_margins(adj, scaffold, enc, proto, episodes, states,
                               [tuple(e) for e in eg])
        diff = np.abs(m_les[:, 1:, :] - base_margins[:, 1:, :])
        L[g] = diff.sum(axis=2).mean(axis=0)        # (TAU,) 每 τ 的 mean_states
    return L, base_margins, n_edges


def joint_interaction(adj, scaffold, enc, proto, episodes, states,
                      group_edges, D_single, base_margins, rng):
    """分层抽对联合损伤：8 对最低伤组（遮蔽区，超加性检验）+ 8 对最高伤组
    （天花板区，对照审计）。低伤组要求组内有边（可损伤）。"""
    has_edge = [g for g in range(G_GROUPS) if len(group_edges[g]) > 0]
    low = sorted(has_edge, key=lambda g: D_single[g])[:16]
    high = sorted(has_edge, key=lambda g: -D_single[g])[:TOPK_PAIR]

    def sample(pool, k, regime):
        all_pairs = [(pool[i], pool[j]) for i in range(len(pool))
                     for j in range(i + 1, len(pool))]
        if not all_pairs:
            return []
        idx = rng.choice(len(all_pairs), size=min(k, len(all_pairs)),
                         replace=False)
        return [(all_pairs[i], regime) for i in idx]

    pairs = sample(low, N_PAIRS // 2, "low_masked") + \
        sample(high, N_PAIRS // 2, "high_ceiling")
    out = []
    for (g1, g2), regime in pairs:
        eg = [tuple(e) for e in np.vstack([group_edges[g1], group_edges[g2]])]
        m_les = replay_margins(adj, scaffold, enc, proto, episodes, states, eg)
        diff = np.abs(m_les[:, 1:, :] - base_margins[:, 1:, :])
        D_joint = float(diff.sum(axis=(1, 2)).mean())
        D1, D2 = float(D_single[g1]), float(D_single[g2])
        out.append({"g1": int(g1), "g2": int(g2), "regime": regime,
                    "D_joint": D_joint, "D1": D1, "D2": D2,
                    "I": D_joint - D1 - D2,
                    "I_norm": (D_joint - D1 - D2) / (D1 + D2 + 1e-12)})
    return out


def one_network(adj, scaffold, enc, proto, task, n_states, calib_rng,
                pair_rng, env=None):
    """单个网络：自有状态校准 + 指纹 + 双组交互。
    返回 (L, D, pairs, n_edges, n_states_used)；可用态 <10 时返回 None 组。"""
    from experiments_v2.tasks.adapter import make_env
    ctrl = UnifiedSNNController(scaffold, adj, enc, proto, condition=F0)
    own_env = env if env is not None else make_env(task)
    episodes, states = collect_calibration(ctrl, own_env, task, n_states, calib_rng)
    if env is None:
        own_env.close()
    # 崩塌变体 episode 短，需保证 t+TAU_MAX 在序列内
    states = [st for st in states
              if st[1] + TAU_MAX < len(episodes[st[0]]["seq"])]
    if len(states) < 10:
        return None, None, [], None, len(states)
    ge = fixed_groups(adj, task.n_input, task.n_output)
    L, base, n_edges = fingerprint(adj, scaffold, enc, proto, episodes, states, ge)
    D = L.sum(axis=1)                               # (G,) 单组总损伤
    pairs = joint_interaction(adj, scaffold, enc, proto, episodes, states,
                              ge, D, base, pair_rng)
    return L, D, pairs, n_edges, len(states)


def run(task_name, evo_seed, mode, n_states, out_dir, topo_subdir="topology"):
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/observation_stats_{task_name}.json"))
    f0_adj = load_v3_topology(task_name, evo_seed, str(ROOT), topo_subdir)
    if f0_adj is None:
        print(f"[{task_name} {evo_seed}] topology missing", flush=True)
        return
    scaffold = NeuronScaffold.from_seed(seed=evo_seed, task=task, controller=proto)
    t0 = time.time()
    n_in, n_out = task.n_input, task.n_output

    rec = {"version": "v6_lesion_v2", "task": task_name, "seed": evo_seed,
           "G": G_GROUPS, "tau_max": TAU_MAX, "n_states": n_states,
           "mode": mode, "conditions": []}

    # F0：自有状态 + 指纹 + retest
    L_F0, D_F0, pairs_F0, ne_F0, ns_F0 = one_network(
        f0_adj, scaffold, enc, proto, task, n_states,
        np.random.RandomState(2026), np.random.RandomState(evo_seed * 431 + 7))
    rec["F0_n_states_used"] = ns_F0
    rec["L_F0_norm"] = float(np.linalg.norm(L_F0)) if L_F0 is not None else None
    rec["F0_D_total"] = float(D_F0.sum()) if D_F0 is not None else None
    rec["F0_pairs"] = pairs_F0
    rec["F0_n_edges_per_group"] = ne_F0.tolist() if ne_F0 is not None else None

    # split-half retest（仪器噪声下限，pilot 冻结用）
    from experiments_v2.tasks.adapter import make_env
    ctrl = UnifiedSNNController(scaffold, f0_adj, enc, proto, condition=F0)
    env = make_env(task)
    episodes, states = collect_calibration(ctrl, env, task, n_states,
                                           np.random.RandomState(2026))
    env.close()
    states = [st for st in states
              if st[1] + TAU_MAX < len(episodes[st[0]]["seq"])]
    ge = fixed_groups(f0_adj, n_in, n_out)
    rng_qc = np.random.RandomState(7)
    tr = []
    for rep in range(2):
        perm = rng_qc.permutation(len(states))
        h1 = [states[i] for i in perm[: len(states) // 2]]
        h2 = [states[i] for i in perm[len(states) // 2:]]
        L1, _, _ = fingerprint(f0_adj, scaffold, enc, proto, episodes, h1, ge)
        L2, _, _ = fingerprint(f0_adj, scaffold, enc, proto, episodes, h2, ge)
        c = cosine(L1.ravel(), L2.ravel())
        if c is not None:
            tr.append(c)
    rec["F0_retest"] = tr
    rec["F0_retest_mean"] = float(np.mean(tr)) if tr else None
    print(f"[{mode} {task_name} {evo_seed}] F0 retest={rec['F0_retest_mean']} "
          f"D_total={rec['F0_D_total']} ns={ns_F0} ({(time.time()-t0)/60:.1f} min)",
          flush=True)

    # 12 重建条件（同 v1 种子）
    variants = reconstruct_variants(f0_adj, n_in, n_out, evo_seed)
    for v in variants:
        for cond in ("preserved", "destroyed", "rescue", "rescue_ctrl"):
            adj_v = v[cond]
            L_v, D_v, pairs_v, _, ns_v = one_network(
                adj_v, scaffold, enc, proto, task, n_states,
                np.random.RandomState(2026), np.random.RandomState(evo_seed * 431 + 7))
            I_vals = [p["I"] for p in pairs_v]
            I_low = [p["I"] for p in pairs_v if p.get("regime") == "low_masked"]
            I_high = [p["I"] for p in pairs_v if p.get("regime") == "high_ceiling"]
            S_v = (cosine(L_v.ravel(), L_F0.ravel())
                   if (L_v is not None and L_F0 is not None) else None)
            rec["conditions"].append({
                "search_seed": v["search_seed"], "condition": cond,
                "S_lesion": S_v,
                "D_total": float(D_v.sum()) if D_v is not None else None,
                "n_states_used": ns_v,
                "I_mean": float(np.mean(I_vals)) if I_vals else None,
                "I_median": float(np.median(I_vals)) if I_vals else None,
                "I_frac_pos": float(np.mean(np.array(I_vals) > 0)) if I_vals else None,
                "I_low_mean": float(np.mean(I_low)) if I_low else None,
                "I_low_median": float(np.median(I_low)) if I_low else None,
                "I_low_frac_pos": float(np.mean(np.array(I_low) > 0)) if I_low else None,
                "I_high_mean": float(np.mean(I_high)) if I_high else None,
                "pairs": pairs_v,
            })
        cs = rec["conditions"][-4:]
        def _f(x, nd=4):
            return f"{x:.{nd}f}" if x is not None else "None"
        print(f"[{task_name} {evo_seed} s{v['search_seed']}] "
              f"S p={_f(cs[0]['S_lesion'],3)} d={_f(cs[1]['S_lesion'],3)} "
              f"r={_f(cs[2]['S_lesion'],3)} rc={_f(cs[3]['S_lesion'],3)} | "
              f"Ilow p={_f(cs[0]['I_low_median'])} d={_f(cs[1]['I_low_median'])} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    rec["minutes"] = (time.time() - t0) / 60
    op = out_dir / (f"lesion_v2p_{task_name}_{evo_seed}.json" if mode == "pilot"
                    else f"lesion_v2_{task_name}_{evo_seed}.json")
    with open(op, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved {op} ({rec['minutes']:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["pilot", "measure"], default="measure")
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--topo-subdir", default="topology",
                    help="results_v3 下的拓扑目录；c2 队列用 topology_c2")
    ap.add_argument("--out-dir", default="results_v6/lesion_v2",
                    help="ROOT 下的相对输出目录")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    op = out_dir / (f"lesion_v2p_{args.task}_{args.seed}.json" if args.mode == "pilot"
                    else f"lesion_v2_{args.task}_{args.seed}.json")
    if op.exists():
        with open(op) as f:
            d = json.load(f)
        if len(d.get("conditions", [])) >= 12:
            print("done, skip", flush=True)
            return
    run(args.task, args.seed, args.mode, args.n_states, out_dir, args.topo_subdir)


if __name__ == "__main__":
    main()
