"""E5b 精确匹配支持域 dry-run —— 第六轮修订 P0-2 后续（纯图计算，无行为评估）。

第六轮指出骨架分数退化为 score(s→d) = 0.25·pin1(s)·pout1(d)（三跳桥接计数），
而 E5 匹配特征（符号/源模块/目标模块/源全出度/目标全入度）**恰好未匹配
pin1/pout1 这两个定义因子**。E5b 要求精确匹配全部定义因子+端点度。

本脚本先做零行为评估的可行性 dry-run：对每个 cell，检查每条骨架边在
非骨架池中是否存在全部特征精确匹配的候选（共同支持域）。
  K1 = (符号, 源模块8, 目标模块8, pin1, pout1, 源全出度, 目标全入度)
  K2 = 同上但度限 HH 子图
  K3 = (符号, 源模块8, 目标模块8, pin1, pout1)（松弛：不限度）
并做贪心可匹配性检查（每条骨架边需占一个不同候选）。

若共同支持域缺失/稀薄 → 如实报告"精确匹配在多数 cell 不可行"（本身是发现：
骨架边的 pin1×pout1 组合在非骨架池中稀缺，即骨架位置具有端点属性不可
复制的特殊性）；若支持充分 → 进入全量 E5b 行为评估。

队列：e1 60 cells（CP）+ main CP 20 + c2 CP 15 + main Acr 20。
产出：results_v7/e5b_support/e5b_support_{cohort}_{task}_{tag}.json + _summary.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol  # noqa: E402
from experiments_v2.core.scaffold import NeuronScaffold  # noqa: E402
from experiments_v2.mechanisms.run_causal_propagation_v5 import (  # noqa: E402
    load_v3_topology)
from experiments_v2.mechanisms.run_skeleton_v6 import (  # noqa: E402
    edge_contributions, SKEL_FRAC)

K_MODULES = 8


def cell_features(f0_adj, scaffold, n_in, n_out):
    """全部 HH 边的特征表 + 骨架标记。"""
    n = f0_adj.shape[0]
    hid = np.arange(n_in, n - n_out)
    A = (f0_adj != 0)
    pin1 = A[:, :n_in].sum(axis=1)          # pin1[i] = 输入→i 边数
    pout1 = A[n - n_out:, :].sum(axis=0)    # pout1[j] = j→输出 边数
    full_out = A.sum(axis=0)                # 源全出度（与 E5 同定义）
    full_in = A.sum(axis=1)                 # 目标全入度
    hh_out = A[np.ix_(hid, hid)].sum(axis=0)
    hh_in = A[np.ix_(hid, hid)].sum(axis=1)
    mod = scaffold.module_labels_for_k(K_MODULES)
    ntype = scaffold.neuron_types

    edges, contrib = edge_contributions(f0_adj, n_in, n_out)
    n_del = max(1, int(round(SKEL_FRAC * len(edges))))
    order = np.argsort(-contrib)
    skel_set = {tuple(map(int, e)) for e in edges[order[:n_del]]}

    feats = {}
    for e, c in zip(edges, contrib):
        s, d = int(e[0]), int(e[1])
        hi, lo = s - n_in, d - n_in
        feats[(s, d)] = {
            "sign": int(np.sign(ntype[s])),
            "src_mod": int(mod[s]), "dst_mod": int(mod[d]),
            "pin1": int(pin1[s]), "pout1": int(pout1[d]),
            "full_out": int(full_out[s]), "full_in": int(full_in[d]),
            "hh_out": int(hh_out[hi]), "hh_in": int(hh_in[lo]),
            "contrib": float(c),
        }
    return feats, skel_set, n_del


def key_of(f, kind):
    if kind == "K1":
        return (f["sign"], f["src_mod"], f["dst_mod"], f["pin1"], f["pout1"],
                f["full_out"], f["full_in"])
    if kind == "K2":
        return (f["sign"], f["src_mod"], f["dst_mod"], f["pin1"], f["pout1"],
                f["hh_out"], f["hh_in"])
    return (f["sign"], f["src_mod"], f["dst_mod"], f["pin1"], f["pout1"])


def support_report(feats, skel_set, kind):
    pool = [e for e in feats if e not in skel_set]
    pool_keys = Counter(key_of(feats[e], kind) for e in pool)
    n_with, sizes = 0, []
    for e in skel_set:
        c = pool_keys.get(key_of(feats[e], kind), 0)
        n_with += int(c > 0)
        sizes.append(c)
    # 贪心可匹配性（每条骨架边占一个不同候选）
    remaining = dict(pool_keys)
    greedy_ok = 0
    for e in sorted(skel_set, key=lambda x: pool_keys.get(key_of(feats[x], kind), 0)):
        k = key_of(feats[e], kind)
        if remaining.get(k, 0) > 0:
            remaining[k] -= 1
            greedy_ok += 1
    return {"kind": kind, "n_skel": len(skel_set),
            "frac_with_support": n_with / max(len(skel_set), 1),
            "support_size_med": float(np.median(sizes)) if sizes else 0.0,
            "support_size_max": int(max(sizes)) if sizes else 0,
            "greedy_matchable_frac": greedy_ok / max(len(skel_set), 1)}


def run_cell(cohort, task_name, f0_adj, scaffold_seed, tag, out_dir, extra):
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    scaffold = NeuronScaffold.from_seed(seed=scaffold_seed, task=task,
                                        controller=proto)
    n_in, n_out = task.n_input, task.n_output
    t0 = time.time()
    feats, skel_set, n_del = cell_features(f0_adj, scaffold, n_in, n_out)
    rec = {"version": "v7_e5b_support", "cohort": cohort, "task": task_name,
           "tag": tag, "scaffold_seed": scaffold_seed,
           "n_hh": len(feats), "n_skel": n_del, "skel_frac": SKEL_FRAC,
           "k_modules": K_MODULES,
           "reports": [support_report(feats, skel_set, k)
                       for k in ("K1", "K2", "K3")],
           "elapsed_sec": time.time() - t0}
    rec.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"e5b_support_{cohort}_{task_name}_{tag}.json",
              "w") as f:
        json.dump(rec, f, indent=1)
    r1, r2, r3 = rec["reports"]
    print(f"[{cohort} {task_name} {tag}] n_hh={rec['n_hh']} skel={n_del} "
          f"K1 sup={r1['frac_with_support']:.2f}/greedy={r1['greedy_matchable_frac']:.2f} "
          f"K2 sup={r2['frac_with_support']:.2f} "
          f"K3 sup={r3['frac_with_support']:.2f}/greedy={r3['greedy_matchable_frac']:.2f}",
          flush=True)
    return rec


def main():
    out_dir = ROOT / "results_v7/e5b_support"
    recs = []
    # E1 60 cells（CP）
    for sf in range(7000, 7010):
        for ga in (7100, 7101, 7102):
            for b in (20000, 40000):
                tag = f"sf{sf}_ga{ga}_b{b}"
                p = (ROOT / "results_v3/e1_variance/CartPole" / tag /
                     f"e1_F0_topo_CartPole_{tag}.npy")
                if not p.exists():
                    print(f"missing {p}", flush=True)
                    continue
                recs.append(run_cell("e1", "CartPole", np.load(p), sf, tag,
                                     out_dir,
                                     {"ga_seed": ga, "budget": b}))
    # main CP + Acr（3000-3019）与 c2 CP（3020-3034）
    for task_name in ("CartPole", "Acrobot"):
        for seed in range(3000, 3020):
            adj = load_v3_topology(task_name, seed, str(ROOT), "topology")
            if adj is None:
                print(f"missing {task_name} {seed}", flush=True)
                continue
            recs.append(run_cell("main", task_name, adj, seed, f"seed{seed}",
                                 out_dir, {"evo_seed": seed}))
    for seed in range(3020, 3035):
        adj = load_v3_topology("CartPole", seed, str(ROOT), "topology_c2")
        if adj is None:
            print(f"missing CartPole {seed} c2", flush=True)
            continue
        recs.append(run_cell("c2", "CartPole", adj, seed, f"seed{seed}",
                             out_dir, {"evo_seed": seed}))
    # 汇总
    summary = {}
    for rec in recs:
        ck = (rec["cohort"], rec["task"])
        summary.setdefault(str(ck), []).append({
            r["kind"]: (r["frac_with_support"], r["greedy_matchable_frac"])
            for r in rec["reports"]})
    agg = {}
    for ck, rows in summary.items():
        agg[ck] = {k: {"frac_with_support_mean": float(np.mean([x[k][0] for x in rows])),
                       "greedy_matchable_mean": float(np.mean([x[k][1] for x in rows])),
                       "frac_with_support_min": float(np.min([x[k][0] for x in rows])),
                       "n_cells": len(rows)}
                   for k in ("K1", "K2", "K3")}
    with open(out_dir / "_summary.json", "w") as f:
        json.dump(agg, f, indent=1)
    print(json.dumps(agg, indent=1), flush=True)


if __name__ == "__main__":
    main()
