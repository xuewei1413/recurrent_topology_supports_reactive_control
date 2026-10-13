"""
v5 固定接口的匹配连接对照（修复 v4 缺陷 F1）

v4 的 matched_random 从全网络随机删除与 noHH 相同数量的边。
由于 HH 边占全部边的 92-94%，随机删除几乎必然摧毁 input→hidden
和 hidden→output 接口，因此 "noHH > matched_random" 只能证明
"保留接口完整性更好"，不能证明 HH 边有超越边数的特殊贡献。

本模块的所有对照条件严格固定 I→H 和 H→O 接口，只在 HH 子图内操作：

  - delete_hh_edges_only:      仅从 HH 子图随机删除指定数量的边
  - rewire_hh_preserving_degree: 保持 HH 子图出入度分布的重连（Maslov-Sneppen）
  - make_hh_dag:               HH 子图无环化（按随机拓扑序删除反向边）

邻接矩阵约定（与 core.weight_construction 一致）：
  A[j, i] = 1 表示存在 i → j 的边（j = 目标/突触后，i = 源/突触前）。
  行和 = 入度，列和 = 出度。
"""

from __future__ import annotations

import numpy as np


def hh_bounds(n_total: int, n_in: int, n_out: int) -> tuple[int, int]:
    """返回 hidden 神经元的索引区间 [h_start, h_end)。"""
    return n_in, n_total - n_out


def interface_counts(adj: np.ndarray, n_in: int, n_out: int) -> dict:
    """统计 I→H、H→O、HH 子图边数，用于接口固定审计。"""
    n_total = adj.shape[0]
    h_start, h_end = hh_bounds(n_total, n_in, n_out)
    return {
        "n_ih_edges": int(adj[h_start:h_end, :n_in].sum()),      # I → H
        "n_ho_edges": int(adj[n_total - n_out:, h_start:h_end].sum()),  # H → O
        "n_hh_edges": int(adj[h_start:h_end, h_start:h_end].sum()),     # H → H
        "n_total_edges": int(adj.sum()),
    }


def check_interface_preserved(
    adj_ref: np.ndarray, adj_new: np.ndarray, n_in: int, n_out: int
) -> bool:
    """验证 I→H 和 H→O 接口与参考矩阵完全一致。"""
    n_total = adj_ref.shape[0]
    h_start, h_end = hh_bounds(n_total, n_in, n_out)
    ih_same = np.array_equal(
        adj_new[h_start:h_end, :n_in], adj_ref[h_start:h_end, :n_in]
    )
    ho_same = np.array_equal(
        adj_new[n_total - n_out:, h_start:h_end],
        adj_ref[n_total - n_out:, h_start:h_end],
    )
    # 非接口、非 HH 区域（如 I→O 直连，若存在）也必须不变
    rest_ref = adj_ref.copy()
    rest_new = adj_new.copy()
    rest_ref[h_start:h_end, h_start:h_end] = 0
    rest_new[h_start:h_end, h_start:h_end] = 0
    rest_same = np.array_equal(rest_ref, rest_new)
    return bool(ih_same and ho_same and rest_same)


def delete_hh_edges_only(
    adj: np.ndarray, n_in: int, n_out: int, n_delete: int, rng: np.random.RandomState
) -> np.ndarray:
    """仅从 HH 子图中随机删除指定数量的边，保留所有 I→H 和 H→O。

    注意：当 n_delete == HH 子图边数时，结果与 noHH 完全相同（删除全部 HH
    边是确定性的，与随机选择无关）。
    """
    h_start, h_end = hh_bounds(adj.shape[0], n_in, n_out)
    adj_out = adj.copy()
    hh_edges = np.argwhere(adj[h_start:h_end, h_start:h_end] > 0)
    n_delete = min(int(n_delete), len(hh_edges))
    if n_delete <= 0:
        return adj_out
    del_idx = rng.choice(len(hh_edges), size=n_delete, replace=False)
    for idx in del_idx:
        r, c = hh_edges[idx]
        adj_out[h_start + r, h_start + c] = 0.0
    return adj_out


def rewire_hh_preserving_degree(
    adj: np.ndarray,
    n_in: int,
    n_out: int,
    rng: np.random.RandomState,
    n_swap_attempts: int | None = None,
) -> np.ndarray:
    """保持 HH 子图每个神经元出入度不变的随机重连。

    使用 Maslov-Sneppen 度保持边交换：随机取两条 HH 边 (t1←s1) 和
    (t2←s2)，交换目标为 (t1←s2) 和 (t2←s1)。每次交换同时保持
    行和（入度）与列和（出度）。拒绝产生自环或重复边的交换。

    所有 I→H 和 H→O 接口严格不变。
    """
    h_start, h_end = hh_bounds(adj.shape[0], n_in, n_out)
    adj_out = adj.copy()
    hh = adj_out[h_start:h_end, h_start:h_end]

    edges = np.argwhere(hh > 0)  # (n_edges, 2)，每行 (t, s)
    n_edges = len(edges)
    if n_edges < 2:
        return adj_out

    edge_set = set(map(tuple, edges.tolist()))
    if n_swap_attempts is None:
        n_swap_attempts = max(20 * n_edges, 2000)

    n_h = hh.shape[0]
    n_swapped = 0
    for _ in range(n_swap_attempts):
        i1, i2 = rng.choice(n_edges, size=2, replace=False)
        t1, s1 = edges[i1]
        t2, s2 = edges[i2]
        # 交换目标：(t1,s1),(t2,s2) → (t1,s2),(t2,s1)
        if t1 == s2 or t2 == s1:
            continue  # 会产生自环
        if (t1, s2) in edge_set or (t2, s1) in edge_set:
            continue  # 会产生重复边
        edge_set.discard((t1, s1))
        edge_set.discard((t2, s2))
        edge_set.add((t1, s2))
        edge_set.add((t2, s1))
        edges[i1] = (t1, s2)
        edges[i2] = (t2, s1)
        n_swapped += 1

    new_hh = np.zeros((n_h, n_h))
    for t, s in edges:
        new_hh[t, s] = 1.0
    adj_out[h_start:h_end, h_start:h_end] = new_hh
    return adj_out


def make_hh_dag(
    adj: np.ndarray, n_in: int, n_out: int, rng: np.random.RandomState
) -> np.ndarray:
    """将 HH 子图改为 DAG（有向无环图），保持边数尽可能接近原图。

    对 hidden 神经元做随机拓扑排序，只保留排序方向一致的边
    （order[t] > order[s] 时保留 s→t），其余丢弃。结果必然无环。

    所有 I→H 和 H→O 接口严格不变。
    """
    h_start, h_end = hh_bounds(adj.shape[0], n_in, n_out)
    n_h = h_end - h_start
    adj_out = adj.copy()

    order = rng.permutation(n_h)
    hh = adj[h_start:h_end, h_start:h_end]
    hh_edges = np.argwhere(hh > 0)  # (t, s)

    new_hh = np.zeros((n_h, n_h))
    for t, s in hh_edges:
        if order[t] > order[s]:  # 只保留与拓扑序同向的边
            new_hh[t, s] = 1.0

    adj_out[h_start:h_end, h_start:h_end] = new_hh
    return adj_out


def is_dag(adj: np.ndarray) -> bool:
    """检查（子）图是否为 DAG（Kahn 拓扑排序）。"""
    n = adj.shape[0]
    # 注意：A[t, s]=1 表示 s→t。入度 = 行和；Kahn 从入度为 0 的节点开始，
    # 沿列方向（出边）消减目标的入度。
    in_deg = (adj > 0).sum(axis=1).astype(int).copy()
    queue = [i for i in range(n) if in_deg[i] == 0]
    seen = 0
    adj_bin = (adj > 0).astype(int)
    while queue:
        node = queue.pop()
        seen += 1
        # node 作为源（列），其所有目标行入度减 1
        targets = np.where(adj_bin[:, node] > 0)[0]
        for t in targets:
            in_deg[t] -= 1
            if in_deg[t] == 0:
                queue.append(t)
    return seen == n


def audit_condition(
    adj_ref: np.ndarray, adj_new: np.ndarray, n_in: int, n_out: int
) -> dict:
    """审计一个对照条件：接口是否固定、HH 边数、度数变化。"""
    ref = interface_counts(adj_ref, n_in, n_out)
    new = interface_counts(adj_new, n_in, n_out)
    n_total = adj_ref.shape[0]
    h_start, h_end = hh_bounds(n_total, n_in, n_out)
    return {
        "interface_preserved": check_interface_preserved(adj_ref, adj_new, n_in, n_out),
        "n_hh_edges_ref": ref["n_hh_edges"],
        "n_hh_edges_new": new["n_hh_edges"],
        "n_total_edges_new": new["n_total_edges"],
        "hh_out_degree_mean_ref": float(adj_ref[h_start:h_end, h_start:h_end].sum(axis=0).mean()),
        "hh_out_degree_mean_new": float(adj_new[h_start:h_end, h_start:h_end].sum(axis=0).mean()),
        "hh_in_degree_mean_ref": float(adj_ref[h_start:h_end, h_start:h_end].sum(axis=1).mean()),
        "hh_in_degree_mean_new": float(adj_new[h_start:h_end, h_start:h_end].sum(axis=1).mean()),
    }
