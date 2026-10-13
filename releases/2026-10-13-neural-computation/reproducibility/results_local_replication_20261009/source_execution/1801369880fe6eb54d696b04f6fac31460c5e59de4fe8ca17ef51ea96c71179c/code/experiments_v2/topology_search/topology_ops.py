"""
固定边预算拓扑操作（Phase 2）

设计文档 §6：
  - genotype 为满足允许连接掩码的二值邻接矩阵
  - 每次变异删除 k 条已有边并增加 k 条不存在的边，保持边数不变
  - k 采用 1 + Poisson(2)，截断到 1--8
  - 主搜索使用 mutation-only，取消逐边 uniform crossover

关键约束：
  1. 任何变异后边数不变（删 k + 加 k）
  2. 自连接为零
  3. 非法边为零（必须满足 connection_mask）
  4. k 次 edge swap 的删除数和新增数严格匹配
"""

from __future__ import annotations

import numpy as np

from ..core.connection_mask import build_allowed_mask


# ── 固定边数随机初始化 ────────────────────────────────────────────────

def random_fixed_edge_topology(
    allowed_mask: np.ndarray,
    edge_count: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """从允许位置中均匀随机选取 edge_count 条边。

    Args:
        allowed_mask: (n, n) bool，True = 允许该连接
        edge_count:   目标边数
        rng:          随机数生成器

    Returns:
        (n, n) float64 二值邻接矩阵，恰好有 edge_count 条边
    """
    n = allowed_mask.shape[0]
    allowed_positions = np.argwhere(allowed_mask)  # (n_allowed, 2) [row, col]
    n_allowed = len(allowed_positions)

    if edge_count > n_allowed:
        raise ValueError(
            f"edge_count {edge_count} exceeds allowed {n_allowed}"
        )
    if edge_count < 0:
        raise ValueError(f"edge_count {edge_count} < 0")

    chosen = rng.choice(n_allowed, size=edge_count, replace=False)
    adjacency = np.zeros((n, n), dtype=np.float64)
    for idx in chosen:
        r, c = allowed_positions[idx]
        adjacency[r, c] = 1.0

    return adjacency


# ── k-edge-swap 变异（保持边数不变）────────────────────────────────────

def sample_k(rng: np.random.RandomState, lam: float = 2.0) -> int:
    """采样 k = 1 + Poisson(lam)，截断到 [1, 8]。

    设计文档 §6：k 采用 1 + Poisson(2)，截断到 1--8。
    """
    k = 1 + rng.poisson(lam)
    return int(np.clip(k, 1, 8))


def mutate_edge_swap(
    adjacency: np.ndarray,
    allowed_mask: np.ndarray,
    k: int | None = None,
    rng: np.random.RandomState | None = None,
    lam: float = 2.0,
) -> np.ndarray:
    """删除 k 条已有边并增加 k 条不存在的边，保持边数不变。

    确保新增加的边在 allowed_mask 内且不是自连接。
    如果 allowed_mask 内无足够的空位（边数接近饱和），会尽可能操作。

    Args:
        adjacency:    (n, n) 二值邻接矩阵
        allowed_mask: (n, n) bool 允许掩码
        k:            swap 次数；None 则自动采样
        rng:          随机数生成器
        lam:          Poisson 参数（仅 k=None 时使用）

    Returns:
        新的 (n, n) 邻接矩阵，边数与输入相同
    """
    if rng is None:
        rng = np.random.RandomState()

    child = adjacency.copy()
    n = child.shape[0]

    # 当前活跃边和不存在的允许边
    active = child > 0
    existing_positions = np.argwhere(active)           # (n_exist, 2)
    empty_allowed = allowed_mask & ~active              # 允许但当前不存在的边
    empty_positions = np.argwhere(empty_allowed)        # (n_empty, 2)

    n_exist = len(existing_positions)
    n_empty = len(empty_positions)

    if k is None:
        k = sample_k(rng, lam)

    # 实际可操作的 k 不超过 min(n_exist, n_empty)
    k_actual = min(k, n_exist, n_empty)

    if k_actual == 0:
        return child  # 无法操作

    # 随机选 k_actual 条已有边删除
    del_indices = rng.choice(n_exist, size=k_actual, replace=False)
    # 随机选 k_actual 条空位添加
    add_indices = rng.choice(n_empty, size=k_actual, replace=False)

    for idx in del_indices:
        r, c = existing_positions[idx]
        child[r, c] = 0.0

    for idx in add_indices:
        r, c = empty_positions[idx]
        child[r, c] = 1.0

    # 双重保险
    np.fill_diagonal(child, 0.0)
    child[~allowed_mask] = 0.0

    return child


# ── 种群操作 ──────────────────────────────────────────────────────────

def init_population(
    pop_size: int,
    allowed_mask: np.ndarray,
    edge_count: int,
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    """初始化种群：每个个体是独立的随机固定边数拓扑。"""
    return [
        random_fixed_edge_topology(allowed_mask, edge_count, rng)
        for _ in range(pop_size)
    ]


def tournament_select(
    pop: list[np.ndarray],
    fitnesses: np.ndarray,
    tournament_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """锦标赛选择：随机选 tournament_size 个个体，返回最优的。"""
    n = len(pop)
    candidates = rng.choice(n, size=min(tournament_size, n), replace=False)
    best = candidates[np.argmax(fitnesses[candidates])]
    return pop[best]


# ── 验证工具 ──────────────────────────────────────────────────────────

def verify_edge_count(adjacency: np.ndarray, expected: int) -> bool:
    """验证边数恰好等于 expected。"""
    actual = int(adjacency.sum())
    return actual == expected


def verify_topology_invariants(
    adjacency: np.ndarray,
    allowed_mask: np.ndarray,
    expected_edge_count: int,
) -> tuple[bool, str]:
    """验证拓扑的所有不变量。

    Returns:
        (ok, message)
    """
    n = adjacency.shape[0]

    # 边数
    actual_edges = int(adjacency.sum())
    if actual_edges != expected_edge_count:
        return False, f"edge_count {actual_edges} != {expected_edge_count}"

    # 无自连接
    diag_sum = int(np.diag(adjacency).sum())
    if diag_sum != 0:
        return False, f"self-connections: {diag_sum}"

    # 无非法边
    illegal = (adjacency > 0) & ~allowed_mask
    n_illegal = int(illegal.sum())
    if n_illegal != 0:
        return False, f"illegal edges: {n_illegal}"

    # 二值
    unique_vals = set(np.unique(adjacency).tolist())
    if not unique_vals.issubset({0.0, 1.0}):
        return False, f"non-binary values: {unique_vals}"

    return True, "ok"
