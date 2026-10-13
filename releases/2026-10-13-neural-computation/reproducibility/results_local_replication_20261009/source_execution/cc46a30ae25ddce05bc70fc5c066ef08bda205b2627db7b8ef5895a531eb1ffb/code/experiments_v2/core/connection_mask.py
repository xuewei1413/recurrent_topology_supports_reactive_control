"""
允许连接掩码（Phase 1）

设计文档 §4：
  - 禁止自连接
  - 允许的其他连接方向必须在配置中显式声明并全任务一致

本实现采用四任务统一的方向策略：
  允许：input→hidden, hidden→hidden, hidden→output
  禁止：input→input, input→output, output→*, *→input, 自连接

理由：
  - input→output 直连跳过传播层，违背"多步传播"因果链（设计文档 §2）
  - output 作为源没有下游消费者
  - input 作为目标没有上游生物学意义
"""

from __future__ import annotations

import numpy as np


def build_allowed_mask(n_total: int, n_input: int, n_output: int) -> np.ndarray:
    """返回 (n_total, n_total) 的 bool 矩阵。

    mask[i, j] = True 表示允许从源神经元 i 到目标神经元 j 的连接。
    约定：W[j, i] 表示 i→j 的连接，故邻接矩阵的 [j, i] 元素对应 mask[j, i]。
    本函数返回的 mask 与邻接矩阵同形同索引，即 mask[j, i] 对应 i→j。

    实际操作中，掩码直接作用于邻接矩阵元素：adjacency[j, i] & mask[j, i]。
    """
    mask = np.zeros((n_total, n_total), dtype=bool)

    hidden_start = n_input
    hidden_end = n_total - n_output

    # input (源) → hidden (目标)
    mask[hidden_start:hidden_end, :n_input] = True

    # hidden (源) → hidden (目标)
    mask[hidden_start:hidden_end, hidden_start:hidden_end] = True

    # hidden (源) → output (目标)
    mask[hidden_end:, hidden_start:hidden_end] = True

    # 禁止自连接
    np.fill_diagonal(mask, False)

    return mask


def is_valid_topology(adjacency: np.ndarray, allowed_mask: np.ndarray) -> bool:
    """检查邻接矩阵是否只包含允许的连接且无自连接。"""
    if adjacency.shape != allowed_mask.shape:
        return False
    # 不允许的边不应存在
    illegal = adjacency.astype(bool) & ~allowed_mask
    if illegal.any():
        return False
    # 无自连接
    if np.diag(adjacency).sum() != 0:
        return False
    return True


def enforce_mask(adjacency: np.ndarray, allowed_mask: np.ndarray) -> np.ndarray:
    """将邻接矩阵中非法边清零，返回新矩阵。"""
    result = adjacency.copy()
    result[~allowed_mask] = 0.0
    np.fill_diagonal(result, 0.0)
    return result


def count_allowed_edges(allowed_mask: np.ndarray) -> int:
    """掩码中允许的连接总数（最大可能边数）。"""
    return int(allowed_mask.sum())


def max_edges(n_total: int, n_input: int, n_output: int) -> int:
    """在当前连接方向策略下的最大边数。"""
    mask = build_allowed_mask(n_total, n_input, n_output)
    return count_allowed_edges(mask)
