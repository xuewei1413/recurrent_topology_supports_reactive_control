"""
统一有效权重构造（Phase 1）

== 有效权重公式 ==

  本实现的有效传播权重为：

    W_eff[j,i] = A[j,i] * sign(type_source_i) * w0 * gain_source_i

  其中 i 是源（突触前）神经元，j 是目标（突触后）神经元。

== E/I 符号决策（源 i 类型）==

  设计文档 §5 原文写 sign(type[j])（目标 j 类型），
  但经计划审批确认为选项 B：采用源 i 的类型。理由：
    1. 生物学正确：突触前神经元决定递质类型（兴奋/抑制）
    2. 与旧代码 src/snn.py:39 一致（neuron_types[:, np.newaxis] = 源 i）
    3. 与现有已验证 CartPole/MC 结果连续

  因此实现公式与设计文档字面不同，但与生物学和旧代码一致。

== gain 维度对应条件 ==

  F0:  gain_vector 全为 1（标量 None 或长度 1）
  G1:  gain_vector 长度 1（全局广播到所有 hidden 源）
  M4:  gain_vector 长度 4（按源所属模块索引）
  N:   gain_vector 长度 n_total（每源神经元一个，input/output 恒为 1）
  E:   gain_vector 长度 = 活跃边数（每边独立正幅值，特殊处理）
"""

from __future__ import annotations

import numpy as np

from .scaffold import NeuronScaffold


# 条件标识符
F0 = "F0"
G1 = "G1"
M4 = "M4"  # 向后兼容别名 = MK(4)
N = "N"
E = "E"

# v4 通用模块调制
def mk_condition(k: int) -> str:
    """生成 MK 条件标识符，如 MK2, MK4, MK8。"""
    return f"MK{k}"

# 预定义别名
M2 = "MK2"
M8 = "MK8"

def is_mk_condition(condition: str) -> bool:
    """检查是否为 MK 条件（包括 M4 别名）。"""
    if condition == M4:
        return True
    return condition.startswith("MK") and condition[2:].isdigit()

def get_mk_k(condition: str) -> int:
    """从 MK 条件获取 K 值。M4 → 4, MK2 → 2, MK8 → 8。"""
    if condition == M4:
        return 4
    if condition.startswith("MK") and condition[2:].isdigit():
        return int(condition[2:])
    raise ValueError(f"Not an MK condition: {condition}")


def build_gain_per_neuron(
    scaffold: NeuronScaffold,
    gain_vector: np.ndarray | None,
    condition: str = F0,
) -> np.ndarray:
    """将条件特定的 gain_vector 展开为逐神经元 gain 数组。

    返回 shape (n_total,)，input/output 神经元恒为 1.0。
    只有 hidden 源神经元可能获得非 1 的 gain。

    注意：E 条件（逐边）不通过此函数处理，因为它不是逐源神经元的。
    E 条件直接调用 build_effective_weight_edge。
    """
    n = scaffold.n_total
    gains = np.ones(n, dtype=np.float64)

    if gain_vector is None:
        return gains  # F0

    gv = np.asarray(gain_vector, dtype=np.float64)

    if condition == F0 or len(gv) == 0:
        return gains  # 全 1

    # MK 条件优先检查（避免 len(gv)==1 被误判为 G1）
    if is_mk_condition(condition):
        K = get_mk_k(condition)
        if len(gv) != K:
            raise ValueError(
                f"MK condition K={K} but gain_vector length={len(gv)}. "
                f"Optimization dimension must strictly equal K."
            )
        module_labels = scaffold.module_labels
        hidden_mask = scaffold.hidden_mask
        hidden_idx = np.where(hidden_mask)[0]
        for idx in hidden_idx:
            m = module_labels[idx]
            if m < 0 or m >= K:
                raise ValueError(f"Module label {m} at neuron {idx} out of range [0, {K}).")
            gains[idx] = gv[m]
        for k in range(K):
            count = sum(1 for idx in hidden_idx if module_labels[idx] == k)
            if count == 0:
                raise ValueError(f"Module {k} has 0 neurons.")
        return gains

    if condition == G1 or len(gv) == 1:
        # 全局 gain 广播到所有 hidden 源
        g = float(gv[0])
        hidden_mask = scaffold.hidden_mask
        gains[hidden_mask] = g
        return gains

    if condition == N or len(gv) == n:
        if len(gv) != n:
            raise ValueError(f"N gain_vector length {len(gv)} != n_total {n}")
        # 每源神经元一个 gain，input/output 强制为 1
        gains = gv.copy()
        gains[:scaffold.n_input] = 1.0       # input 源无 gain
        gains[scaffold.n_total - scaffold.n_output:] = 1.0  # output 源无 gain
        return gains

    raise ValueError(
        f"Cannot interpret gain_vector: condition={condition}, len={len(gv)}"
    )


def build_effective_weight(
    adjacency: np.ndarray,
    scaffold: NeuronScaffold,
    gain_vector: np.ndarray | None = None,
    w0: float = 0.6,
    condition: str = F0,
) -> np.ndarray:
    """构造有效传播权重矩阵。

    W_eff[j,i] = A[j,i] * sign(type_source_i) * w0 * gain_source_i

    其中：
      - A[j,i] 是二值邻接矩阵（j=目标行, i=源列）
      - sign(type_source_i) 由源神经元 i 的 E/I 类型决定
      - gain_source_i 由 gain_vector 按条件展开到每个源神经元

    返回 shape (n_total, n_total) 的 float64 权重矩阵。
    """
    n = scaffold.n_total

    # 源神经元的符号
    # hidden 神经元：+1 (兴奋) / -1 (抑制)
    # input 神经元：作为源时是兴奋性的（+1），因为它们代表外部刺激
    # output 神经元：无出边（mask 禁止），符号不影响
    source_sign = scaffold.neuron_types.astype(np.float64)  # (n_total,)
    # input 神经元（type=0）的源符号设为 +1（兴奋性驱动）
    input_mask = np.zeros(n, dtype=bool)
    input_mask[:scaffold.n_input] = True
    source_sign[input_mask & (source_sign == 0)] = 1.0

    # 展开逐神经元 gain
    if condition == E:
        raise ValueError(
            "E condition (per-edge gain) must use build_effective_weight_edge()"
        )
    gains = build_gain_per_neuron(scaffold, gain_vector, condition)

    # W[j, i] = A[j, i] * sign(type_i) * w0 * gain_i
    # 用广播：sign 和 gain 沿列（源 i）方向
    W = (
        adjacency.astype(np.float64)
        * source_sign[np.newaxis, :]    # 源 i 的符号
        * w0
        * gains[np.newaxis, :]           # 源 i 的 gain
    )
    return W


def build_effective_weight_edge(
    adjacency: np.ndarray,
    scaffold: NeuronScaffold,
    edge_gains: np.ndarray,
    w0: float = 0.6,
) -> np.ndarray:
    """E 条件：每条活跃边一个独立正幅值。

    edge_gains 长度必须等于 adjacency 中活跃边数。
    符号仍由 E/I 类型决定（源 i 的类型），gain 只调幅值大小。

    返回 shape (n_total, n_total) 的 float64 权重矩阵。
    """
    adj = adjacency.astype(np.float64)
    active_rows, active_cols = np.where(adj > 0)
    n_active = len(active_rows)

    edge_gains = np.asarray(edge_gains, dtype=np.float64)
    if len(edge_gains) != n_active:
        # 长度不匹配时自动调整（截断或填充 1.0）
        if len(edge_gains) > n_active:
            edge_gains = edge_gains[:n_active]
        else:
            edge_gains = np.pad(edge_gains, (0, n_active - len(edge_gains)),
                                constant_values=1.0)

    source_sign = scaffold.neuron_types.astype(np.float64)
    # input 神经元（type=0）的源符号设为 +1（兴奋性驱动）
    n = scaffold.n_total
    input_mask = np.zeros(n, dtype=bool)
    input_mask[:scaffold.n_input] = True
    source_sign[input_mask & (source_sign == 0)] = 1.0

    # 逐边赋值
    W = np.zeros_like(adj)
    for k in range(n_active):
        j = active_rows[k]   # 目标
        i = active_cols[k]   # 源
        W[j, i] = source_sign[i] * w0 * edge_gains[k]

    return W


def f0_weight(adjacency: np.ndarray, scaffold: NeuronScaffold, w0: float = 0.6) -> np.ndarray:
    """F0 条件的便捷构造：所有 gain = 1。"""
    return build_effective_weight(adjacency, scaffold, gain_vector=None, w0=w0, condition=F0)


def gain_energy(gain_vector: np.ndarray | None, condition: str = F0) -> float:
    """计算 gain 向量的能量 ||g||^2，用于诊断极端放大。

    F0 条件返回 0（无自由 gain）。
    """
    if gain_vector is None:
        return 0.0
    return float(np.sum(np.asarray(gain_vector, dtype=np.float64) ** 2))
