"""
Activity 诊断（Phase 1）

设计文档 §8.2 / §9.1：
  同时记录 firing rate、静默比例、饱和比例、输出平局率和 gain 能量，
  防止极端放大伪造性能提升。

这些诊断在每次评估时记录，用于：
  - 检测整体失活（hidden 全静默）vs 传播中断
  - 检测极端饱和（hidden 全程发放）
  - 检测输出平局（argmax 无法区分动作）
  - 检测 gain 能量爆炸（M4/N/E 的极端放大）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class EpisodeDiagnostics:
    """单个 episode 的活动诊断。"""

    hidden_firing_rate: float       # hidden 神经元平均发放率 (0..1)
    output_firing_rate: float       # output 神经元平均发放率 (0..1)
    silent_fraction: float          # hidden 中整个 episode 零发放的比例
    saturated_fraction: float       # hidden 中整个 episode 每步都发放的比例
    output_tie_rate: float          # 输出层 argmax 出现平局的比例（跨步）
    n_steps: int                    # episode 总步数


@dataclass
class StepDiagnostics:
    """单步的细粒度诊断（可选收集）。"""

    step: int
    hidden_spike_count: int         # 该步 hidden 发放神经元数
    output_spike_count: int         # 该步 output 发放神经元数
    output_max: float               # output 累积发放最大值
    output_runner_up: float         # output 累积发放第二值
    is_tie: bool                    # argmax 是否平局


def compute_step_diagnostics(
    spike_accum: np.ndarray,
    n_input: int,
    n_output: int,
    n_total: int,
    step: int,
) -> StepDiagnostics:
    """从单步的 spike_accum 计算步级诊断。"""

    hidden_acc = spike_accum[n_input:n_total - n_output]
    output_acc = spike_accum[-n_output:]

    if len(output_acc) >= 2:
        sorted_out = np.sort(output_acc)[::-1]
        output_max = float(sorted_out[0])
        output_runner_up = float(sorted_out[1])
        is_tie = bool(abs(output_max - output_runner_up) < 1e-12)
    else:
        output_max = float(output_acc[0]) if len(output_acc) > 0 else 0.0
        output_runner_up = 0.0
        is_tie = False

    return StepDiagnostics(
        step=step,
        hidden_spike_count=int(hidden_acc.sum()),
        output_spike_count=int(output_acc.sum()),
        output_max=output_max,
        output_runner_up=output_runner_up,
        is_tie=is_tie,
    )


def aggregate_episode_diagnostics(
    step_diags: List[StepDiagnostics],
    n_hidden: int,
    T_steps: int,
    hidden_per_step_activity: np.ndarray | None = None,
) -> EpisodeDiagnostics:
    """将步级诊断聚合为 episode 级诊断。

    hidden_per_step_activity: shape (n_steps, n_hidden) 的逐步发放计数，
      用于精确计算 silent/saturated fraction。如为 None 则用近似估计。
    """
    n_steps = len(step_diags)

    # hidden firing rate：平均每步发放的 hidden 神经元比例
    if hidden_per_step_activity is not None:
        # 每步每个 hidden 神经元是否发放过（在 T_steps 内至少一次）
        # 但 spike_accum 已经累积了 T_steps 步，所以这里用累积值 / T_steps
        total_hidden_firing = sum(d.hidden_spike_count for d in step_diags)
        hidden_firing_rate = total_hidden_firing / (n_hidden * T_steps * n_steps) if n_steps > 0 else 0.0

        # silent: 整个 episode 该 hidden 神经元一次都没发放
        # 用逐步 hidden activity 的总和判断
        # hidden_per_step_activity shape (n_steps, n_hidden)
        total_per_neuron = hidden_per_step_activity.sum(axis=0)  # (n_hidden,)
        silent_fraction = float((total_per_neuron == 0).mean())
        # saturated: 每个 T_steps 窗口都发放（即累积 == T_steps * n_windows）
        # 近似：累积值 >= n_steps（至少每步发放一次）
        saturated_fraction = float((total_per_neuron >= n_steps).mean())
    else:
        total_hidden_firing = sum(d.hidden_spike_count for d in step_diags)
        hidden_firing_rate = total_hidden_firing / (n_hidden * n_steps) if n_steps > 0 else 0.0
        # 无法精确计算 silent/saturated
        silent_fraction = -1.0
        saturated_fraction = -1.0

    # output firing rate
    total_output_firing = sum(d.output_spike_count for d in step_diags)
    output_firing_rate = total_output_firing / (len(step_diags[0:1]) or 1)

    # output tie rate
    if n_steps > 0:
        n_ties = sum(1 for d in step_diags if d.is_tie)
        output_tie_rate = n_ties / n_steps
    else:
        output_tie_rate = 0.0

    return EpisodeDiagnostics(
        hidden_firing_rate=float(hidden_firing_rate),
        output_firing_rate=float(output_firing_rate),
        silent_fraction=silent_fraction,
        saturated_fraction=saturated_fraction,
        output_tie_rate=float(output_tie_rate),
        n_steps=n_steps,
    )


@dataclass
class MultiEpisodeDiagnostics:
    """多个 episode 的聚合诊断，用于最终结果报告。"""

    hidden_firing_rate_mean: float
    hidden_firing_rate_std: float
    output_firing_rate_mean: float
    output_firing_rate_std: float
    silent_fraction_mean: float
    saturated_fraction_mean: float
    output_tie_rate_mean: float
    n_episodes: int


def aggregate_multi_episode(episode_diags: List[EpisodeDiagnostics]) -> MultiEpisodeDiagnostics:
    """将 episode 级诊断聚合为多 episode 报告。"""
    if not episode_diags:
        return MultiEpisodeDiagnostics(0, 0, 0, 0, 0, 0, 0, 0)

    hfr = np.array([d.hidden_firing_rate for d in episode_diags])
    ofr = np.array([d.output_firing_rate for d in episode_diags])
    silent = np.array([d.silent_fraction for d in episode_diags if d.silent_fraction >= 0])
    sat = np.array([d.saturated_fraction for d in episode_diags if d.saturated_fraction >= 0])
    tie = np.array([d.output_tie_rate for d in episode_diags])

    return MultiEpisodeDiagnostics(
        hidden_firing_rate_mean=float(hfr.mean()),
        hidden_firing_rate_std=float(hfr.std()),
        output_firing_rate_mean=float(ofr.mean()),
        output_firing_rate_std=float(ofr.std()),
        silent_fraction_mean=float(silent.mean()) if len(silent) > 0 else -1.0,
        saturated_fraction_mean=float(sat.mean()) if len(sat) > 0 else -1.0,
        output_tie_rate_mean=float(tie.mean()),
        n_episodes=len(episode_diags),
    )
