"""
统一 LIF 控制器（Phase 1）

设计原则：一个控制器类，通过 gain 向量和 condition 参数化所有条件（F0/G1/M4/N/E）。

LIF 动力学复用旧 src/snn.py:77-93 的核心逻辑（已验证有效），
但编码、权重构造和诊断全部替换为 v2 统一实现。

动力学（与旧代码一致）：
  v(t+1) = v(t) * (1 - dt/tau)          # 漏电
         + spikes(t) @ W_eff              # 突触 PSP 跳变
         + I_ext * dt / tau               # 外部驱动（仅输入层）
  if v >= threshold: spike → v reset to 0

读出：output 层 spike 累积数 argmax（离散动作）。

关键区别于旧代码：
  1. 编码：唯一 ObservationEncoder（z-score + clip + half-wave），无 tanh，无 INPUT_SCALE
  2. 权重：通过 build_effective_weight 统一构造，支持 gain 向量参数化
  3. 诊断：可选返回 firing rate / silent / saturated / tie_rate
  4. scaffold：E/I 类型和模块标签由 seed 确定，全配对条件共享
"""

from __future__ import annotations

import numpy as np

from .diagnostics import (
    EpisodeDiagnostics,
    StepDiagnostics,
    aggregate_episode_diagnostics,
    compute_step_diagnostics,
)
from .encoder import ObservationEncoder
from .protocol import ControllerProtocol
from .scaffold import NeuronScaffold
from .weight_construction import (
    E,
    F0,
    build_effective_weight,
    build_effective_weight_edge,
)


class UnifiedSNNController:
    """统一 LIF 控制器。

    通过 condition + gain_vector 参数化全部自由度条件。
    gains 全为 1 时，所有条件的输出完全一致（F0 回退保证）。
    """

    def __init__(
        self,
        scaffold: NeuronScaffold,
        adjacency: np.ndarray,
        encoder: ObservationEncoder,
        protocol: ControllerProtocol | None = None,
        gain_vector: np.ndarray | None = None,
        edge_gains: np.ndarray | None = None,
        condition: str = F0,
    ):
        """
        Args:
            scaffold:       E/I + module 分配（seed 确定，冻结）
            adjacency:      二值邻接矩阵 (n_total, n_total)
            encoder:        冻结的 observation encoder
            protocol:       LIF 参数（默认 ControllerProtocol）
            gain_vector:    G1/M4/N 的 gain 向量；None 或全 1 = F0
            edge_gains:     E 条件的逐边 gain（与 gain_vector 互斥）
            condition:      F0/G1/M4/N/E
        """
        if protocol is None:
            protocol = ControllerProtocol()

        self.scaffold = scaffold
        self.encoder = encoder
        self.protocol = protocol
        self.condition = condition

        # 维度
        self.n = scaffold.n_total
        self.n_in = scaffold.n_input
        self.n_out = scaffold.n_output
        self.n_hidden = scaffold.n_hidden

        # LIF 参数
        self.tau = protocol.tau
        self.threshold = protocol.threshold
        self.dt = protocol.dt
        self.T = protocol.T_steps
        self.w0 = protocol.w0
        self._leak = 1.0 - self.dt / self.tau

        # 验证编码器输出维度与输入神经元数一致
        if encoder.output_dim != self.n_in:
            raise ValueError(
                f"encoder output_dim {encoder.output_dim} != n_input {self.n_in}"
            )

        # 构造有效权重
        if condition == E:
            if edge_gains is None:
                raise ValueError("E condition requires edge_gains")
            self.W = build_effective_weight_edge(adjacency, scaffold, edge_gains, self.w0)
            self._edge_gains = np.asarray(edge_gains, dtype=np.float64)
            self._gain_vector = None
        else:
            self.W = build_effective_weight(
                adjacency, scaffold, gain_vector, self.w0, condition
            )
            self._gain_vector = gain_vector
            self._edge_gains = None

        # 膜电位状态
        self.v = np.zeros(self.n, dtype=np.float64)

        # 外部驱动（每步相同，预计算）
        self._I_ext = np.zeros(self.n_in, dtype=np.float64)

    def reset(self) -> None:
        """重置膜电位和外部驱动。"""
        self.v[:] = 0.0
        self._I_ext[:] = 0.0

    def forward(
        self,
        obs: np.ndarray,
        return_trace: bool = False,
    ) -> int | tuple[int, np.ndarray, np.ndarray]:
        """一步环境决策。

        Args:
            obs:          原始观测 (obs_dim,)
            return_trace: 如果 True，额外返回 (spike_accum, spike_trace)

        Returns:
            action: 离散动作索引
            如果 return_trace，返回 (action, spike_accum, spike_trace)
              spike_accum: (n_total,) T_steps 内累积发放计数
              spike_trace: (T_steps, n_total) 逐步发放记录
        """
        # 唯一编码入口
        self._I_ext = self.encoder.encode(obs)

        spike_accum = np.zeros(self.n, dtype=np.float64)
        spikes = np.zeros(self.n, dtype=np.float64)
        spike_trace = np.zeros((self.T, self.n), dtype=np.float64) if return_trace else None

        for t in range(self.T):
            # 1. 漏电
            self.v *= self._leak

            # 2. 突触 PSP 跳变
            # W[j,i] = 权重从源 i 到目标 j
            # PSP 到目标 j = sum_i W[j,i] * spikes[i] = (W @ spikes)[j]
            if spikes.any():
                self.v += self.W @ spikes

            # 3. 外部持续电流注入输入层
            self.v[: self.n_in] += self._I_ext * self.dt / self.tau

            # 4. 阈值判断与重置
            fired = self.v >= self.threshold
            spikes = fired.astype(np.float64)
            self.v[fired] = 0.0

            spike_accum += spikes
            if return_trace:
                spike_trace[t] = spikes

        # 读出：output 层累积发放 argmax
        output_activity = spike_accum[-self.n_out:]
        action = int(np.argmax(output_activity))

        if return_trace:
            return action, spike_accum, spike_trace
        return action

    def forward_with_step_diagnostics(
        self,
        obs: np.ndarray,
    ) -> tuple[int, np.ndarray, StepDiagnostics]:
        """forward 并返回步级诊断。

        Returns:
            action: 离散动作
            spike_accum: (n_total,) 累积发放
            step_diag: 该步的诊断
        """
        action, spike_accum, _ = self.forward(obs, return_trace=False)
        step_diag = compute_step_diagnostics(
            spike_accum, self.n_in, self.n_out, self.n, step=0
        )
        return action, spike_accum, step_diag

    def rollout(
        self,
        env,
        max_steps: int,
        seed: int | None = None,
        collect_diagnostics: bool = False,
    ) -> dict:
        """运行一个完整 episode。

        Args:
            env: gymnasium 环境
            max_steps: 最大步数
            seed: 环境 reset seed
            collect_diagnostics: 是否收集步级诊断

        Returns:
            {
                'total_reward': float,
                'n_steps': int,
                'actions': list[int],
                'step_diags': list[StepDiagnostics] (如果 collect_diagnostics),
            }
        """
        if seed is not None:
            obs, _ = env.reset(seed=seed)
        else:
            obs, _ = env.reset()

        self.reset()
        total_reward = 0.0
        actions = []
        step_diags = []

        hidden_activity_list = []  # for diagnostics

        for step in range(max_steps):
            if collect_diagnostics:
                # 需要 return_trace=True 才能拿到 spike_accum
                action, spike_accum, _ = self.forward(obs, return_trace=True)
                hidden_acc = spike_accum[self.n_in:self.n - self.n_out]
                hidden_activity_list.append(hidden_acc.copy())
                step_diag = compute_step_diagnostics(
                    spike_accum, self.n_in, self.n_out, self.n, step
                )
                step_diags.append(step_diag)
            else:
                action = self.forward(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            actions.append(action)

            if terminated or truncated:
                break

        result = {
            'total_reward': total_reward,
            'n_steps': len(actions),
            'actions': actions,
        }

        if collect_diagnostics:
            hidden_activity = np.array(hidden_activity_list) if hidden_activity_list else None
            ep_diag = aggregate_episode_diagnostics(
                step_diags, self.n_hidden, self.T, hidden_activity
            )
            result['episode_diagnostics'] = ep_diag
            result['step_diags'] = step_diags

        return result

    def evaluate(
        self,
        env,
        seeds: list[int],
        max_steps: int,
        collect_diagnostics: bool = False,
    ) -> dict:
        """运行多个 episode 的评估。

        Args:
            env: gymnasium 环境
            seeds: 每个 episode 的 reset seed 列表
            max_steps: 每个 episode 最大步数
            collect_diagnostics: 是否收集诊断

        Returns:
            {
                'rewards': list[float],
                'mean': float,
                'std': float,
                'min': float,
                'max': float,
                'n_episodes': int,
                'diagnostics': MultiEpisodeDiagnostics (如果 collect_diagnostics),
            }
        """
        rewards = []
        episode_diags = []

        for seed in seeds:
            result = self.rollout(env, max_steps, seed, collect_diagnostics)
            rewards.append(result['total_reward'])
            if collect_diagnostics:
                episode_diags.append(result['episode_diagnostics'])

        rewards_arr = np.array(rewards)
        out = {
            'rewards': rewards,
            'mean': float(rewards_arr.mean()),
            'std': float(rewards_arr.std()),
            'min': float(rewards_arr.min()),
            'max': float(rewards_arr.max()),
            'n_episodes': len(seeds),
        }

        if collect_diagnostics:
            from .diagnostics import aggregate_multi_episode
            out['diagnostics'] = aggregate_multi_episode(episode_diags)

        return out
