"""
Continuous SNN Controller — 连续动作空间脉冲网络控制器

与 SNNController 的区别：
  - 输出层发放率映射到连续值 [-1, 1]（而非 argmax 离散动作）
  - 支持 1 维或多维连续输出
  - 其余 LIF 动力学完全一致
"""

import numpy as np

INPUT_SCALE = 4.0


class ContinuousSNNController:
    def __init__(self, config: dict, adjacency: np.ndarray, neuron_types: np.ndarray):
        self.n      = config['n_total']
        self.n_in   = config['n_input']
        self.n_out  = config['n_output']
        self.tau    = config['tau']
        self.thresh = config['threshold']
        self.T      = config['T_steps']
        self.dt     = 1.0

        self.W = adjacency * neuron_types[:, np.newaxis] * config['weight_scale']
        self.v = np.zeros(self.n)

    def reset(self):
        self.v[:] = 0.0

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        I_ext = np.zeros(self.n_in)
        n_dims = self.n_in // 2
        for i in range(n_dims):
            val = np.tanh(state[i]) * INPUT_SCALE
            I_ext[2 * i]     = max(0.0, val)
            I_ext[2 * i + 1] = max(0.0, -val)
        return I_ext

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        连续输出版本：输出层发放率归一化到 [-1, 1]。
        对于 1 维输出：2 个输出神经元，spike_rate 差值 → 连续值。
        对于 n 维输出：每维用 2 个神经元（正负编码），归一化到 [-1, 1]。
        """
        I_ext = self.encode_state(np.array(state, dtype=np.float64))
        leak  = 1.0 - self.dt / self.tau

        spike_accum = np.zeros(self.n)
        spikes      = np.zeros(self.n)

        for _ in range(self.T):
            self.v *= leak
            if spikes.any():
                self.v += spikes @ self.W
            self.v[:self.n_in] += I_ext * self.dt / self.tau

            fired         = self.v >= self.thresh
            spikes        = fired.astype(np.float64)
            self.v[fired] = 0.0

            spike_accum += spikes

        # 连续解码：每个输出维度从 2 个神经元（正/负）解码
        output_activity = spike_accum[-self.n_out * 2:] if self.n_out * 2 <= self.n_out else spike_accum[-self.n_out:]

        # 简单解码：发放率映射到 [-1, 1]
        rates = spike_accum[-self.n_out:] / self.T

        # 如果输出层恰好是偶数个且有正负配对
        if self.n_out == 1:
            # 1维：取最后一个神经元的发放率，sigmoid 映射
            rate = rates[0]
            action = np.tanh(rate * 4.0 - 2.0)  # 映射到 [-1, 1]
            return np.array([action])
        else:
            # 多维：每维发放率映射到 [-1, 1]
            actions = np.tanh(rates * 4.0 - 2.0)
            return actions


def evaluate_continuous(adj, neuron_types, config, env_name,
                        n_episodes=5, max_steps=1000, seed=0):
    """评估连续控制拓扑在指定环境中的平均回报。"""
    import gymnasium as gym

    ctrl = ContinuousSNNController(config, adj, neuron_types)
    env = gym.make(env_name)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ctrl.reset()
        total = 0.0
        for _ in range(max_steps):
            action = ctrl.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)

    env.close()
    return float(np.mean(rewards))
