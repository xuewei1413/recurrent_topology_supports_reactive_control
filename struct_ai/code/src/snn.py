"""
SNN Controller — 脉冲神经网络控制器 v2

关键修复：
1. 输入编码：8个神经元，每维状态用正/负两个神经元编码（半波整流）
   - 神经元 2i:   state[i] > 0 时激活
   - 神经元 2i+1: state[i] < 0 时激活
2. 突触模型：直接电压跳变（PSP），而非电流注入除以tau
3. 参数调整：tau=5, threshold=0.8, weight_scale=0.6

结构智能核心不变：
- 权重由神经元类型决定（E/I），固定不变
- 进化的只有拓扑（邻接矩阵，谁连谁）
"""

import numpy as np

# 输入编码缩放系数（确保典型 CartPole 状态值能驱动输入神经元发放）
INPUT_SCALE = 4.0


class SNNController:
    def __init__(self, config: dict, adjacency: np.ndarray, neuron_types: np.ndarray):
        """
        config       : 实验配置字典（n_input 应 = 8）
        adjacency    : 二值邻接矩阵 (N×N)，W[i,j]=1 表示 i→j 有连接
        neuron_types : +1=兴奋性，-1=抑制性
        """
        self.n      = config['n_total']
        self.n_in   = config['n_input']   # 8：±编码 4 维状态
        self.n_out  = config['n_output']
        self.tau    = config['tau']
        self.thresh = config['threshold']
        self.T      = config['T_steps']
        self.dt     = 1.0

        # 权重矩阵：直接电压跳变幅度
        # W[i,j] = adj[i,j] * type_sign[i] * weight_scale
        self.W = adjacency * neuron_types[:, np.newaxis] * config['weight_scale']

        self.v = np.zeros(self.n)

    def reset(self):
        self.v[:] = 0.0

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """
        将 N 维状态编码为 2N 维外部驱动电流
        正负分离，确保负值状态也能激活对应神经元
        维度数由 n_input // 2 自动推断，支持任意状态维度。
        """
        I_ext = np.zeros(self.n_in)
        n_dims = self.n_in // 2
        for i in range(n_dims):
            val = np.tanh(state[i]) * INPUT_SCALE
            I_ext[2 * i]     = max(0.0, val)   # 正方向神经元
            I_ext[2 * i + 1] = max(0.0, -val)  # 负方向神经元
        return I_ext

    def forward(self, state: np.ndarray) -> int:
        """
        输入：CartPole 状态（4维）
        输出：动作（0=左, 1=右）

        LIF 动力学（直接 PSP 版本）：
          v(t+1) = v(t) * (1 - dt/tau)          # 漏电
                 + spikes(t) @ W                  # 突触 PSP 跳变
          + I_ext * dt/tau                        # 外部驱动（仅输入层）
          if v >= threshold: spike → v reset to 0
        """
        I_ext = self.encode_state(np.array(state, dtype=np.float64))
        leak  = 1.0 - self.dt / self.tau  # 漏电系数

        spike_accum = np.zeros(self.n)
        spikes      = np.zeros(self.n)

        for _ in range(self.T):
            # 1. 漏电
            self.v *= leak

            # 2. 突触直接电压跳变（PSP）
            if spikes.any():
                self.v += spikes @ self.W

            # 3. 外部持续电流注入输入层
            self.v[:self.n_in] += I_ext * self.dt / self.tau

            # 4. 阈值判断与重置
            fired         = self.v >= self.thresh
            spikes        = fired.astype(np.float64)
            self.v[fired] = 0.0

            spike_accum += spikes

        # 解码：输出层 spike 数量最多的神经元对应动作
        output_activity = spike_accum[-self.n_out:]
        return int(np.argmax(output_activity))


class WeightedSNNController(SNNController):
    """支持独立权重矩阵的SNN控制器，用于Topology+Independent模式"""
    def __init__(self, config, adjacency, neuron_types, weight_matrix=None):
        super().__init__(config, adjacency, neuron_types)
        if weight_matrix is not None:
            # 覆盖默认的 W
            self.W = weight_matrix
