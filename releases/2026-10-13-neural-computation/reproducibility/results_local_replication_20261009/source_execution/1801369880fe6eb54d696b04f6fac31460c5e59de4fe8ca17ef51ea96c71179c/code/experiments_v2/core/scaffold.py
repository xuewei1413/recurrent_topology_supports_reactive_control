"""
E/I Scaffold 与 Module 分配（Phase 1）

设计文档 §4 / §5：
  - hidden neurons 中 30% 为抑制性
  - 同一 evolution seed 的全部配对条件共享 E/I 类型
  - hidden neurons 均衡随机划分为 4 个模块，标签由 seed 确定并永久冻结
  - 模块标签不限制连接方向或模块内/模块间边数

关键约束：
  scaffold 由 evolution seed 唯一确定。F0/G1/M4/N/E/RA/RW 全部配对条件
  必须共享同一 scaffold，才能进行配对比较。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .protocol import ControllerProtocol, TaskSpec


# neuron_types 编码：
#   +1 = 兴奋性（hidden）
#   -1 = 抑制性（hidden）
#    0 = input / output（不参与 E/I 划分）
#
# module_labels 编码：
#   0..n_modules-1 = hidden 神经元的模块标签
#  -1 = input / output（无模块标签）


@dataclass(frozen=True)
class NeuronScaffold:
    """由 evolution seed 确定的永久冻结结构。

    包含 E/I 类型分配和 4-module 标签。同一 seed 的全部配对条件共享此 scaffold。
    """

    n_total: int
    n_input: int
    n_output: int
    neuron_types: np.ndarray   # shape (n_total,), int8: +1 / -1 / 0
    module_labels: np.ndarray  # shape (n_total,), int8: 0..3 或 -1

    @property
    def n_hidden(self) -> int:
        return self.n_total - self.n_input - self.n_output

    @property
    def hidden_mask(self) -> np.ndarray:
        """bool mask, True = hidden neuron。"""
        m = np.zeros(self.n_total, dtype=bool)
        m[self.n_input:self.n_total - self.n_output] = True
        return m

    @property
    def excitatory_mask(self) -> np.ndarray:
        return self.neuron_types == 1

    @property
    def inhibitory_mask(self) -> np.ndarray:
        return self.neuron_types == -1

    def module_mask(self, module_id: int) -> np.ndarray:
        """返回指定模块的 bool mask（仅 hidden）。"""
        return self.module_labels == module_id

    def type_hash(self) -> str:
        """E/I 分配的 SHA256 哈希，用于结果审计。"""
        return _sha256_array(self.neuron_types)

    def module_hash(self) -> str:
        """模块标签的 SHA256 哈希，用于结果审计。"""
        return _sha256_array(self.module_labels)

    def module_labels_for_k(self, k: int) -> np.ndarray:
        """生成 K 个模块的标签（用于通用 MK 条件）。

        使用与 from_seed 相同的 seed，但划分到 K 个模块而非默认的 4 个。
        均衡随机，大小差 ≤ 1。

        对于 K=4，返回与 module_labels 完全相同的结果（向后兼容）。
        """
        if k == 4:
            return self.module_labels.copy()

        # 用 evolution seed 的确定性映射重新生成
        # NeuronScaffold 的 seed 不直接存储，但可以用 hidden_indices 的排列
        # 为保持确定性，使用 neuron_types 的哈希作为 seed
        seed_int = int(hashlib.sha256(self.neuron_types.tobytes()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed_int)

        labels = np.full(self.n_total, -1, dtype=np.int8)
        hidden_indices = np.arange(self.n_input, self.n_total - self.n_output)
        shuffled = rng.permutation(hidden_indices)

        for i, idx in enumerate(shuffled):
            labels[idx] = i % k

        return labels

    def scaffold_with_k_modules(self, k: int) -> "NeuronScaffold":
        """返回一个使用 K 模块标签的新 scaffold（保持其他属性不变）。"""
        return NeuronScaffold(
            n_total=self.n_total,
            n_input=self.n_input,
            n_output=self.n_output,
            neuron_types=self.neuron_types,
            module_labels=self.module_labels_for_k(k),
        )

    @staticmethod
    def from_seed(
        seed: int,
        task: TaskSpec,
        controller: ControllerProtocol | None = None,
    ) -> NeuronScaffold:
        """从 evolution seed 确定性生成 scaffold。

        同一 seed 永远产生相同的 E/I 分配和模块标签。
        """
        if controller is None:
            controller = ControllerProtocol()

        n_total = controller.n_total
        n_input = task.n_input
        n_output = task.n_output
        frac_inh = controller.frac_inhibitory
        n_modules = controller.n_modules

        n_hidden = n_total - n_input - n_output
        if n_hidden <= 0:
            raise ValueError(
                f"n_hidden <= 0: n_total={n_total}, n_input={n_input}, n_output={n_output}"
            )

        rng = np.random.RandomState(seed)

        # ── E/I 分配 ──
        hidden_indices = np.arange(n_input, n_total - n_output)
        n_inh = int(n_hidden * frac_inh)
        if n_inh < 1:
            raise ValueError(f"n_inh={n_inh} < 1, hidden too small for {frac_inh}")

        inh_idx = rng.choice(hidden_indices, size=n_inh, replace=False)
        inh_set = set(inh_idx.tolist())

        neuron_types = np.zeros(n_total, dtype=np.int8)
        for idx in hidden_indices:
            neuron_types[idx] = -1 if idx in inh_set else 1

        # ── 模块划分（均衡随机，四模块大小差 ≤ 1）──
        module_labels = np.full(n_total, -1, dtype=np.int8)
        shuffled = rng.permutation(hidden_indices)
        # 轮流分配到 0..n_modules-1，保证均衡
        for i, idx in enumerate(shuffled):
            module_labels[idx] = i % n_modules

        return NeuronScaffold(
            n_total=n_total,
            n_input=n_input,
            n_output=n_output,
            neuron_types=neuron_types,
            module_labels=module_labels,
        )


def _sha256_array(arr: np.ndarray) -> str:
    """对 numpy 数组生成稳定的 SHA256 哈希。"""
    return hashlib.sha256(arr.tobytes()).hexdigest()
