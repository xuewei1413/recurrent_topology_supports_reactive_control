"""
唯一 Observation Encoder（Phase 1）

设计文档 §4.1：
  使用与策略奖励无关的随机轨迹估计每个状态维度的均值和标准差，
  并在所有实验前冻结：

    z_d = clip((x_d - mu_d) / sigma_d, -3, 3)
    I_d_pos = max(z_d, 0)
    I_d_neg = max(-z_d, 0)

  训练、验证和测试只能调用同一个编码实现。
  禁止在环境包装器和控制器内部重复缩放、额外乘常数或再次施加未声明的 tanh。

== 禁止事项（旧代码的 bug，v2 必须根除）==

  1. 禁止 np.tanh(state)          — 旧 src/snn.py:55
  2. 禁止 obs * INPUT_SCALE       — 旧 exp_multitask.py:187 等 13 处
  3. 禁止在 encode() 内部乘任何常数
  4. 禁止环境侧包装器做任何缩放

  旧代码的有效编码是 tanh(obs * 4.0) * 4.0（双重缩放），与设计文档不符。
  本实现是全实验唯一的编码入口。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObservationEncoder:
    """冻结的 z-score + clip + half-wave 编码器。

    mu / sigma 在 pilot 阶段用随机轨迹估计，冻结后全实验共用。
    不允许在构造后修改 mu / sigma。

    编码输出维度 = 2 * obs_dim（每维拆正/负两个输入神经元）。
    布局：[d0_pos, d0_neg, d1_pos, d1_neg, ...]（interleaved）
    """

    mu: np.ndarray           # shape (obs_dim,), 均值
    sigma: np.ndarray        # shape (obs_dim,), 标准差
    clip: float = 3.0        # z-score clip 范围 [-clip, +clip]
    # 输入增益 alpha：将 z-score 放大到能驱动输入层高频发放
    # z-score 典型值 ~0.5，乘 input_gain=8.0 后 I_ext ~4.0
    # 每步注入 4.0*0.2=0.8，稳态 v=0.8/0.2=4.0 >> threshold 0.4
    input_gain: float = 8.0

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
        if self.mu.shape != self.sigma.shape:
            raise ValueError(
                f"mu shape {self.mu.shape} != sigma shape {self.sigma.shape}"
            )
        if self.mu.ndim != 1:
            raise ValueError(f"mu/sigma must be 1-D, got {self.mu.ndim}-D")
        if (self.sigma <= 0).any():
            raise ValueError("sigma must be strictly positive for all dims")
        if self.clip <= 0:
            raise ValueError(f"clip must be positive, got {self.clip}")

    @property
    def obs_dim(self) -> int:
        return self.mu.shape[0]

    @property
    def output_dim(self) -> int:
        """编码后输入神经元数 = 2 * obs_dim。"""
        return 2 * self.obs_dim

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """将原始观测编码为 2*obs_dim 维外部驱动电流。

        编码公式：
          z_d = clip((x_d - mu_d) / sigma_d, -clip, +clip)
          I_d_pos = input_gain * max(z_d, 0)
          I_d_neg = input_gain * max(-z_d, 0)

        input_gain 是 protocol 冻结的输入增益，确保输入神经元能发放。
        禁止 tanh、环境侧缩放、控制器内部额外乘常数。

        返回布局：[d0_pos, d0_neg, d1_pos, d1_neg, ...]
        """
        x = np.asarray(obs, dtype=np.float64)
        if x.shape != (self.obs_dim,):
            raise ValueError(
                f"obs shape {x.shape} != expected ({self.obs_dim},)"
            )

        z = np.clip((x - self.mu) / self.sigma, -self.clip, self.clip)
        i_pos = self.input_gain * np.maximum(z, 0.0)
        i_neg = self.input_gain * np.maximum(-z, 0.0)

        # interleave: [d0_pos, d0_neg, d1_pos, d1_neg, ...]
        out = np.empty(2 * self.obs_dim, dtype=np.float64)
        out[0::2] = i_pos
        out[1::2] = i_neg
        return out

    def stats_hash(self) -> str:
        """mu/sigma/clip/input_gain 的 SHA256，用于结果审计和冻结验证。"""
        payload = {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "clip": self.clip,
            "input_gain": self.input_gain,
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        """保存 mu/sigma/clip/input_gain 到 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "clip": self.clip,
            "input_gain": self.input_gain,
            "stats_hash": self.stats_hash(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str | Path) -> ObservationEncoder:
        """从 JSON 加载冻结的 encoder。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        enc = ObservationEncoder(
            mu=np.array(data["mu"], dtype=np.float64),
            sigma=np.array(data["sigma"], dtype=np.float64),
            clip=float(data["clip"]),
            input_gain=float(data.get("input_gain", 4.0)),
        )
        # 验证哈希一致
        if "stats_hash" in data and data["stats_hash"] != enc.stats_hash():
            raise ValueError(
                f"stats_hash mismatch on load: file={data['stats_hash']} "
                f"!= computed={enc.stats_hash()}"
            )
        return enc

    @staticmethod
    def identity(obs_dim: int, clip: float = 3.0, input_gain: float = 8.0) -> ObservationEncoder:
        """创建 mu=0, sigma=1 的 encoder（仅用于单元测试）。

        生产环境必须用 estimate_normalization.py 估计真实统计量。
        """
        return ObservationEncoder(
            mu=np.zeros(obs_dim, dtype=np.float64),
            sigma=np.ones(obs_dim, dtype=np.float64),
            clip=clip,
            input_gain=input_gain,
        )
