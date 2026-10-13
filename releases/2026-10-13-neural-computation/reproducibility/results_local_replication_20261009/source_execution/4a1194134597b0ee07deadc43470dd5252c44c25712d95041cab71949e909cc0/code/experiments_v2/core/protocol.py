"""
统一实验协议配置（Phase 0）

四任务完全统一的控制器协议、任务维度、拓扑预算和 seed split。
所有参数在 pilot 后冻结，确认性实验不得修改。

设计文档参考：
  - §4 统一控制器协议
  - §4.1 输入编码
  - §5 预先固定四个 hidden modules
  - §6 固定边预算的拓扑 GA
  - §11.2 Confirmatory
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


# ── 控制器协议（四任务统一，不可按任务修改）──────────────────────────────

@dataclass(frozen=True)
class ControllerProtocol:
    """LIF 控制器的固定参数。

    设计文档 §4：N=128，30% 抑制性，tau=5，threshold=0.8，T_steps=20，w0=0.6。
    禁止自连接；允许 input→hidden、hidden→hidden、hidden→output。
    """

    n_total: int = 128
    frac_inhibitory: float = 0.30       # hidden 中抑制性比例
    n_modules: int = 4                  # 固定 4 个 hidden modules
    tau: float = 5.0
    threshold: float = 0.4              # 发放阈值（pilot 调整：w0=0.6 > 0.4 使单 PSP 可触发）
    dt: float = 1.0
    T_steps: int = 20                   # 固定总内部步数
    w0: float = 0.6                     # 基础非零突触幅值
    allow_self_connection: bool = False


# ── 任务维度（四任务各自的观测/动作维度）────────────────────────────────

@dataclass(frozen=True)
class TaskSpec:
    """单个任务的维度和环境标识。

    n_input = obs_dim * 2（半波拆分：正/负各一个输入神经元）。
    hidden = n_total - n_input - n_output。
    """

    name: str                           # CartPole / MountainCar / Acrobot / LunarLander
    env_id: str                         # gymnasium env id
    obs_dim: int                        # 原始观测维度
    n_output: int                       # 离散动作数

    # 以下由 obs_dim 推导，保持只读
    @property
    def n_input(self) -> int:
        return self.obs_dim * 2

    @property
    def n_hidden(self) -> int:
        return 128 - self.n_input - self.n_output  # n_total 固定 128

    @property
    def max_steps(self) -> int:
        """每个 episode 的最大步数（gymnasium 默认截断）。"""
        return _TASK_MAX_STEPS[self.name]


_TASK_MAX_STEPS: dict[str, int] = {
    "CartPole": 500,
    "MountainCar": 200,
    "Acrobot": 500,
    "LunarLander": 1000,
}

# 四任务标准规格
TASK_SPECS: dict[str, TaskSpec] = {
    "CartPole": TaskSpec("CartPole", "CartPole-v1", obs_dim=4, n_output=2),
    "MountainCar": TaskSpec("MountainCar", "MountainCar-v0", obs_dim=2, n_output=3),
    "Acrobot": TaskSpec("Acrobot", "Acrobot-v1", obs_dim=6, n_output=3),
    "LunarLander": TaskSpec("LunarLander", "LunarLander-v3", obs_dim=8, n_output=4),
}


# ── 拓扑预算（四任务统一）──────────────────────────────────────────────

@dataclass(frozen=True)
class TopologyBudget:
    """固定边预算。设计文档 §4：主实验固定 10% 边预算。"""

    edge_budget_fraction: float = 0.10  # 占 n_total*(n_total-1) 的比例

    def edge_count(self, n_total: int = 128) -> int:
        """精确边数 = floor(fraction × n × (n-1))。N=128 → 1638。"""
        return int(self.edge_budget_fraction * n_total * (n_total - 1))


# ── Seed split（严格的 seed registry）───────────────────────────────────

@dataclass(frozen=True)
class SeedSplit:
    """训练、验证、测试 seed 集合，严格不重叠。

    设计文档 §6/§11.2：训练 seed batch 按固定日程轮换；
    测试集只运行一次；train/val/test seed 集合严格不重叠。
    pilot 后冻结。
    """

    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]

    def __post_init__(self):
        train = set(self.train_seeds)
        val = set(self.validation_seeds)
        test = set(self.test_seeds)
        if train & val:
            raise ValueError(f"train/validation seeds overlap: {train & val}")
        if train & test:
            raise ValueError(f"train/test seeds overlap: {train & test}")
        if val & test:
            raise ValueError(f"validation/test seeds overlap: {val & test}")


# ── 完整协议（组合以上全部）─────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentProtocol:
    """一个完整的冻结实验协议。"""

    protocol_version: str               # 如 "v2.0"
    code_commit: str                    # git hash
    controller: ControllerProtocol
    task: TaskSpec
    budget: TopologyBudget
    seed_split: SeedSplit
    n_eval_episodes_train: int = 8      # GA 中每个 genome 的训练评估 episode 数（pilot 调）
    n_confirmatory_seeds: int = 20      # 确认性 evolution seeds 数量

    def config_hash(self) -> str:
        """对整个配置生成 SHA256，用于结果文件审计。"""
        d = _to_serializable(asdict(self))
        canonical = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def edge_count(self) -> int:
        return self.budget.edge_count(self.controller.n_total)


def _to_serializable(obj: Any) -> Any:
    """递归转换 dataclass dict 为 JSON 可序列化对象。"""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialize_item(v) for v in obj]
    return obj


def _to_serialize_item(v: Any) -> Any:
    if isinstance(v, dict):
        return _to_serializable(v)
    if isinstance(v, (list, tuple)):
        return [_to_serialize_item(x) for x in v]
    return v
