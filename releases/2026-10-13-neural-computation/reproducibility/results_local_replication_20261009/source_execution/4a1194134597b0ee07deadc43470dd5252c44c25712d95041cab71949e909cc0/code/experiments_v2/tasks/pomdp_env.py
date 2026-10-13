"""POMDP 任务扩展：位置-only CartPole（遮罩速度分量）。

设计动机（论文广度缺口 2）：
v5 全部结论建立在 4 个近似 Markov 的经典任务上，"leak 积分器未被
招募为记忆体"的结论需要在部分可观测任务上接受检验。
位置-only CartPole 只给 (x, θ)，遮罩 (x_dot, θ_dot)：
速度信息只能通过跨步积分恢复——若框架正确，进化应在此任务上
真正招募 leak 积分器（记忆深度 > 0，leak 崩塌应显著损伤回报）。

注册后 TASK_SPECS["CartPolePosOnly"] 与 gym id "CartPolePosOnly-v0"
全局可用，既有 assay 脚本（memory_curve / route_shuffle / motif /
causal_propagation）可直接 --task CartPolePosOnly 复用。
"""
from __future__ import annotations

import numpy as np

from experiments_v2.core.protocol import TASK_SPECS, TaskSpec, _TASK_MAX_STEPS

TASK_NAME = "CartPolePosOnly"
GYM_ID = "CartPolePosOnly-v1"
OBS_IDX = [0, 2]  # x, theta（遮罩 x_dot, theta_dot）


def _make_env():
    import gymnasium as gym
    from gymnasium import spaces

    class _PosOnlyCartPole(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            low = env.observation_space.low[OBS_IDX]
            high = env.observation_space.high[OBS_IDX]
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        def observation(self, obs):
            return np.asarray(obs, dtype=np.float32)[OBS_IDX]

    return _PosOnlyCartPole(gym.make("CartPole-v1"))


_REGISTERED = False


def register() -> None:
    """注册 gym 环境 + TaskSpec + max_steps（幂等）。"""
    global _REGISTERED
    if _REGISTERED:
        return
    import gymnasium as gym
    try:
        gym.register(GYM_ID, entry_point=_make_env, max_episode_steps=500)
    except Exception:
        pass
    TASK_SPECS[TASK_NAME] = TaskSpec(TASK_NAME, GYM_ID,
                                     obs_dim=len(OBS_IDX), n_output=2)
    _TASK_MAX_STEPS[TASK_NAME] = 500
    _REGISTERED = True


def get_task() -> TaskSpec:
    register()
    return TASK_SPECS[TASK_NAME]
