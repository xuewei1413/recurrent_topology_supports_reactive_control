"""
任务适配与 raw-return 验证（Phase 5）

设计文档 §3 / §4：
  四任务统一采用 raw return 作为主 fitness：
  - CartPole-v1：raw return = 存活步数（reward 每步 +1）
  - MountainCar-v0：raw return，额外报告 goal rate、maximum position、episode length
  - Acrobot-v1：raw return（reward 为负）
  - LunarLander-v3：raw return

  设计文档 §8.3 外部定位基线：
  - Random policy：环境最低基线
  - 固定随机 recurrent substrate + trained readout：reservoir computing 定位
  - NEAT 和 PPO：只用于能力尺度定位

关键：
  统一 reset/step/truncation 处理和 raw-return 计算。
  相同 test seed 在全部条件中对应相同初始环境。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.protocol import TASK_SPECS, TaskSpec


@dataclass
class EpisodeResult:
    """单 episode 的完整结果。"""

    total_reward: float          # raw return
    n_steps: int
    terminated: bool
    truncated: bool

    # MountainCar 额外指标
    max_position: float | None = None
    goal_reached: bool | None = None


def make_env(task: TaskSpec):
    """创建 gymnasium 环境。"""
    import gymnasium as gym
    return gym.make(task.env_id)


# 导入时注册 POMDP 任务（TASK_SPECS / gym id），使 assay 脚本可直接引用
try:
    from experiments_v2.tasks import pomdp_env
    pomdp_env.register()
except Exception:  # noqa: BLE001
    pass


def run_episode(
    env,
    policy_fn,
    task: TaskSpec,
    seed: int | None = None,
    max_steps: int | None = None,
) -> EpisodeResult:
    """运行一个 episode，返回 raw return 和辅助指标。

    设计文档 §5 测试：
      - 与 Gymnasium 直接 rollout 的 raw return 一致
      - terminated 与 truncated 均正确结束 episode

    Args:
        env:       gymnasium 环境
        policy_fn: 接受 obs 返回 action 的 callable
        task:      TaskSpec
        seed:      reset seed
        max_steps: 最大步数（默认 task.max_steps）
    """
    if max_steps is None:
        max_steps = task.max_steps

    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    total_reward = 0.0
    n_steps = 0
    terminated = False
    truncated = False

    # MountainCar 额外指标
    max_position = None
    goal_reached = None
    if task.name == "MountainCar":
        max_position = float(obs[0])

    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        n_steps += 1

        if task.name == "MountainCar":
            max_position = max(max_position, float(obs[0]))

        if terminated or truncated:
            break

    if task.name == "MountainCar":
        goal_reached = bool(max_position >= 0.5) if max_position is not None else False

    return EpisodeResult(
        total_reward=total_reward,
        n_steps=n_steps,
        terminated=terminated,
        truncated=truncated,
        max_position=max_position,
        goal_reached=goal_reached,
    )


def run_episodes(
    env,
    policy_fn,
    task: TaskSpec,
    seeds: list[int],
    max_steps: int | None = None,
) -> dict:
    """运行多个 episode 的评估。

    Returns:
        {
            'rewards': list[float],
            'mean', 'std', 'min', 'max': float,
            'n_episodes': int,
            # MountainCar 额外
            'goal_rate': float | None,
            'max_positions': list[float] | None,
            'episode_lengths': list[int],
        }
    """
    results = [run_episode(env, policy_fn, task, seed, max_steps) for seed in seeds]

    rewards = [r.total_reward for r in results]
    rewards_arr = np.array(rewards)

    out = {
        "rewards": rewards,
        "mean": float(rewards_arr.mean()),
        "std": float(rewards_arr.std()),
        "min": float(rewards_arr.min()),
        "max": float(rewards_arr.max()),
        "n_episodes": len(seeds),
        "episode_lengths": [r.n_steps for r in results],
    }

    if task.name == "MountainCar":
        max_positions = [r.max_position for r in results]
        goal_reached = [r.goal_reached for r in results]
        out["max_positions"] = max_positions
        out["goal_rate"] = float(np.mean(goal_reached))
        out["mean_max_position"] = float(np.mean(max_positions))

    return out


# ── 基线策略 ──────────────────────────────────────────────────────────

def random_policy(task: TaskSpec, seed: int = 0):
    """返回一个随机策略 callable。

    设计文档 §8.3：Random policy — 环境最低基线。
    """
    rng = np.random.RandomState(seed)
    n_actions = task.n_output

    def policy(obs):
        return int(rng.randint(0, n_actions))

    return policy


def evaluate_random_baseline(
    task: TaskSpec,
    seeds: list[int],
    policy_seed: int = 0,
) -> dict:
    """评估随机策略基线。"""
    env = make_env(task)
    policy = random_policy(task, policy_seed)
    result = run_episodes(env, policy, task, seeds)
    env.close()
    result["policy"] = "random"
    return result


def evaluate_ppo_baseline(
    task: TaskSpec,
    seeds: list[int],
    total_timesteps: int = 100_000,
    verbose: int = 0,
) -> dict | None:
    """评估 PPO 基线（能力尺度定位）。

    设计文档 §8.3：NEAT 和 PPO 只用于能力尺度定位，
    不作算法速度或参数效率的严格公平比较。

    需要 stable-baselines3。如果不可用返回 None。
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        return None

    env = make_env(task)

    # 训练
    def make_env_fn():
        return make_env(task)

    vec_env = DummyVecEnv([make_env_fn])
    model = PPO(
        "MlpPolicy", vec_env, verbose=verbose,
        n_steps=1024, batch_size=64, n_epochs=10,
        learning_rate=3e-4, gamma=0.99,
        seed=42,
    )
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()

    # 评估
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    result = run_episodes(env, policy_fn, task, seeds)
    env.close()
    result["policy"] = "ppo"
    result["total_timesteps"] = total_timesteps
    return result


def evaluate_neat_baseline(
    task: TaskSpec,
    seeds: list[int],
    n_generations: int = 50,
    pop_size: int = 50,
) -> dict | None:
    """评估 NEAT 基线（能力尺度定位）。

    需要 neat-python。如果不可用返回 None。
    """
    try:
        import neat
    except ImportError:
        return None

    # NEAT 实现较复杂，这里提供接口但实际实现推迟到需要时
    # 标记为未实现
    return None


# ── Reservoir Computing 定位基线 ──────────────────────────────────────

class ReservoirComputer:
    """固定随机 recurrent substrate + trained readout。

    设计文档 §8.3：固定随机 recurrent substrate + trained readout：
    与 reservoir computing 直接定位。

    这是不进化的随机 SNN 固定拓扑，只训练一个线性读出层（Ridge 回归）。
    用于回答："固定结构 + 学习权重读出" 能达到什么水平。

    与 F0 的对比意义：
      - F0 = 进化拓扑 + 固定读出（argmax spike count）
      - Reservoir = 随机拓扑 + 学习读出（Ridge regression）
      如果 F0 ≈ 或 > Reservoir，说明进化拓扑本身承载了控制功能，
      而不是"随便一个 substrate + 学习读出"就能达到。
    """

    def __init__(
        self,
        n_total: int = 128,
        obs_dim: int = 4,
        n_output: int = 2,
        sparsity: float = 0.10,
        spectral_radius: float = 0.9,
        tau: float = 5.0,
        threshold: float = 0.8,
        T_steps: int = 20,
        seed: int = 42,
    ):
        from ..core.connection_mask import build_allowed_mask
        n_input = obs_dim * 2
        rng = np.random.RandomState(seed)

        self.n_total = n_total
        self.n_input = n_input
        self.n_output = n_output
        self.tau = tau
        self.threshold = threshold
        self.T_steps = T_steps
        self.dt = 1.0
        self._leak = 1.0 - self.dt / tau
        self.obs_dim = obs_dim

        # 固定随机拓扑（不进化）
        mask = build_allowed_mask(n_total, n_input, n_output)
        allowed_pos = np.argwhere(mask)
        n_edges = int(sparsity * n_total * (n_total - 1))
        chosen = rng.choice(len(allowed_pos), size=n_edges, replace=False)

        # 随机权重 + E/I 符号
        neuron_types = np.ones(n_total, dtype=np.float64)
        hidden_range = np.arange(n_input, n_total - n_output)
        n_inh = int(len(hidden_range) * 0.3)
        inh_idx = rng.choice(hidden_range, n_inh, replace=False)
        neuron_types[inh_idx] = -1.0

        W = np.zeros((n_total, n_total), dtype=np.float64)
        w0 = 0.6
        for idx in chosen:
            r, c = allowed_pos[idx]
            W[r, c] = neuron_types[c] * w0  # 源 c 的类型

        # 缩放到目标 spectral radius（近似）
        if n_edges > 0:
            try:
                from scipy.sparse.linalg import eigs
                eigenvalues = eigs(W, k=min(10, n_total - 2), which='LM', return_eigenvectors=False)
                max_radius = np.max(np.abs(eigenvalues))
                if max_radius > 1e-10:
                    W *= spectral_radius / max_radius
            except Exception:
                pass  # 缩放失败时使用原始权重

        self.W = W
        self.v = np.zeros(n_total, dtype=np.float64)
        self.readout = None  # Ridge 训练后的读出权重

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        """half-wave 编码（与 ObservationEncoder.identity 等价）。"""
        I_ext = np.zeros(2 * self.obs_dim, dtype=np.float64)
        I_ext[0::2] = np.maximum(obs, 0.0)
        I_ext[1::2] = np.maximum(-obs, 0.0)
        return I_ext

    def _forward_collect(self, obs: np.ndarray) -> np.ndarray:
        """forward 并返回 hidden + output 的 spike 累积（作为 reservoir state）。"""
        I_ext = self._encode(obs)
        spike_accum = np.zeros(self.n_total, dtype=np.float64)
        spikes = np.zeros(self.n_total, dtype=np.float64)

        for _ in range(self.T_steps):
            self.v *= self._leak
            if spikes.any():
                self.v += spikes @ self.W
            self.v[:self.n_input] += I_ext * self.dt / self.tau
            fired = self.v >= self.threshold
            spikes = fired.astype(np.float64)
            self.v[fired] = 0.0
            spike_accum += spikes

        return spike_accum

    def reset(self):
        self.v[:] = 0.0

    def collect_states(self, env, task: TaskSpec, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """收集 reservoir states 和对应的 action labels。

        用于训练 readout：state → action 的映射。
        """
        all_states = []
        all_labels = []

        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            self.reset()

            for _ in range(task.max_steps):
                state = self._forward_collect(obs)

                # 用 random action 收集数据（teacher）
                rng = np.random.RandomState(seed)
                action = int(rng.randint(0, self.n_output))

                all_states.append(state.copy())
                # one-hot label
                label = np.zeros(self.n_output, dtype=np.float64)
                label[action] = 1.0
                all_labels.append(label)

                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        return np.array(all_states), np.array(all_labels)

    def train_readout(self, env, task: TaskSpec, train_seeds: list[int], alpha: float = 1.0):
        """用 Ridge 回归训练 readout。"""
        from sklearn.linear_model import Ridge

        X, Y = self.collect_states(env, task, train_seeds)
        self.readout = Ridge(alpha=alpha)
        self.readout.fit(X, Y)

    def predict_action(self, obs: np.ndarray) -> int:
        """用训练好的 readout 预测 action。"""
        if self.readout is None:
            raise RuntimeError("readout not trained. Call train_readout() first.")
        state = self._forward_collect(obs)
        logits = self.readout.predict(state.reshape(1, -1))[0]
        return int(np.argmax(logits))


def evaluate_reservoir_baseline(
    task: TaskSpec,
    eval_seeds: list[int],
    train_seeds: list[int],
    n_total: int = 128,
    reservoir_seed: int = 42,
) -> dict:
    """评估 reservoir computing 基线。

    设计文档 §8.3：固定随机 recurrent substrate + trained readout。

    Returns:
        评估结果字典
    """
    reservoir = ReservoirComputer(
        n_total=n_total,
        obs_dim=task.obs_dim,
        n_output=task.n_output,
        seed=reservoir_seed,
    )

    env = make_env(task)

    # 训练 readout
    reservoir.train_readout(env, task, train_seeds)

    # 评估
    rewards = []
    for seed in eval_seeds:
        obs, _ = env.reset(seed=seed)
        reservoir.reset()
        total = 0.0
        for _ in range(task.max_steps):
            action = reservoir.predict_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        rewards.append(total)

    env.close()

    rewards_arr = np.array(rewards)
    return {
        "rewards": rewards,
        "mean": float(rewards_arr.mean()),
        "std": float(rewards_arr.std()),
        "min": float(rewards_arr.min()),
        "max": float(rewards_arr.max()),
        "n_episodes": len(eval_seeds),
        "policy": "reservoir",
        "reservoir_seed": reservoir_seed,
        "n_total": n_total,
    }


# ── raw-return 验证工具 ────────────────────────────────────────────────

def verify_raw_return(
    task: TaskSpec,
    policy_fn,
    seed: int = 0,
) -> tuple[bool, str, float, float]:
    """验证我们的 rollout 与 gymnasium 直接 rollout 的 raw return 一致。

    设计文档 §5 测试：与 Gymnasium 直接 rollout 的 raw return 一致。
    """
    env = make_env(task)

    # 方法 1：用我们的 run_episode
    result_ours = run_episode(env, policy_fn, task, seed=seed)
    env.close()

    # 方法 2：直接 gymnasium rollout（重新跑，确定性策略应该一致）
    env2 = make_env(task)
    obs, _ = env2.reset(seed=seed)
    total = 0.0
    for _ in range(task.max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env2.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    env2.close()

    match = abs(result_ours.total_reward - total) < 1e-6
    msg = "ok" if match else f"mismatch: ours={result_ours.total_reward}, direct={total}"
    return match, msg, result_ours.total_reward, total
