"""
Observation Normalization 估计工具（Phase 1）

设计文档 §4.1：
  使用与策略奖励无关的随机轨迹估计每个状态维度的均值和标准差，
  并在所有实验前冻结。

本工具用 random policy 采样轨迹，估计 mu/sigma，生成冻结的 encoder。
必须在 pilot 阶段调用一次，确认性实验只加载冻结结果。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .encoder import ObservationEncoder
from .protocol import TASK_SPECS, TaskSpec


def estimate_observation_stats(
    task: TaskSpec,
    n_episodes: int = 1000,
    seed: int = 12345,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """用 random policy 采样轨迹，估计观测的均值和标准差。

    Args:
        task:       TaskSpec
        n_episodes: 采样 episode 数
        seed:       随机种子（固定，保证可复现）
        max_steps:  每 episode 最大步数（默认用 task.max_steps）

    Returns:
        mu:    (obs_dim,) 均值
        sigma: (obs_dim,) 标准差
    """
    import gymnasium as gym

    if max_steps is None:
        max_steps = task.max_steps

    env = gym.make(task.env_id)
    rng = np.random.RandomState(seed)
    n_actions = task.n_output

    all_obs = []

    for ep in range(n_episodes):
        ep_seed = int(rng.randint(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        all_obs.append(obs.copy())

        for _ in range(max_steps):
            action = int(rng.randint(0, n_actions))
            obs, reward, terminated, truncated, _ = env.step(action)
            all_obs.append(obs.copy())
            if terminated or truncated:
                break

    env.close()

    obs_array = np.array(all_obs, dtype=np.float64)  # (n_samples, obs_dim)

    # 防止极端值导致 sigma 过大
    mu = obs_array.mean(axis=0)
    sigma = obs_array.std(axis=0)

    # sigma 为 0 的维度（常数特征）设为 1，避免除零
    sigma[sigma < 1e-8] = 1.0

    return mu, sigma


def build_frozen_encoder(
    task: TaskSpec,
    n_episodes: int = 1000,
    seed: int = 12345,
    clip: float = 3.0,
    input_gain: float = 8.0,
    save_path: str | Path | None = None,
) -> ObservationEncoder:
    """估计统计量并构造冻结的 encoder。

    Args:
        task:        TaskSpec
        n_episodes:  采样 episode 数
        seed:        随机种子
        clip:        z-score clip 范围
        input_gain:  输入增益 alpha（冻结参数）
        save_path:   保存路径（如指定）

    Returns:
        冻结的 ObservationEncoder
    """
    mu, sigma = estimate_observation_stats(task, n_episodes, seed)
    encoder = ObservationEncoder(mu=mu, sigma=sigma, clip=clip, input_gain=input_gain)

    if save_path is not None:
        encoder.save(save_path)
        # 额外保存估计元数据
        meta_path = Path(save_path).with_suffix(".meta.json")
        meta = {
            "task": task.name,
            "env_id": task.env_id,
            "obs_dim": task.obs_dim,
            "n_episodes": n_episodes,
            "estimation_seed": seed,
            "clip": clip,
            "input_gain": input_gain,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return encoder


def estimate_all_tasks(
    output_dir: str | Path,
    tasks: dict[str, TaskSpec] | None = None,
    n_episodes: int = 1000,
    seed: int = 12345,
    clip: float = 3.0,
    input_gain: float = 8.0,
) -> dict[str, ObservationEncoder]:
    """为所有四任务估计并保存冻结的 encoder。

    Args:
        output_dir:  保存目录
        tasks:       任务字典（默认 TASK_SPECS）
        n_episodes:  采样 episode 数
        seed:        随机种子
        clip:        z-score clip 范围
        input_gain:  输入增益 alpha（冻结参数）

    Returns:
        {task_name: ObservationEncoder}
    """
    if tasks is None:
        tasks = TASK_SPECS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoders = {}
    for name, task in tasks.items():
        print(f"[{name}] estimating observation stats ({n_episodes} episodes)...")
        save_path = output_dir / f"observation_stats_{name}.json"
        encoder = build_frozen_encoder(task, n_episodes, seed, clip, input_gain, save_path)
        encoders[name] = encoder
        print(
            f"[{name}] done. mu range [{encoder.mu.min():.4f}, {encoder.mu.max():.4f}], "
            f"sigma range [{encoder.sigma.min():.4f}, {encoder.sigma.max():.4f}]"
        )

    return encoders


if __name__ == "__main__":
    # 命令行入口：为四任务生成冻结的 observation encoder
    import argparse

    parser = argparse.ArgumentParser(
        description="为四任务估计并冻结 observation normalization 统计量"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_v2/observation_stats",
        help="输出目录",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=1000, help="采样 episode 数"
    )
    parser.add_argument("--seed", type=int, default=12345, help="估计随机种子")
    parser.add_argument("--clip", type=float, default=3.0, help="z-score clip 范围")

    args = parser.parse_args()
    estimate_all_tasks(args.output_dir, None, args.n_episodes, args.seed, args.clip)
    print("\nAll observation stats frozen.")
