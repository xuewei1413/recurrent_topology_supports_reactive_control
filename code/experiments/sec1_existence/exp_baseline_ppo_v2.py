"""
实验 D1v2: PPO 基线（修复 steps_to_450 采集）
与原版唯一区别：手动每 EVAL_INTERVAL global steps 暂停训练做评估，
避免 EvalCallback 在 n_envs>1 下的 eval_freq 歧义问题。
"""
import os, sys, csv, json, time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import gymnasium as gym
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)

# ── 配置 ──────────────────────────────────────────────────────────
PPO_CONFIG = {
    'policy': 'MlpPolicy',
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'verbose': 0,
    'policy_kwargs': dict(net_arch=[128, 128])
}

N_SEEDS = 10
TOTAL_TIMESTEPS = 200_000
EVAL_INTERVAL = 5_000   # global timesteps between evals
N_EVAL_EPISODES = 5     # quick eval during training
N_EVAL_FINAL = 20       # strict final eval
ENV_NAME = 'CartPole-v1'
N_ENVS = 8              # parallel envs for fast training
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'ppo_baseline')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_policy(model, n_episodes=N_EVAL_EPISODES, base_seed=0):
    env = gym.make(ENV_NAME)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        total = 0
        while not done and total < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += 1
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def train_ppo_single_seed(seed):
    print(f"\n[Seed {seed + 1}/{N_SEEDS}] Training PPO on {ENV_NAME}...")
    t0 = time.time()

    vec_env = make_vec_env(ENV_NAME, n_envs=N_ENVS, seed=seed)
    model = PPO(env=vec_env, seed=seed, device='cpu', **PPO_CONFIG)
    total_params = sum(p.numel() for p in model.policy.parameters())

    # Manual periodic evaluation
    training_curve = []
    steps_done = 0
    steps_to_450 = None
    next_eval = EVAL_INTERVAL

    while steps_done < TOTAL_TIMESTEPS:
        remaining = min(EVAL_INTERVAL, TOTAL_TIMESTEPS - steps_done)
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)
        steps_done += remaining

        # Evaluate at each checkpoint
        mean_r = evaluate_policy(model, n_episodes=N_EVAL_EPISODES, base_seed=seed + 1000)
        training_curve.append((int(steps_done), mean_r))

        if steps_to_450 is None and mean_r >= 450:
            steps_to_450 = int(steps_done)

        if steps_done >= next_eval:
            print(f"  step {steps_done:>7d}: mean_reward={mean_r:.1f}"
                  + ("  *** 450 reached" if steps_to_450 == steps_done else ""))
            next_eval += EVAL_INTERVAL

    # Final strict 20-episode eval
    env = gym.make(ENV_NAME)
    final_lengths = []
    for ep in range(N_EVAL_FINAL):
        obs, _ = env.reset(seed=seed + 2000 + ep)
        done = False
        steps = 0
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        final_lengths.append(steps)
    env.close()

    final_mean = float(np.mean(final_lengths))
    final_std = float(np.std(final_lengths))
    if steps_to_450 is None:
        steps_to_450 = TOTAL_TIMESTEPS if final_mean >= 450 else f">{TOTAL_TIMESTEPS}"

    elapsed = time.time() - t0
    print(f"[Seed {seed + 1}/{N_SEEDS}] Final: {final_mean:.1f} ± {final_std:.1f}, "
          f"steps_to_450={steps_to_450}, elapsed={elapsed:.1f}s")

    vec_env.close()
    return {
        'seed': seed,
        'training_curve': training_curve,
        'final_lengths': final_lengths,
        'final_mean': final_mean,
        'final_std': final_std,
        'final_min': int(np.min(final_lengths)),
        'final_max': int(np.max(final_lengths)),
        'steps_to_450': steps_to_450,
        'elapsed_sec': elapsed,
        'total_params': total_params,
    }


def save_results(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total_params = all_results[0]['total_params']
    all_curves = [r['training_curve'] for r in all_results]

    # JSON
    json_path = os.path.join(RESULTS_DIR, 'ppo_training_curves.json')
    json.dump([{
        'seed': r['seed'],
        'training_curve': r['training_curve'],
        'final_mean': r['final_mean'],
        'final_std': r['final_std'],
        'steps_to_450': r['steps_to_450'],
    } for r in all_results], open(json_path, 'w'), indent=2)
    print(f"JSON saved: {json_path}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, 'ppo_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed','final_mean','final_std','min','max','steps_to_450','total_params'])
        for r in all_results:
            w.writerow([r['seed'], f"{r['final_mean']:.2f}", f"{r['final_std']:.2f}",
                        r['final_min'], r['final_max'], r['steps_to_450'], total_params])
    print(f"CSV saved: {csv_path}")

    # Training curve figure
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_results:
        curve = r['training_curve']
        if curve:
            ts, rs = zip(*curve)
            ax.plot(ts, rs, alpha=0.7, lw=1.5, label=f"Seed {r['seed']}")
    ax.axhline(450, color='red', ls='--', alpha=0.5, label='Target 450')
    ax.axhline(500, color='green', ls='--', alpha=0.5, label='Max 500')
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Mean Reward')
    ax.set_title('PPO Training Curves (CartPole-v1, 10 seeds)')
    ax.legend(ncol=2, fontsize=8); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RESULTS_DIR, 'ppo_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    steps_450_vals = [r['steps_to_450'] for r in all_results if isinstance(r['steps_to_450'], int)]
    print(f"\nSteps-to-450 median: {int(np.median(steps_450_vals)) if steps_450_vals else 'N/A'}")
    print(f"Steps-to-450 range: {min(steps_450_vals)} / {max(steps_450_vals)}" if steps_450_vals else "")
    print(f"Final mean: {np.mean([r['final_mean'] for r in all_results]):.1f}")


def main():
    print("=" * 70)
    print("  PPO Baseline (v2 - fixed eval tracking)")
    print(f"  envs={N_ENVS}, eval_interval={EVAL_INTERVAL}, seeds={N_SEEDS}")
    print("=" * 70)

    all_results = [train_ppo_single_seed(s) for s in range(N_SEEDS)]
    save_results(all_results)
    print("\nDone.")


if __name__ == '__main__':
    main()
