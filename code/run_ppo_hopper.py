"""
PPO Hopper-v4 baseline: verify paper values.

Paper claims:
  - PPO 100k steps: ~574
  - PPO 1M-step reference: 2083.7 mean across 3 seeds

This script trains PPO on Hopper-v4 at two budgets:
  1. 100,000 timesteps  (matching paper's "100k steps" condition)
  2. 1,000,000 timesteps (1M-step reference condition)

Each is evaluated over 20 episodes, repeated for 3 seeds.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install stable-baselines3 gymnasium[mujoco]")
    sys.exit(1)


def make_env():
    def _init():
        return gym.make('Hopper-v4')
    return _init


def train_and_eval(total_timesteps, seed, n_eval=20):
    vec_env = DummyVecEnv([make_env()])
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=seed,
                n_steps=2048, batch_size=64, n_epochs=10,
                learning_rate=3e-4, ent_coef=0.0)

    print(f"  Training PPO (seed={seed}, {total_timesteps:,} steps)...", flush=True)
    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.0f}s", flush=True)

    eval_env = gym.make('Hopper-v4')
    rewards = []
    for ep in range(n_eval):
        obs, _ = eval_env.reset(seed=seed * 1000 + ep)
        total = 0.0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    eval_env.close()
    vec_env.close()

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    print(f"    Eval: {mean_r:.1f} +/- {std_r:.1f}")
    return mean_r, std_r, elapsed


def main():
    print("=" * 60)
    print("PPO Hopper-v4 Baseline Verification")
    print("=" * 60)

    seeds = [0, 1, 2]
    budgets = [
        (100_000, "100k"),
        (1_000_000, "1M"),
    ]

    results = {}
    for timesteps, label in budgets:
        print(f"\n--- Budget: {label} ({timesteps:,} steps) ---")
        means = []
        stds = []
        for seed in seeds:
            m, s, t = train_and_eval(timesteps, seed, n_eval=20)
            means.append(m)
            stds.append(s)
        results[label] = {
            'means': means,
            'stds': stds,
            'overall_mean': np.mean(means),
            'overall_std': np.std(means),
        }
        print(f"  => {label}: {np.mean(means):.1f} +/- {np.std(means):.1f} "
              f"(per-seed: {[f'{m:.0f}' for m in means]})")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, r in results.items():
        print(f"  {label}: {r['overall_mean']:.1f} +/- {r['overall_std']:.1f}")

    # Paper values for comparison
    print(f"\nPaper values:")
    print(f"  100k: ~574")
    print(f"  1M-step reference: 2083.7")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), 'experiments', 'results',
                           'ppo_hopper')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'ppo_hopper_results.npy'), results,
            allow_pickle=True)
    print(f"\nResults saved to {out_dir}/ppo_hopper_results.npy")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(results.keys())
    means = [results[l]['overall_mean'] for l in labels]
    stds_plot = [results[l]['overall_std'] for l in labels]

    colors = ['#F39C12', '#3498DB']
    bars = ax.bar(labels, means, yerr=stds_plot, color=colors, alpha=0.7,
                  edgecolor='black', capsize=5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{m:.0f}', ha='center', fontsize=12, fontweight='bold')

    # Add paper reference lines
    ax.axhline(574, color='#F39C12', linestyle='--', alpha=0.5,
               label='Paper: 100k (~574)')
    ax.axhline(2083.7, color='#3498DB', linestyle='--', alpha=0.5,
               label='Paper: 1M-step reference (2083.7)')

    ax.set_ylabel('Mean Reward (20 episodes)', fontsize=12)
    ax.set_title('PPO Hopper-v4 Baseline Verification', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, 'ppo_hopper_verification.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {png_path}")


if __name__ == '__main__':
    main()
