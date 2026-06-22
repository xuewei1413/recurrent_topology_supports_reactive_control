"""
Weight Scale Robustness Experiment (Fig 05)

Loads evolved topologies from cross_task results (one seed per task),
sweeps weight_scale from 0.1 to 2.0, evaluates performance, and plots
robustness curves for all 5 tasks.

Output:
  figures_arxiv/fig05_weight_robustness.png
  figures_arxiv/fig05_weight_robustness.pdf
  experiments/results/weight_scale_ablation_data.npy
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
SNAKE_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'sec1_existence', 'snake'))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, SNAKE_DIR)

from src.snn import SNNController

INPUT_SCALE = 4.0

# ── Task configs (from exp_multitask.py) ──────────────────────────────────────

TASK_CONFIGS = {
    'CartPole': {
        'n_total': 128, 'n_input': 8, 'n_output': 2,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'max_steps': 500, 'fitness_fn': 'steps',
        'n_eval': 20,
    },
    'MountainCar': {
        'n_total': 64, 'n_input': 4, 'n_output': 3,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'max_steps': 200, 'fitness_fn': 'mc_special',
        'n_eval': 15,
    },
    'Acrobot': {
        'n_total': 128, 'n_input': 12, 'n_output': 3,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'max_steps': 500, 'fitness_fn': 'acrobot_reward',
        'n_eval': 15,
    },
    'FlappyBird': {
        'n_total': 64, 'n_input': 8, 'n_output': 2,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'max_steps': 2000, 'fitness_fn': 'flappy_steps',
        'n_eval': 15,
    },
    'Snake': {
        'n_total': 80, 'n_input': 16, 'n_output': 4,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'max_steps': 1000, 'fitness_fn': 'snake_score',
        'n_eval': 15,
    },
}

# ── Weight scale sweep range ──────────────────────────────────────────────────
WEIGHT_SCALES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]

CROSS_TASK_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'cross_task')
SEED_ID = 0  # Use seed 0 topology for each task


def make_env(task_name):
    if task_name == 'FlappyBird':
        from src.flappy_env_parametric import FlappyEnvParametric
        return FlappyEnvParametric()
    elif task_name == 'Snake':
        from snake_env import SnakeEnv
        return SnakeEnv()
    else:
        import gymnasium as gym
        env_id = {
            'CartPole': 'CartPole-v1',
            'MountainCar': 'MountainCar-v0',
            'Acrobot': 'Acrobot-v1',
        }[task_name]
        return gym.make(env_id)


def evaluate_at_weight_scale(adj, neuron_types, task_name, task_cfg, weight_scale, n_eval):
    """Evaluate topology at a given weight_scale."""
    config = {k: task_cfg[k] for k in ('n_total', 'n_input', 'n_output',
              'tau', 'threshold', 'T_steps')}
    config['weight_scale'] = weight_scale
    config['dt'] = 1.0

    env = make_env(task_name)
    scores = []

    for ep in range(n_eval):
        if task_name in ('FlappyBird', 'Snake'):
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=42 + ep)

        ctrl = SNNController(config, adj, neuron_types)
        total = 0.0
        max_pos = obs[0] if task_name == 'MountainCar' else None

        for step in range(task_cfg['max_steps']):
            action = ctrl.forward(np.array(obs, dtype=np.float64) * INPUT_SCALE)

            if task_name in ('FlappyBird', 'Snake'):
                obs, reward, done, info = env.step(int(action))
                if task_name == 'FlappyBird':
                    total += 1.0
                else:
                    total += reward
                if done:
                    break
            else:
                obs, reward, terminated, truncated, _ = env.step(int(action))
                if task_cfg['fitness_fn'] == 'steps':
                    total += 1.0
                elif task_cfg['fitness_fn'] == 'acrobot_reward':
                    total += reward
                elif task_cfg['fitness_fn'] == 'mc_special':
                    if obs[0] > max_pos:
                        max_pos = obs[0]
                if terminated or truncated:
                    break

        # Score calculation
        if task_cfg['fitness_fn'] == 'mc_special':
            avg_max = max_pos if max_pos is not None else obs[0]
            fitness = (avg_max + 1.2) / 1.8 * 1000
            if avg_max >= 0.3:  fitness += 300
            if avg_max >= 0.45: fitness += 700
            if avg_max >= 0.5:  fitness += 1000
            scores.append(fitness)
        elif task_name == 'Snake':
            scores.append(info.get('score', 0) * 100 + total)
        elif task_name == 'FlappyBird':
            scores.append(total)
        elif task_cfg['fitness_fn'] == 'acrobot_reward':
            scores.append(500.0 + total)
        else:
            scores.append(total)

    if task_name not in ('FlappyBird', 'Snake'):
        env.close()

    return float(np.mean(scores)), float(np.std(scores))


def run_sweep():
    all_results = {}

    for task_name in TASK_CONFIGS:
        task_cfg = TASK_CONFIGS[task_name]
        npz_path = os.path.join(CROSS_TASK_DIR, f'{task_name}_seed{SEED_ID}.npz')

        if not os.path.exists(npz_path):
            print(f'  [SKIP] {task_name}: {npz_path} not found')
            continue

        d = np.load(npz_path)
        adj = d['adj']
        neuron_types = d['types']
        print(f'\n{"="*60}')
        print(f'{task_name} (n={adj.shape[0]}, connections={int(adj.sum())})')
        print(f'{"="*60}')

        task_results = []
        for ws in WEIGHT_SCALES:
            mean, std = evaluate_at_weight_scale(
                adj, neuron_types, task_name, task_cfg, ws, task_cfg['n_eval']
            )
            task_results.append({'ws': ws, 'mean': mean, 'std': std})
            print(f'  ws={ws:.1f}: mean={mean:.1f}, std={std:.1f}')

        all_results[task_name] = task_results

    return all_results


def plot_figure(all_results, outdir):
    """Generate publication-quality fig05 with normalized performance."""
    os.makedirs(outdir, exist_ok=True)

    TASK_ORDER = ['CartPole', 'MountainCar', 'Acrobot', 'FlappyBird', 'Snake']
    COLORS = {
        'CartPole': '#2196F3', 'MountainCar': '#FF9800', 'Acrobot': '#4CAF50',
        'FlappyBird': '#E91E63', 'Snake': '#9C27B0',
    }
    MARKERS = {
        'CartPole': 'o', 'MountainCar': 's', 'Acrobot': '^',
        'FlappyBird': 'D', 'Snake': 'v',
    }

    # Two-panel figure: (a) raw scores, (b) normalized to default ws=0.6
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel (a): Raw performance scores ──
    for task_name in TASK_ORDER:
        if task_name not in all_results:
            continue
        res = all_results[task_name]
        ws_vals = [r['ws'] for r in res]
        means = [r['mean'] for r in res]
        stds = [r['std'] for r in res]

        color = COLORS[task_name]
        marker = MARKERS[task_name]

        ax_raw.errorbar(ws_vals, means, yerr=stds,
                        fmt=f'{marker}-', color=color,
                        linewidth=2, markersize=7, capsize=4,
                        markerfacecolor='white', markeredgewidth=2,
                        markeredgecolor=color,
                        label=task_name, alpha=0.9)

        # Mark the default weight_scale=0.6
        for r in res:
            if r['ws'] == 0.6:
                ax_raw.plot(0.6, r['mean'], marker='*', markersize=15,
                            color=color, zorder=10, markeredgecolor='black',
                            markeredgewidth=0.8)

    ax_raw.axvline(0.6, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    ax_raw.set_xlabel('weight_scale', fontsize=12)
    ax_raw.set_ylabel('Performance Score', fontsize=12)
    ax_raw.set_title('(a) Raw Performance', fontsize=13, fontweight='bold')
    ax_raw.legend(loc='best', fontsize=9, framealpha=0.9)
    ax_raw.grid(True, alpha=0.2)
    ax_raw.set_xlim(0.05, 2.05)

    # ── Panel (b): Normalized to default (ws=0.6 = 100%) ──
    for task_name in TASK_ORDER:
        if task_name not in all_results:
            continue
        res = all_results[task_name]
        ws_vals = [r['ws'] for r in res]
        means = [r['mean'] for r in res]
        stds = [r['std'] for r in res]

        # Find the default (ws=0.6) value for normalization
        default_mean = None
        for r in res:
            if r['ws'] == 0.6:
                default_mean = r['mean']
                break

        if default_mean is None or abs(default_mean) < 1e-6:
            continue

        norm_means = [m / default_mean * 100.0 for m in means]
        norm_stds  = [s / abs(default_mean) * 100.0 for s in stds]

        color = COLORS[task_name]
        marker = MARKERS[task_name]

        ax_norm.errorbar(ws_vals, norm_means, yerr=norm_stds,
                         fmt=f'{marker}-', color=color,
                         linewidth=2, markersize=7, capsize=4,
                         markerfacecolor='white', markeredgewidth=2,
                         markeredgecolor=color,
                         label=task_name, alpha=0.9)

    ax_norm.axvline(0.6, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    ax_norm.axhline(100, color='gray', linestyle=':', alpha=0.4, linewidth=1.0)
    ax_norm.set_xlabel('weight_scale', fontsize=12)
    ax_norm.set_ylabel('Performance (% of default)', fontsize=12)
    ax_norm.set_title('(b) Normalized Robustness', fontsize=13, fontweight='bold')
    ax_norm.legend(loc='best', fontsize=9, framealpha=0.9)
    ax_norm.grid(True, alpha=0.2)
    ax_norm.set_xlim(0.05, 2.05)

    fig.suptitle('Weight Scale Robustness Across Tasks',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = os.path.join(outdir, 'fig05_weight_robustness.png')
    pdf_path = os.path.join(outdir, 'fig05_weight_robustness.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'\nSaved: {png_path} ({os.path.getsize(png_path)/1024:.0f} KB)')
    print(f'Saved: {pdf_path} ({os.path.getsize(pdf_path)/1024:.0f} KB)')

    return png_path, pdf_path


if __name__ == '__main__':
    print('=' * 60)
    print('Weight Scale Robustness Experiment (Fig 05)')
    print('=' * 60)
    print(f'Weight scales: {WEIGHT_SCALES}')
    print(f'Using seed {SEED_ID} topology from cross_task/')

    # Run the sweep
    all_results = run_sweep()

    # Save raw data
    data_path = os.path.join(SCRIPT_DIR, '..', 'results', 'weight_scale_ablation_data.npy')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.save(data_path, all_results, allow_pickle=True)
    print(f'\nData saved: {data_path}')

    # Generate figure
    outdir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'figures_arxiv'))
    plot_figure(all_results, outdir)

    # Print summary table
    print('\n' + '=' * 80)
    print(f'{"Task":<14}', end='')
    for ws in WEIGHT_SCALES:
        print(f' {ws:>5.1f}', end='')
    print()
    print('-' * 80)
    for task_name, res in all_results.items():
        print(f'{task_name:<14}', end='')
        for r in res:
            print(f' {r["mean"]:>5.0f}', end='')
        print()

    print('\nDone.')
