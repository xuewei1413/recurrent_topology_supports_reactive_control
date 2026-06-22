"""
Generate Fig 10: Cross-Task Transfer Matrix Heatmap

For each topology evolved on a source task, evaluate it on all 5 target tasks.
Since network sizes differ across tasks, we embed the source topology into the
target network size.

The transfer matrix shows normalized performance:
  - Each cell = mean fitness when source-task topologies are evaluated on target task
  - Normalized by dividing by the diagonal (native task) mean fitness per column
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Paths --
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_DIR = os.path.join(CODE_DIR, 'experiments', 'results', 'cross_task')
OUT_DIR = os.path.join(os.path.dirname(CODE_DIR), 'figures_arxiv')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'experiments', 'sec1_existence', 'snake'))

# -- Task configurations (from exp_multitask.py) --
TASK_CONFIGS = {
    'CartPole': {
        'env_id': 'CartPole-v1',
        'n_total': 128, 'n_input': 8, 'n_output': 2,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.10,
        'max_steps': 500,
        'fitness_fn': 'steps',
    },
    'MountainCar': {
        'env_id': 'MountainCar-v0',
        'n_total': 64, 'n_input': 4, 'n_output': 3,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.12,
        'max_steps': 200,
        'fitness_fn': 'mc_special',
        'target_sparsity': 0.15, 'density_penalty': 200,
    },
    'Acrobot': {
        'env_id': 'Acrobot-v1',
        'n_total': 128, 'n_input': 12, 'n_output': 3,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.10,
        'max_steps': 500,
        'fitness_fn': 'acrobot_reward',
    },
    'FlappyBird': {
        'env_id': 'FlappyBird',
        'n_total': 64, 'n_input': 8, 'n_output': 2,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.10,
        'max_steps': 2000,
        'fitness_fn': 'flappy_steps',
    },
    'Snake': {
        'env_id': 'Snake',
        'n_total': 80, 'n_input': 16, 'n_output': 4,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.10,
        'max_steps': 1000,
        'fitness_fn': 'snake_score',
    },
}

TASK_ORDER = ['CartPole', 'MountainCar', 'Acrobot', 'FlappyBird', 'Snake']
N_SEEDS = 10
N_EVAL = 5
INPUT_SCALE = 4.0


def make_env(task_name):
    if task_name == 'FlappyBird':
        from src.flappy_env import FlappyBirdEnv
        return FlappyBirdEnv()
    elif task_name == 'Snake':
        from snake_env import SnakeEnv
        return SnakeEnv()
    else:
        import gymnasium as gym
        return gym.make(TASK_CONFIGS[task_name]['env_id'])


def embed_topology(src_adj, src_types, src_task, tgt_task):
    """
    Embed a source topology into a target network size.

    Strategy: Keep the source topology's internal structure but adapt the
    interface to the target task dimensions. Pad or truncate as needed.
    """
    src_cfg = TASK_CONFIGS[src_task]
    tgt_cfg = TASK_CONFIGS[tgt_task]
    n_src = src_cfg['n_total']
    n_tgt = tgt_cfg['n_total']

    if n_src == n_tgt and src_cfg['n_input'] == tgt_cfg['n_input'] and \
       src_cfg['n_output'] == tgt_cfg['n_output']:
        return src_adj.copy(), src_types.copy()

    adj_tgt = np.zeros((n_tgt, n_tgt), dtype=np.float64)
    types_tgt = np.ones(n_tgt, dtype=np.float64)

    n_overlap = min(n_src, n_tgt)
    adj_tgt[:n_overlap, :n_overlap] = src_adj[:n_overlap, :n_overlap]
    types_tgt[:n_overlap] = src_types[:n_overlap]

    if n_tgt > n_src:
        hidden_extra = np.arange(n_src, n_tgt - tgt_cfg['n_output'])
        if len(hidden_extra) > 0:
            n_inh_extra = int(len(hidden_extra) * 0.3)
            if n_inh_extra > 0:
                inh_idx = np.random.choice(hidden_extra, n_inh_extra, replace=False)
                types_tgt[inh_idx] = -1.0

    np.fill_diagonal(adj_tgt, 0.0)

    if n_tgt > n_src:
        sparsity = tgt_cfg['sparsity']
        new_region = np.random.rand(n_tgt, n_tgt) < sparsity
        new_region[:n_overlap, :n_overlap] = False
        np.fill_diagonal(new_region, False)
        adj_tgt[new_region] = 1.0

    return adj_tgt, types_tgt


def evaluate_on_task(adj, types, task_name, n_episodes=5):
    """Evaluate an SNN controller with given adjacency on a specific task."""
    from src.snn import SNNController
    cfg = TASK_CONFIGS[task_name]
    config = {k: cfg[k] for k in ('n_total', 'n_input', 'n_output',
              'tau', 'threshold', 'T_steps', 'weight_scale')}

    env = make_env(task_name)
    scores = []

    for ep in range(n_episodes):
        if task_name in ('FlappyBird', 'Snake'):
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=42000 + ep)

        ctrl = SNNController(config, adj, types)
        total = 0.0
        max_pos = obs[0] if task_name == 'MountainCar' else None

        for step in range(cfg['max_steps']):
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
                if cfg['fitness_fn'] == 'steps':
                    total += 1.0
                elif cfg['fitness_fn'] == 'acrobot_reward':
                    total += reward
                elif cfg['fitness_fn'] == 'mc_special':
                    if obs[0] > max_pos:
                        max_pos = obs[0]
                if terminated or truncated:
                    break

        if cfg['fitness_fn'] == 'mc_special':
            avg_max = max_pos if max_pos is not None else obs[0]
            fitness = (avg_max + 1.2) / 1.8 * 1000
            if avg_max >= 0.3:  fitness += 300
            if avg_max >= 0.45: fitness += 700
            if avg_max >= 0.5:  fitness += 1000
            n = adj.shape[0]
            density = adj.sum() / (n * (n - 1))
            over = max(0.0, density - cfg.get('target_sparsity', 0.15))
            fitness -= over * cfg.get('density_penalty', 200)
            scores.append(fitness)
        elif task_name == 'Snake':
            scores.append(info.get('score', 0) * 100 + total)
        elif task_name == 'FlappyBird':
            scores.append(total)
        elif cfg['fitness_fn'] == 'acrobot_reward':
            scores.append(500.0 + total)
        else:
            scores.append(total)

    if task_name not in ('FlappyBird', 'Snake'):
        env.close()

    return float(np.mean(scores))


def compute_transfer_matrix():
    """
    Build the 5x5 cross-task transfer matrix.
    transfer[i, j] = mean performance when topologies from task i
                     are evaluated on task j.
    """
    n_tasks = len(TASK_ORDER)
    raw_matrix = np.zeros((n_tasks, n_tasks))
    std_matrix = np.zeros((n_tasks, n_tasks))

    all_topologies = {}
    for src_task in TASK_ORDER:
        topologies = []
        for seed in range(N_SEEDS):
            fname = os.path.join(CROSS_DIR, f'{src_task}_seed{seed}.npz')
            if os.path.exists(fname):
                d = np.load(fname, allow_pickle=True)
                topologies.append({
                    'adj': d['adj'],
                    'types': d['types'],
                    'fitness': float(d['best_fitness']),
                    'seed': int(d['seed']),
                })
        all_topologies[src_task] = topologies
        print(f"Loaded {len(topologies)} topologies for {src_task}")

    for i, src_task in enumerate(TASK_ORDER):
        for j, tgt_task in enumerate(TASK_ORDER):
            scores = []
            topos = all_topologies[src_task]

            for topo in topos:
                if src_task == tgt_task:
                    scores.append(topo['fitness'])
                else:
                    np.random.seed(42)
                    adj_emb, types_emb = embed_topology(
                        topo['adj'], topo['types'], src_task, tgt_task)
                    score = evaluate_on_task(adj_emb, types_emb, tgt_task,
                                             n_episodes=N_EVAL)
                    scores.append(score)

            raw_matrix[i, j] = np.mean(scores)
            std_matrix[i, j] = np.std(scores)
            tag = " <-- native" if i == j else ""
            print(f"  {src_task:12s} -> {tgt_task:12s}: "
                  f"{raw_matrix[i,j]:.1f} +/- {std_matrix[i,j]:.1f}{tag}")

    return raw_matrix, std_matrix


def plot_transfer_heatmap(raw_matrix, std_matrix):
    """Create the cross-task transfer heatmap figure."""

    n_tasks = len(TASK_ORDER)

    # Normalize: divide each column by its diagonal value
    norm_matrix = np.zeros_like(raw_matrix)
    for j in range(n_tasks):
        diag_val = raw_matrix[j, j]
        if diag_val != 0:
            norm_matrix[:, j] = raw_matrix[:, j] / diag_val

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))

    # -- Panel (a): Raw performance heatmap --
    ax = axes[0]
    im = ax.imshow(raw_matrix, cmap='YlOrRd', aspect='auto')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Fitness', fontsize=11)

    for i in range(n_tasks):
        for j in range(n_tasks):
            val = raw_matrix[i, j]
            bg_val = im.norm(val)
            text_color = 'white' if bg_val > 0.6 else 'black'
            fontweight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=10, color=text_color, fontweight=fontweight)

    for k in range(n_tasks):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1, linewidth=2.5,
                              edgecolor='#1565C0', facecolor='none', linestyle='-')
        ax.add_patch(rect)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(TASK_ORDER, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(TASK_ORDER, fontsize=10)
    ax.set_xlabel('Target Task (evaluated on)', fontsize=12)
    ax.set_ylabel('Source Task (evolved on)', fontsize=12)
    ax.set_title('(a) Cross-Task Transfer: Raw Performance',
                 fontsize=13, fontweight='bold', pad=10)

    # Compute annotation stats
    off_diag_norm = []
    for ii in range(n_tasks):
        for jj in range(n_tasks):
            if ii != jj:
                off_diag_norm.append(norm_matrix[ii, jj])
    mean_off = np.mean(off_diag_norm)
    max_off = np.max(off_diag_norm)

    ax.text(0.5, -0.18,
            f'Diagonal (blue boxes) = native task performance\n'
            f'Mean off-diagonal transfer ratio: {mean_off:.2f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.8))

    # -- Panel (b): Normalized transfer ratio heatmap --
    ax = axes[1]
    norm_display = np.clip(norm_matrix, 0, 1.5)
    im2 = ax.imshow(norm_display, cmap='RdYlGn', aspect='auto',
                     vmin=0, vmax=1.2)
    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.8, pad=0.02)
    cbar2.set_label('Normalized Performance\n(transfer / native)', fontsize=11)

    for i in range(n_tasks):
        for j in range(n_tasks):
            val = norm_matrix[i, j]
            bg_val = im2.norm(np.clip(val, 0, 1.2))
            text_color = 'white' if (bg_val < 0.2 or bg_val > 0.8) else 'black'
            fontweight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, color=text_color, fontweight=fontweight)

    for k in range(n_tasks):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1, linewidth=2.5,
                              edgecolor='#1565C0', facecolor='none', linestyle='-')
        ax.add_patch(rect)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(TASK_ORDER, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(TASK_ORDER, fontsize=10)
    ax.set_xlabel('Target Task (evaluated on)', fontsize=12)
    ax.set_ylabel('Source Task (evolved on)', fontsize=12)
    ax.set_title('(b) Cross-Task Transfer: Normalized Ratio',
                 fontsize=13, fontweight='bold', pad=10)

    ax.text(0.5, -0.18,
            f'1.00 = native performance (diagonal)\n'
            f'Best cross-transfer: {max_off:.2f}, '
            f'Worst: {np.min(off_diag_norm):.2f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', alpha=0.8))

    fig.suptitle('Cross-Task Transfer Performance',
                 fontsize=15, fontweight='bold', y=1.01)

    fig.subplots_adjust(bottom=0.22, wspace=0.35)

    png_path = os.path.join(OUT_DIR, 'fig10_cross_task.png')
    pdf_path = os.path.join(OUT_DIR, 'fig10_cross_task.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    png_size = os.path.getsize(png_path) / 1024
    pdf_size = os.path.getsize(pdf_path) / 1024
    print(f"\nSaved: {png_path} ({png_size:.0f} KB)")
    print(f"Saved: {pdf_path} ({pdf_size:.0f} KB)")

    return raw_matrix, norm_matrix


if __name__ == '__main__':
    print("=" * 70)
    print("  Fig 10: Cross-Task Transfer Matrix")
    print("=" * 70)

    print("\n[Step 1] Computing cross-task transfer matrix...")
    print("  (Evaluating each topology on all 5 tasks, 5 episodes each)")
    print("  (diagonal = stored best_fitness, off-diagonal = live evaluation)\n")

    raw_matrix, std_matrix = compute_transfer_matrix()

    print("\n[Step 2] Raw transfer matrix:")
    header = "  " + " " * 12
    for t in TASK_ORDER:
        header += f"  {t[:8]:>8s}"
    print(header)
    for i, t in enumerate(TASK_ORDER):
        row = f"  {t:>12s}"
        for j in range(len(TASK_ORDER)):
            row += f"  {raw_matrix[i,j]:>8.1f}"
        print(row)

    print("\n[Step 3] Generating figure...")
    raw_out, norm_out = plot_transfer_heatmap(raw_matrix, std_matrix)

    print("\n[Step 4] Normalized transfer matrix (transfer/native):")
    header = "  " + " " * 12
    for t in TASK_ORDER:
        header += f"  {t[:8]:>8s}"
    print(header)
    for i, t in enumerate(TASK_ORDER):
        row = f"  {t:>12s}"
        for j in range(len(TASK_ORDER)):
            row += f"  {norm_out[i,j]:>8.2f}"
        print(row)

    off_diag = []
    for i in range(len(TASK_ORDER)):
        for j in range(len(TASK_ORDER)):
            if i != j:
                off_diag.append(norm_out[i, j])
    mean_transfer = np.mean(off_diag)
    print(f"\n  Mean off-diagonal transfer ratio: {mean_transfer:.3f}")
    print("  (1.0 = perfect transfer, 0.0 = no transfer)")

    print("\nDone.")
