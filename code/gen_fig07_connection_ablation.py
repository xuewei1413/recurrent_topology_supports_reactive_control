"""
Regenerate fig07 (connection ablation) with CORRECT input scaling.

CRITICAL BUG FIX: The GA evaluates with ctrl.forward(obs * INPUT_SCALE) where
INPUT_SCALE=4.0.  This means encode_state receives obs*4.0 and internally does
tanh(obs*4.0)*4.0.  The previous figure used raw obs → tanh(obs)*4.0, yielding
baseline ~130 instead of ~490.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from src.snn import SNNController
import gymnasium as gym

# ── Config ─────────────────────────────────────────────────────────────────
INPUT_SCALE = 4.0
CONFIG = {
    'n_total': 128,
    'n_input': 8,
    'n_output': 2,
    'T_steps': 20,
    'tau': 5.0,
    'threshold': 0.8,
    'weight_scale': 0.6,
    'dt': 1.0,
}
TOPO_PATH = os.path.join(os.path.dirname(__file__),
                         'experiments', 'results', 'cross_task', 'CartPole_seed0.npz')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
N_EVAL = 20
N_RANDOM = 10

# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(adj, neuron_types, n_episodes=N_EVAL):
    """Evaluate topology on CartPole-v1 with CORRECT input scaling."""
    ctrl = SNNController(CONFIG, adj, neuron_types)
    env = gym.make('CartPole-v1')
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10000 + ep)
        ctrl.reset()
        for step in range(500):
            action = ctrl.forward(np.array(obs, dtype=np.float64) * INPUT_SCALE)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
    env.close()
    return total / n_episodes


def ablate_type(adj, types, src_type, dst_type):
    """Zero out connections from src_type to dst_type. Returns modified copy."""
    mask = (types[:, None] == src_type) & (types[None, :] == dst_type)
    adj_new = adj.copy()
    adj_new[mask] = 0
    return adj_new


def count_type(adj, types, src_type, dst_type):
    mask = (types[:, None] == src_type) & (types[None, :] == dst_type)
    return int(np.sum(mask & (adj > 0)))


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    # Load topology
    data = np.load(TOPO_PATH)
    adj = data['adj'].astype(np.float64)
    types = data['types'].astype(np.float64)
    print(f"Loaded topology: adj {adj.shape}, types {types.shape}")
    print(f"  Excitatory: {np.sum(types==1)}, Inhibitory: {np.sum(types==-1)}")

    # Connection type counts
    conn_types = [(1, 1, 'E->E'), (1, -1, 'E->I'), (-1, 1, 'I->E'), (-1, -1, 'I->I')]
    for src, dst, name in conn_types:
        n = count_type(adj, types, src, dst)
        print(f"  {name}: {n} connections")

    # Baseline
    print(f"\nEvaluating baseline ({N_EVAL} episodes)...")
    baseline = evaluate(adj, types)
    print(f"  Baseline score: {baseline:.1f}")

    # Specific ablations
    results = []
    all_edges = list(zip(*np.where(adj > 0)))
    for src, dst, name in conn_types:
        n_edges = count_type(adj, types, src, dst)
        adj_abl = ablate_type(adj, types, src, dst)
        score = evaluate(adj_abl, types)
        drop = baseline - score
        print(f"  {name}: score={score:.1f}, drop={drop:.1f} (removed {n_edges} edges)")

        # Random control: remove same number of random edges
        rand_drops = []
        if n_edges > 0 and len(all_edges) >= n_edges:
            for r in range(N_RANDOM):
                rand_idx = np.random.choice(len(all_edges), size=n_edges, replace=False)
                rand_edges = [all_edges[i] for i in rand_idx]
                adj_rand = adj.copy()
                for (i, j) in rand_edges:
                    adj_rand[i, j] = 0
                rand_score = evaluate(adj_rand, types)
                rand_drops.append(baseline - rand_score)
        rand_drops = np.array(rand_drops)
        rand_mean = float(rand_drops.mean()) if len(rand_drops) > 0 else 0.0
        rand_std = float(rand_drops.std()) if len(rand_drops) > 0 else 0.0

        results.append({
            'name': name, 'n_edges': n_edges,
            'score': score, 'drop': drop,
            'rand_mean': rand_mean, 'rand_std': rand_std,
        })

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Type':<8} {'Score':>8} {'Drop':>8} {'RandDrop':>14} {'Spec':>8}")
    print("-" * 65)
    for r in results:
        spec = (r['drop'] - r['rand_mean']) / r['rand_mean'] if abs(r['rand_mean']) > 0.1 else float('inf')
        print(f"{r['name']:<8} {r['score']:>8.1f} {r['drop']:>8.1f} "
              f"{r['rand_mean']:>7.1f}+/-{r['rand_std']:<5.1f} {spec:>8.2f}")
    print("=" * 65)

    # ── Plot ────────────────────────────────────────────────────────────────
    labels = [r['name'] for r in results]
    # Add connection count to labels
    pct_edges = []
    total_edges = sum(r['n_edges'] for r in results)
    for r in results:
        pct = 100 * r['n_edges'] / total_edges if total_edges > 0 else 0
        pct_edges.append(pct)
    display_labels = [f"{r['name']}\n({pct_edges[i]:.0f}%)" for i, r in enumerate(results)]

    specific_drops = [r['drop'] for r in results]
    random_means = [r['rand_mean'] for r in results]
    random_stds = [r['rand_std'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: Performance Drop ---
    x = np.arange(len(labels))
    width = 0.35

    colors_specific = ['#C0392B', '#E67E22', '#2980B9', '#8E44AD']
    bars1 = ax1.bar(x - width / 2, specific_drops, width, label='Specific Ablation',
                    color=colors_specific, edgecolor='black', zorder=3)
    bars2 = ax1.bar(x + width / 2, random_means, width, yerr=random_stds,
                    label='Random Control (mean$\\pm$std)', color='#BDC3C7',
                    edgecolor='black', capsize=4, zorder=3)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, rm in zip(bars2, random_means):
        h = bar.get_height()
        ax1.annotate(f'{rm:.1f}', xy=(bar.get_x() + bar.get_width() / 2, max(h, 0)),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8, color='#2C3E50')
    ax1.set_ylabel('Performance Drop', fontsize=12)
    ax1.set_title('Connection-Type Ablation', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # --- Right panel: Post-ablation score + specificity ---
    ax2_twin = ax2.twinx()
    post_scores = [r['score'] for r in results]
    rand_scores = [baseline - r['rand_mean'] for r in results]
    specificities = []
    for r in results:
        if abs(r['rand_mean']) > 0.1:
            specificities.append((r['drop'] - r['rand_mean']) / r['rand_mean'])
        else:
            specificities.append(0.0)

    bars_post = ax2.bar(x - width / 2, post_scores, width,
                        color=colors_specific, edgecolor='black',
                        zorder=3, alpha=0.85, label='Post-ablation score')
    ax2.bar(x + width / 2, rand_scores, width,
            color='#BDC3C7', edgecolor='black', alpha=0.6, zorder=3,
            label='Random control score')
    ax2.axhline(baseline, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.0f})')
    for bar, score in zip(bars_post, post_scores):
        ax2.annotate(f'{score:.0f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2_twin.plot(x, specificities, 'D-', color='darkorange', markersize=10, linewidth=2,
                  label='Specificity Index', zorder=5)
    ax2_twin.axhline(0, color='gray', linestyle=':', alpha=0.5)
    for i, spec in enumerate(specificities):
        color = '#27AE60' if spec > 0 else '#E74C3C'
        offset_y = 10 if spec > 0 else -15
        ax2_twin.annotate(f'{spec:+.2f}', xy=(i, spec),
                          xytext=(0, offset_y), textcoords='offset points',
                          ha='center', fontsize=9, fontweight='bold', color=color)

    ax2.set_ylabel('Post-ablation Performance', fontsize=12)
    ax2_twin.set_ylabel('Specificity Index', fontsize=12, color='darkorange')
    ax2.set_title(f'Post-ablation Score (Baseline: {baseline:.0f})', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_labels, fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3, zorder=0)

    plt.tight_layout()

    # Save
    png_path = os.path.join(OUT_DIR, 'fig07_connection_ablation.png')
    pdf_path = os.path.join(OUT_DIR, 'fig07_connection_ablation.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    # Report file sizes
    png_size = os.path.getsize(png_path)
    pdf_size = os.path.getsize(pdf_path)
    print(f"\nSaved: {png_path} ({png_size:,} bytes)")
    print(f"Saved: {pdf_path} ({pdf_size:,} bytes)")


if __name__ == '__main__':
    main()
