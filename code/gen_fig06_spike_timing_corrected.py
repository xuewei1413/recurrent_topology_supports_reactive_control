"""
Regenerate Fig06 (spike timing) with CORRECT input scaling.

CRITICAL BUG FIX: The cross-task topology from CartPole_seed0.npz was evolved
with `ctrl.forward(obs * INPUT_SCALE)` where INPUT_SCALE=4.0. The previous
figure was generated WITHOUT this external scaling, producing incorrect
t-statistics.

This script:
  1. Loads CartPole seed0 topology from cross_task results
  2. Runs 200 CartPole episodes, collecting output neuron spike timing
  3. Filters for timesteps where pole is near balanced (|angle| < threshold)
  4. Compares spike timing between opposite angular velocity groups
  5. CRITICAL: passes `obs * 4.0` as the state to encode_state (INPUT_SCALE=4.0)
  6. Computes t-test statistics
  7. Generates publication-quality figure with boxplots

Expected result from paper: t=24.16, p<0.0001
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import sys, os

# ── Paths ────────────────────────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from src.snn import SNNController

INPUT_SCALE = 4.0
SEED = 42

TOPO_PATH = os.path.join(CODE_DIR, 'experiments', 'results', 'cross_task', 'CartPole_seed0.npz')
OUT_DIR = os.path.join(os.path.dirname(CODE_DIR), 'figures_arxiv')

# ── Config ───────────────────────────────────────────────────────────────
config = {
    'n_total': 128, 'n_input': 8, 'n_output': 2,
    'sparsity': 0.10, 'weight_scale': 0.6,
    'tau': 5.0, 'threshold': 0.8, 'T_steps': 20, 'dt': 1.0,
}


def forward_with_trace(controller, state):
    """Run T_steps forward pass recording spike trace.

    state: already-scaled observation (obs * INPUT_SCALE)
    """
    n = controller.n
    T = controller.T
    dt = controller.dt
    tau = controller.tau

    v = np.zeros(n)
    spikes = np.zeros(n)
    spike_trace = np.zeros((T, n))

    I_ext = controller.encode_state(state)

    for t in range(T):
        v *= (1 - dt / tau)
        if spikes.any():
            v += spikes @ controller.W
        v[:controller.n_in] += I_ext * dt / tau
        fired = v >= controller.thresh
        spikes = fired.astype(np.float64)
        v[fired] = 0.0
        spike_trace[t] = spikes.copy()

    action = int(np.argmax(spike_trace.sum(axis=0)[-controller.n_out:]))
    return spike_trace, action


def collect_data(adj, types, config, n_episodes=200):
    """Collect spike timing data with correct INPUT_SCALE."""
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    records = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        ctrl = SNNController(config, adj, types)

        for step in range(500):
            # CRITICAL: apply INPUT_SCALE before passing to forward
            scaled_state = np.array(obs, dtype=np.float64) * INPUT_SCALE
            spike_trace, action = forward_with_trace(ctrl, scaled_state)

            output_spikes = spike_trace[:, -config['n_output']:]

            # First spike times per output neuron
            first_spike = []
            for out_idx in range(config['n_output']):
                times = np.where(output_spikes[:, out_idx] > 0)[0]
                first_spike.append(int(times[0]) if len(times) > 0 else -1)

            spike_counts = output_spikes.sum(axis=0).astype(int).tolist()

            records.append({
                'pole_angle': float(obs[2]),
                'pole_vel': float(obs[3]),
                'first_spike': first_spike,
                'spike_counts': spike_counts,
            })

            obs, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    env.close()
    return records


def analyze_and_plot(records, save_dir):
    """Filter, compute statistics, and generate the figure."""
    # Filter for balanced pole
    for thresh in [0.05, 0.10]:
        filtered = [r for r in records if abs(r['pole_angle']) < thresh]
        if len(filtered) >= 30:
            break

    pos_vel = [r for r in filtered if r['pole_vel'] > 0]
    neg_vel = [r for r in filtered if r['pole_vel'] < 0]

    print(f"Angle threshold: {thresh}")
    print(f"  Near-balanced samples: {len(filtered)} / {len(records)}")
    print(f"  pos_vel (vel>0): {len(pos_vel)}")
    print(f"  neg_vel (vel<0): {len(neg_vel)}")

    def extract_times(group, idx):
        return np.array([r['first_spike'][idx] for r in group if r['first_spike'][idx] >= 0])

    t_results = {}
    for name, idx in [('left', 0), ('right', 1)]:
        tp = extract_times(pos_vel, idx)
        tn = extract_times(neg_vel, idx)
        t_val, p_val = stats.ttest_ind(tp, tn)
        t_results[name] = {
            't': t_val, 'p': p_val,
            'pos_mean': tp.mean(), 'neg_mean': tn.mean(),
            'pos_std': tp.std(), 'neg_std': tn.std(),
            'tp': tp, 'tn': tn,
            'n_pos': len(tp), 'n_neg': len(tn),
        }
        print(f"\n  {name} neuron:")
        print(f"    vel>0: mean={tp.mean():.2f}, std={tp.std():.2f}, n={len(tp)}")
        print(f"    vel<0: mean={tn.mean():.2f}, std={tn.std():.2f}, n={len(tn)}")
        print(f"    t={t_val:.2f}, p={p_val:.6f}")

    # Win rate analysis
    pos_rw = sum(1 for r in pos_vel if r['spike_counts'][1] > r['spike_counts'][0]) / len(pos_vel)
    neg_rw = sum(1 for r in neg_vel if r['spike_counts'][1] > r['spike_counts'][0]) / len(neg_vel)
    print(f"\n  Right neuron wins: vel>0={pos_rw:.1%}, vel<0={neg_rw:.1%}")

    # ── Figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Boxplot + t-stats
    ax = axes[0]
    data_all = [
        t_results['left']['tn'], t_results['left']['tp'],
        t_results['right']['tn'], t_results['right']['tp'],
    ]
    labels = ['Left\n(vel<0)', 'Left\n(vel>0)', 'Right\n(vel<0)', 'Right\n(vel>0)']
    colors_bp = ['#3498DB', '#85C1E9', '#E74C3C', '#F1948A']
    bp = ax.boxplot(data_all, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('First Spike Time (within T)', fontsize=11)
    ax.set_title(f'(a) Output First Spike Time\n(|pole_angle|<{thresh})',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Annotate t-stats
    p_str_left = 'p<0.0001' if t_results['left']['p'] < 0.0001 else f"p={t_results['left']['p']:.4f}"
    p_str_right = 'p<0.0001' if t_results['right']['p'] < 0.0001 else f"p={t_results['right']['p']:.4f}"
    ax.annotate(f't={t_results["left"]["t"]:.2f}\n{p_str_left}',
                xy=(0.25, 0.95), xycoords='axes fraction', fontsize=8, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.annotate(f't={t_results["right"]["t"]:.2f}\n{p_str_right}',
                xy=(0.75, 0.95), xycoords='axes fraction', fontsize=8, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (b) Mean comparison
    ax = axes[1]
    categories = ['Left neuron', 'Right neuron']
    neg_means = [t_results['left']['neg_mean'], t_results['right']['neg_mean']]
    pos_means = [t_results['left']['pos_mean'], t_results['right']['pos_mean']]
    neg_stds = [t_results['left']['neg_std'], t_results['right']['neg_std']]
    pos_stds = [t_results['left']['pos_std'], t_results['right']['pos_std']]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, neg_means, width, yerr=neg_stds, label='vel<0',
           color='#3498DB', alpha=0.8, capsize=4)
    ax.bar(x + width / 2, pos_means, width, yerr=pos_stds, label='vel>0',
           color='#E74C3C', alpha=0.8, capsize=4)
    ax.set_ylabel('Mean First Spike Time', fontsize=11)
    ax.set_title('(b) Mean First Spike Time by History', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Significance brackets
    for i, name in enumerate(['left', 'right']):
        p_val = t_results[name]['p']
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
        y_max = max(neg_means[i] + neg_stds[i], pos_means[i] + pos_stds[i])
        ax.plot([x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
                [y_max + 0.2, y_max + 0.4, y_max + 0.4, y_max + 0.2], 'k-', linewidth=0.8)
        ax.text(x[i], y_max + 0.5, sig, ha='center', fontsize=10, fontweight='bold')

    # (c) Win rate
    ax = axes[2]
    conditions_wr = ['vel>0\n(right tilt)', 'vel<0\n(left tilt)']
    right_rates = [pos_rw, neg_rw]
    left_rates = [1 - r for r in right_rates]
    x = np.arange(len(conditions_wr))
    ax.bar(x, right_rates, 0.5, label='Right neuron wins', color='#E74C3C', alpha=0.8)
    ax.bar(x, left_rates, 0.5, bottom=right_rates, label='Left neuron wins', color='#3498DB', alpha=0.8)
    for i, (r, l) in enumerate(zip(right_rates, left_rates)):
        ax.text(i, r / 2, f'{r:.1%}', ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
        ax.text(i, r + l / 2, f'{l:.1%}', ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
    ax.set_ylabel('Proportion', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions_wr)
    ax.set_title('(c) Spike Count Winner by History', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, 'fig06_spike_timing.png')
    pdf_path = os.path.join(save_dir, 'fig06_spike_timing.pdf')

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    png_size = os.path.getsize(png_path) / 1024
    pdf_size = os.path.getsize(pdf_path) / 1024
    print(f"\nSaved: {png_path} ({png_size:.0f} KB)")
    print(f"Saved: {pdf_path} ({pdf_size:.0f} KB)")

    return t_results


if __name__ == '__main__':
    print("=" * 60)
    print("Fig 06: Spike timing with CORRECT input scaling (INPUT_SCALE=4.0)")
    print("=" * 60)

    # Load cross-task topology
    data = np.load(TOPO_PATH, allow_pickle=True)
    adj = data['adj']
    types = data['types']
    print(f"Loaded topology: adj {adj.shape}, types {types.shape}")
    print(f"  Adjacency density: {adj.sum() / (adj.shape[0] * (adj.shape[0] - 1)):.4f}")
    print(f"  E/I ratio: {(types > 0).sum()} / {(types < 0).sum()}")

    np.random.seed(SEED)
    print(f"\nCollecting data (200 episodes, SEED={SEED})...")
    records = collect_data(adj, types, config, n_episodes=200)
    print(f"Total timesteps: {len(records)}")

    print("\n" + "-" * 50)
    t_results = analyze_and_plot(records, OUT_DIR)
    print("-" * 50)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name in ['left', 'right']:
        r = t_results[name]
        print(f"  {name} neuron: t={r['t']:.2f}, p={r['p']:.6f}, "
              f"n_pos={r['n_pos']}, n_neg={r['n_neg']}")
    print("=" * 60)
