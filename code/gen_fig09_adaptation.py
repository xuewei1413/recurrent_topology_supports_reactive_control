"""
Generate Fig 09: Adaptation Strategies Comparison (CartPole, gravity perturbation)

Load CartPole seed0 topology, test adaptation to gravity change (9.8 -> 15.0):
  1. Standard baseline (gravity=9.8)
  2. Zero-shot (gravity=15.0, no adaptation)
  3. Tune weight_scale (gravity=15.0, sweep ws from 0.3 to 1.4)
  4. Tune T_steps (gravity=15.0, sweep T from 5 to 50)

CRITICAL: Uses obs * INPUT_SCALE (4.0) before passing to controller.forward(),
matching the evaluation pattern used in gen_fig10_cross_task.py.
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.snn import SNNController

# ── Configuration ──────────────────────────────────────────────────────
INPUT_SCALE = 4.0

SNN_CONFIG = {
    'n_total': 128, 'n_input': 8, 'n_output': 2,
    'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
    'weight_scale': 0.6,
}

GRAVITY_STANDARD = 9.8
GRAVITY_MUTANT = 15.0
N_EVAL_EPISODES = 20
MAX_STEPS = 500

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TOPO_PATH = os.path.join(CODE_DIR, 'experiments', 'results', 'cross_task', 'CartPole_seed0.npz')
OUT_DIR = os.path.join(os.path.dirname(CODE_DIR), 'figures_arxiv')
os.makedirs(OUT_DIR, exist_ok=True)


def evaluate(adj, types, config, gravity, n_episodes=N_EVAL_EPISODES):
    """Evaluate SNN controller with given config on CartPole with specified gravity."""
    controller = SNNController(config, adj, types)
    env = gym.make('CartPole-v1')
    env.unwrapped.gravity = gravity
    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=50000 + ep)
        controller.reset()
        total = 0.0
        for _ in range(MAX_STEPS):
            action = controller.forward(np.array(obs, dtype=np.float64) * INPUT_SCALE)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
    env.close()
    return np.array(scores)


def main():
    print("=" * 70)
    print("  Fig 09: Adaptation Strategies (CartPole gravity 9.8 -> 15.0)")
    print("  Input scaling: obs * INPUT_SCALE (4.0) before forward()")
    print("=" * 70)

    # Load topology
    d = np.load(TOPO_PATH)
    adj = d['adj']
    types = d['types']
    print(f"Loaded CartPole seed0: fitness={float(d['best_fitness']):.1f}, "
          f"conns={int(adj.sum())}, seed={int(d['seed'])}")

    config = SNN_CONFIG.copy()

    # ── 1. Standard baseline (gravity=9.8) ──────────────────────────────
    print("\n[1] Standard baseline (gravity=9.8, ws=0.6, T=20)...")
    baseline = evaluate(adj, types, config, GRAVITY_STANDARD)
    baseline_mean = baseline.mean()
    baseline_std = baseline.std()
    print(f"  Baseline: {baseline_mean:.1f} +/- {baseline_std:.1f}")

    # ── 2. Zero-shot (gravity=15.0, no changes) ────────────────────────
    print("\n[2] Zero-shot (gravity=15.0, ws=0.6, T=20)...")
    zero_shot = evaluate(adj, types, config, GRAVITY_MUTANT)
    zero_mean = zero_shot.mean()
    zero_std = zero_shot.std()
    zero_recovery = zero_mean / baseline_mean * 100
    print(f"  Zero-shot: {zero_mean:.1f} +/- {zero_std:.1f}  ({zero_recovery:.1f}%)")

    # ── 3. Sweep weight_scale ──────────────────────────────────────────
    print("\n[3] Sweeping weight_scale (gravity=15.0)...")
    ws_range = np.arange(0.3, 1.41, 0.1)
    ws_results = []
    best_ws = None
    best_ws_score = -np.inf
    best_ws_scores = None
    for ws in ws_range:
        cfg = config.copy()
        cfg['weight_scale'] = float(ws)
        scores = evaluate(adj, types, cfg, GRAVITY_MUTANT)
        m = scores.mean()
        s = scores.std()
        ws_results.append({'ws': float(ws), 'mean': m, 'std': s})
        marker = ""
        if m > best_ws_score:
            best_ws_score = m
            best_ws = float(ws)
            best_ws_scores = scores
            marker = " *"
        print(f"  ws={ws:.1f}: {m:.1f} +/- {s:.1f}{marker}")
    ws_recovery = best_ws_score / baseline_mean * 100
    print(f"  Best weight_scale: {best_ws:.1f} -> {best_ws_score:.1f} ({ws_recovery:.1f}%)")

    # ── 4. Sweep T_steps ───────────────────────────────────────────────
    print("\n[4] Sweeping T_steps (gravity=15.0)...")
    ts_range = [5, 10, 15, 20, 25, 30, 40, 50]
    ts_results = []
    best_ts = None
    best_ts_score = -np.inf
    best_ts_scores = None
    for ts in ts_range:
        cfg = config.copy()
        cfg['T_steps'] = ts
        scores = evaluate(adj, types, cfg, GRAVITY_MUTANT)
        m = scores.mean()
        s = scores.std()
        ts_results.append({'ts': ts, 'mean': m, 'std': s})
        marker = ""
        if m > best_ts_score:
            best_ts_score = m
            best_ts = ts
            best_ts_scores = scores
            marker = " *"
        print(f"  T={ts:3d}: {m:.1f} +/- {s:.1f}{marker}")
    ts_recovery = best_ts_score / baseline_mean * 100
    print(f"  Best T_steps: {best_ts} -> {best_ts_score:.1f} ({ts_recovery:.1f}%)")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Baseline (g=9.8):            {baseline_mean:.1f} +/- {baseline_std:.1f}")
    print(f"  Zero-shot (g=15.0):          {zero_mean:.1f} +/- {zero_std:.1f}  ({zero_recovery:.1f}%)")
    print(f"  Tune weight_scale (ws={best_ws:.1f}): {best_ws_score:.1f} +/- {best_ws_scores.std():.1f}  ({ws_recovery:.1f}%)")
    print(f"  Tune T_steps (T={best_ts}):         {best_ts_score:.1f} +/- {best_ts_scores.std():.1f}  ({ts_recovery:.1f}%)")

    # ── Generate Figure ─────────────────────────────────────────────────
    print("\nGenerating figure...")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # -- Panel (a): weight_scale sweep --
    ax = axes[0]
    ws_vals = [r['ws'] for r in ws_results]
    ws_means = [r['mean'] for r in ws_results]
    ws_stds = [r['std'] for r in ws_results]

    ax.fill_between(ws_vals,
                    [m - s for m, s in zip(ws_means, ws_stds)],
                    [m + s for m, s in zip(ws_means, ws_stds)],
                    alpha=0.15, color='steelblue')
    ax.plot(ws_vals, ws_means, 'o-', color='steelblue', linewidth=2, markersize=7,
            label='weight scale sweep')
    ax.axhline(baseline_mean, color='green', ls='--', linewidth=1.5,
               label=f'Baseline (g=9.8): {baseline_mean:.0f}')
    ax.axhline(zero_mean, color='red', ls=':', linewidth=1.5,
               label=f'Zero-shot (g=15.0): {zero_mean:.0f}')
    ax.plot(best_ws, best_ws_score, '*', color='darkblue', markersize=18, zorder=5,
            label=f'Best ws={best_ws:.1f}: {best_ws_score:.0f}')
    ax.set_xlabel('weight scale', fontsize=12)
    ax.set_ylabel('Score (20-ep mean)', fontsize=12)
    ax.set_title('(a) Structural Adaptation: weight scale', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(ws_vals[0] - 0.05, ws_vals[-1] + 0.05)

    # -- Panel (b): T_steps sweep --
    ax = axes[1]
    ts_vals = [r['ts'] for r in ts_results]
    ts_means = [r['mean'] for r in ts_results]
    ts_stds = [r['std'] for r in ts_results]

    ax.fill_between(ts_vals,
                    [m - s for m, s in zip(ts_means, ts_stds)],
                    [m + s for m, s in zip(ts_means, ts_stds)],
                    alpha=0.15, color='darkorange')
    ax.plot(ts_vals, ts_means, 's-', color='darkorange', linewidth=2, markersize=7,
            label=r'$T_\mathrm{steps}$ sweep')
    ax.axhline(baseline_mean, color='green', ls='--', linewidth=1.5,
               label=f'Baseline (g=9.8): {baseline_mean:.0f}')
    ax.axhline(zero_mean, color='red', ls=':', linewidth=1.5,
               label=f'Zero-shot (g=15.0): {zero_mean:.0f}')
    ax.plot(best_ts, best_ts_score, '*', color='darkred', markersize=18, zorder=5,
            label=f'Best T={best_ts}: {best_ts_score:.0f}')
    ax.set_xlabel(r'$T_\mathrm{steps}$', fontsize=12)
    ax.set_ylabel('Score (20-ep mean)', fontsize=12)
    ax.set_title(r'(b) Structural Adaptation: $T_\mathrm{steps}$', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    # -- Panel (c): Strategy comparison bar chart --
    ax = axes[2]
    strategies = ['Baseline\n(g=9.8)', 'Zero-shot\n(g=15.0)',
                  f'Tune ws\n(ws={best_ws:.1f})', f'Tune T\n(T={best_ts})']
    means = [baseline_mean, zero_mean, best_ws_score, best_ts_score]
    stds_list = [baseline_std, zero_std, best_ws_scores.std(), best_ts_scores.std()]
    recoveries = [100.0, zero_recovery, ws_recovery, ts_recovery]
    colors = ['#4CAF50', '#F44336', '#FF9800', '#FFC107']

    bars = ax.bar(strategies, means, color=colors, edgecolor='black', linewidth=0.8,
                  yerr=stds_list, capsize=5, error_kw={'linewidth': 1.2})
    ax.axhline(baseline_mean, color='green', ls='--', linewidth=1.5, alpha=0.5)

    ymax = max(means) * 1.15
    for bar, m, rec in zip(bars, means, recoveries):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02,
                f'{m:.0f}\n{rec:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.set_ylabel('Score (20-ep mean)', fontsize=12)
    ax.set_title('(c) Adaptation Strategy Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, ymax)

    # Add parameter count annotation
    ax.text(0.5, -0.22,
            'Baseline = standard env (g=9.8); percentages are descriptive ratios to baseline\n'
            'Each adaptation adjusts only 1 global parameter; changed-gravity ceiling may differ',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.8))

    fig.suptitle('Environmental Perturbation Adaptation (CartPole-v1, gravity 9.8 $\\rightarrow$ 15.0)',
                 fontsize=14, fontweight='bold', y=1.01)

    fig.subplots_adjust(bottom=0.20, wspace=0.35)

    png_path = os.path.join(OUT_DIR, 'fig09_adaptation.png')
    pdf_path = os.path.join(OUT_DIR, 'fig09_adaptation.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    png_size = os.path.getsize(png_path) / 1024
    pdf_size = os.path.getsize(pdf_path) / 1024
    print(f"\nSaved: {png_path} ({png_size:.0f} KB)")
    print(f"Saved: {pdf_path} ({pdf_size:.0f} KB)")

    # ── Final Report ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    print(f"  Baseline (g=9.8):             {baseline_mean:.1f} +/- {baseline_std:.1f}")
    print(f"  Zero-shot (g=15.0):           {zero_mean:.1f} +/- {zero_std:.1f}  recovery={zero_recovery:.1f}%")
    print(f"  Tune weight_scale (ws={best_ws:.1f}):  {best_ws_score:.1f} +/- {best_ws_scores.std():.1f}  recovery={ws_recovery:.1f}%")
    print(f"  Tune T_steps (T={best_ts}):          {best_ts_score:.1f} +/- {best_ts_scores.std():.1f}  recovery={ts_recovery:.1f}%")
    print(f"  PNG: {png_size:.0f} KB")
    print(f"  PDF: {pdf_size:.0f} KB")
    print("Done.")


if __name__ == '__main__':
    main()
