from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('code/experiments/results/lunarlander_coevolution')
OUT = Path('outputs') / 'figures'
OUT.mkdir(exist_ok=True)
with (ROOT / 'lunarlander_coevolution_results.json').open(encoding='utf-8') as f:
    data = json.load(f)

modes = ['TO', 'TG', 'TI']
colors = {'TO': '#2F6F73', 'TG': '#C7832B', 'TI': '#6B5B95'}
returns = {m: [] for m in modes}
for row in data['per_seed']:
    returns[row['mode']].append(float(row['strict_20ep']['return_mean']))
for m in modes:
    returns[m] = np.array(returns[m], dtype=float)

summary = data['summary']
primary = {r['comparison']: r for r in summary['tost_primary']}
comparisons = ['TO-TI', 'TO-TG', 'TG-TI']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'axes.titlesize': 10,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig = plt.figure(figsize=(7.1, 3.25), dpi=300, constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.20)

# Panel a: seed-level returns with deterministic jitter.
ax = fig.add_subplot(gs[0, 0])
rng = np.random.default_rng(7)
for i, m in enumerate(modes, start=1):
    y = returns[m]
    x = np.full_like(y, i, dtype=float) + rng.uniform(-0.12, 0.12, size=len(y))
    ax.scatter(x, y, s=22, color=colors[m], alpha=0.78, edgecolor='white', linewidth=0.35, zorder=3)
    mean = y.mean()
    sem = y.std(ddof=1) / math.sqrt(len(y))
    ci = 1.96 * sem
    ax.errorbar(i, mean, yerr=ci, fmt='o', ms=5.0, color='black', ecolor='black', capsize=4, lw=1.25, zorder=4)
    ax.plot([i - 0.22, i + 0.22], [mean, mean], color='black', lw=1.35, zorder=4)
ax.axhline(0, color='#5B5B5B', lw=0.8, ls='--', alpha=0.75)
ax.axhline(200, color='#B23A48', lw=0.9, ls=':', alpha=0.9)
ax.text(3.42, 200, 'solve threshold', color='#B23A48', va='center', ha='left', fontsize=7.3)
ax.set_xlim(0.45, 3.95)
ax.set_ylim(-190, 245)
ax.set_xticks([1, 2, 3], modes)
ax.set_ylabel('Strict 20-episode return')
ax.set_title('a  LunarLander headroom audit', loc='left', fontweight='bold')
ax.grid(axis='y', color='#E7E1D8', lw=0.75)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Panel b: TOST forest plot.
ax2 = fig.add_subplot(gs[0, 1])
primary_delta = float(summary['equivalence_plan']['primary_delta'])
ax2.axvspan(-primary_delta, primary_delta, color='#DDE9E6', alpha=0.9)
ax2.axvline(0, color='#3A3A3A', lw=0.9)
for x in [-50, 50]:
    ax2.axvline(x, color='#A8A8A8', lw=0.75, ls='--', alpha=0.9)
for y, comp in zip([3, 2, 1], comparisons):
    r = primary[comp]
    diff = float(r['diff'])
    lo = float(r['ci_low'])
    hi = float(r['ci_high'])
    ax2.plot([lo, hi], [y, y], color='#2D2D2D', lw=1.5, solid_capstyle='round')
    ax2.scatter([diff], [y], s=30, color='#2D2D2D', zorder=3)
    ax2.text(103, y, f"pTOST={float(r['p_tost']):.3f}", va='center', ha='right', fontsize=7.2)
ax2.set_yticks([3, 2, 1], comparisons)
ax2.set_xlim(-82, 108)
ax2.set_xlabel('Mean difference in strict return')
ax2.set_title('b  Pre-specified TOST audit', loc='left', fontweight='bold')
ax2.grid(axis='x', color='#E7E1D8', lw=0.75)
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
ax2.text(primary_delta + 1.6, 3.39, f'primary +/-{primary_delta:.1f}', color='#2F6F73', fontsize=7.1, va='center')
ax2.text(50, 0.55, '+/-50 sensitivity', color='#777777', fontsize=7.0, ha='center')

for ext in ['pdf', 'png']:
    path = OUT / f'fig15_lunarlander_headroom.{ext}'
    fig.savefig(path, bbox_inches='tight')
    print(path)
