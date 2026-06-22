"""Generate fig08: Topology-weight co-evolution control figure."""
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────────
with open('results/coevolution/coevolution_results.json') as f:
    data = json.load(f)

per_seed = data['per_seed']

to_data = [s for s in per_seed if s['mode'] == 'TO']
tg_data = [s for s in per_seed if s['mode'] == 'TG']
ti_data = [s for s in per_seed if s['mode'] == 'TI']

to_best = [s['best_fitness'] for s in to_data]
tg_best = [s['best_fitness'] for s in tg_data]
ti_best = [s['best_fitness'] for s in ti_data]

to_gen = [s['generations'] for s in to_data]
tg_gen = [s['generations'] for s in tg_data]
ti_gen = [s['generations'] for s in ti_data]

n_params = {'TO': 1, 'TG': 2, 'TI': 16384}

# ── ANOVA ──────────────────────────────────────────────────────────────
F, p = stats.f_oneway(to_best, tg_best, ti_best)
print(f"ANOVA: F(2,27)={F:.2f}, p={p:.2f}")
print(f"TO: {statistics.mean(to_best):.1f}±{statistics.stdev(to_best):.1f}")
print(f"TG: {statistics.mean(tg_best):.1f}±{statistics.stdev(tg_best):.1f}")
print(f"TI: {statistics.mean(ti_best):.1f}±{statistics.stdev(ti_best):.1f}")

# ── Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
colors = ['#E74C3C', '#3498DB', '#2ECC71']

# (a) Best fitness boxplot
ax = axes[0]
bp = ax.boxplot([to_best, tg_best, ti_best], labels=['TO', 'TG', 'TI'],
                patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for i, vals in enumerate([to_best, tg_best, ti_best]):
    ax.scatter(np.random.normal(i + 1, 0.04, len(vals)), vals,
               alpha=0.6, s=30, color=colors[i], edgecolors='black', linewidth=0.5, zorder=5)
ax.axhline(y=22.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax.set_ylabel('Best Fitness (20-ep mean)')
ax.set_title(f'(a) Final Performance\n$F(2,27)={F:.2f}$, $p={p:.2f}$', fontsize=11)
ax.set_ylim(0, 530)
ax.legend(loc='lower right', fontsize=8)
ax.text(1, statistics.mean(to_best) + 8, f'{statistics.mean(to_best):.1f}',
        ha='center', fontsize=9, fontweight='bold')
ax.text(2, statistics.mean(tg_best) + 8, f'{statistics.mean(tg_best):.1f}',
        ha='center', fontsize=9, fontweight='bold')
ax.text(3, statistics.mean(ti_best) + 8, f'{statistics.mean(ti_best):.1f}',
        ha='center', fontsize=9, fontweight='bold')

# (b) Convergence generations
ax = axes[1]
modes = ['TO', 'TG', 'TI']
gen_means = [statistics.mean(to_gen), statistics.mean(tg_gen), statistics.mean(ti_gen)]
gen_stds = [statistics.stdev(to_gen), statistics.stdev(tg_gen), statistics.stdev(ti_gen)]
bars = ax.bar(modes, gen_means, yerr=gen_stds, capsize=5,
              color=colors, alpha=0.6, edgecolor='black', linewidth=0.8)
for i, (m, s) in enumerate(zip(gen_means, gen_stds)):
    ax.text(i, m + s + 5, f'{m:.0f}±{s:.0f}', ha='center', fontsize=9)
ax.set_ylabel('Generations to Solve')
ax.set_title('(b) Convergence Speed', fontsize=11)
ax.set_ylim(0, max(g + s for g, s in zip(gen_means, gen_stds)) * 1.25)

# (c) Parameter efficiency (log scale)
ax = axes[2]
eff_vals = [statistics.mean(to_best) / n_params[m] for m in modes]
bars = ax.bar(modes, eff_vals, color=colors, alpha=0.6, edgecolor='black', linewidth=0.8)
ax.set_yscale('log')
for i, v in enumerate(eff_vals):
    if v >= 1:
        ax.text(i, v * 1.3, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
    else:
        ax.text(i, v * 1.3, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Performance / #Params (log)')
ax.set_title('(c) Parameter Efficiency', fontsize=11)
ax.text(0.5, -0.15, '1 param', ha='center', fontsize=8, transform=ax.get_xaxis_transform())
ax.text(1.5, -0.15, '2 params', ha='center', fontsize=8, transform=ax.get_xaxis_transform())
ax.text(2.5, -0.15, '~16K params', ha='center', fontsize=8, transform=ax.get_xaxis_transform())

plt.tight_layout()

# Save
out_dir = '../figures_arxiv'
import os
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f'{out_dir}/fig08_coevolution.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{out_dir}/fig08_coevolution.pdf', bbox_inches='tight')
print(f"\nSaved to {out_dir}/fig08_coevolution.png/pdf")
plt.close()
