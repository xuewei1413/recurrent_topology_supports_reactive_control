"""Refined preview of the approved overview; schematic waveforms, real scatter."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
from scipy.stats import spearmanr

from make_overview_integration_sketch import network

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'paper/figures/v5/figure2_source_data.csv'
OUT = ROOT / 'paper/figures/v5/overview_integration_refined'
W, H = 240, 151
INK, MUTED, LINE = '#26333D', '#65747C', '#C5CED2'
TEAL, LESION, EXC, INH = '#197F78', '#B45C47', '#A67B2C', '#4C78A8'


def label(ax, x, y, s, size=8, color=INK, ha='left', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va='center', **kw)


def arrow(ax, p, q, color=MUTED, dash=False, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7,
                 lw=.8, color=color, linestyle=(0, (3, 3)) if dash else 'solid', **kw))


def header(ax, x, y, letter, title):
    label(ax, x, y, letter, 12, fontweight='bold')
    label(ax, x+8, y, title, 10)


def wave(ax, x, y, width, events):
    # These normalized drawings express threshold/reset, not numerical model dynamics.
    threshold = 7.5
    ax.plot([x, x+width], [y+threshold]*2, color=LINE, lw=.65, ls=(0, (2, 2)))
    xs, ys, prev = [x], [y], 0
    for e in events:
        start, end = x+prev*width, x+e*width
        xs.extend([start+.28*(end-start), start+.5*(end-start), end, end])
        ys.extend([y+3.3, y+2.3, y+threshold, y])
        prev = e
    xs.append(x+width)
    ys.append(y+1.4)
    ax.plot(xs, ys, color=INK, lw=1)
    ax.plot([x, x+width], [y-1]*2, color=LINE, lw=.55)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['panel'] == 'a']
    xy = np.array([[float(r['background_idx']), float(r['J24'])] for r in rows])
    rho = float(spearmanr(xy[:, 0], xy[:, 1]).statistic)
    assert len(xy) == 25 and len({r['seed'] for r in rows}) == 25
    expected = json.loads(SOURCE.with_name('figure2_statistics.json').read_text(encoding='utf-8'))['scatter']['main']['rho']
    assert np.isfinite(xy).all() and abs(rho-expected) < 1e-12
    with plt.rc_context({'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
                         'svg.fonttype': 'none', 'svg.hashsalt': 'integration-refined',
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, H))
        ax.set_aspect('equal')
        ax.axis('off')
        header(ax, 7, 144, 'A', '结构干预')
        header(ax, 108, 144, 'B', '输出端的积分与动作读出')

        # The parent and both variants share positions. The network is illustrative.
        label(ax, 51, 134, '固定突触幅值，仅进化连接', 8, ha='center')
        ax.plot([8, 26], [115, 115], color=LINE, lw=.8)
        ax.add_patch(Rectangle((12, 117), 10, 4, ec=INK, fc='#F2F5F6', lw=.8))
        for cx in (14, 20):
            ax.add_patch(Circle((cx, 115.8), .8, fc=INK))
        ax.plot([17, 21], [121, 130], color=INK, lw=1.4)
        label(ax, 17, 110, '任务观测', 7, ha='center', color=MUTED)
        arrow(ax, (28, 122), (36, 122))
        network(ax, 55, 121, scale=.95)
        ax.add_patch(Circle((73, 121), 6.2, ec=INH, fc='none', lw=.8))
        label(ax, 57, 106, '原网络', 7.5, ha='center')
        arrow(ax, (48, 103), (29, 92), color=TEAL)
        arrow(ax, (63, 103), (77, 92), color=LESION)
        for x, mode, color, title, subtitle in [
                (25, 'rewire', TEAL, '背景重接', '保持边数与入度、出度'),
                (74, 'delete', LESION, '指定连接删除', '按次序累计删除连接组')]:
            network(ax, x, 81, mode, scale=.86)
            label(ax, x+1, 67, title, 9, color, ha='center')
            label(ax, x+1, 60, subtitle, 7, MUTED, ha='center')

        ax.plot([79, 102], [125, 134], color=INH, lw=.75, ls=(0, (3, 3)))
        label(ax, 108, 133, '计算过程示意 · 非实验轨迹', 7, MUTED)
        for x, title in [(123, '汇聚输入'), (162, '膜电位'), (201, '发放与计数')]:
            label(ax, x, 125, title, 8, ha='center')

        for row, y, events, ein, iin in [
                (1, 108, [.18, .46, .68, .93], [0, 4, 9], [2, 7]),
                (2, 87, [.36, .79], [1, 8], [0, 4, 10])]:
            for yy, color, times, marker in [(y+4, EXC, ein, '+'), (y-3, INH, iin, '-')]:
                label(ax, 111, yy, marker, 10, color, ha='center')
                ax.plot([115, 127], [yy]*2, color=LINE, lw=.55)
                for t in times:
                    ax.plot([116+t]*2, [yy, yy+(2.7 if marker == '+' else -2)], color=color, lw=.85)
                arrow(ax, (128, yy), (134, y+.7), color=color)
            ax.add_patch(Circle((138, y+.7), 3, ec=INK, fc='#EEF3F5', lw=.85))
            label(ax, 138, y+.7, str(row), 7.5, ha='center')
            arrow(ax, (142, y+.7), (148, y+.7))
            wave(ax, 150, y-3, 28, events)
            arrow(ax, (180, y+.7), (185, y+.7))
            ax.plot([188, 214], [y-1]*2, color=LINE, lw=.6)
            for e in events:
                ax.plot([188+25*e]*2, [y-1, y+5], color=INK, lw=.9)
            label(ax, 220, y+1, f'$n_{row}$', 10, ha='center')
        label(ax, 165, 118, '阈值', 6.5, MUTED, ha='center')
        label(ax, 164, 76, '内部时间', 6.5, MUTED, ha='center')
        label(ax, 118, 74, '+ 兴奋', 6.5, EXC, ha='center')
        label(ax, 118, 69, '− 抑制', 6.5, INH, ha='center')
        ax.plot([226, 230, 230, 226], [111, 111, 87, 87], color=MUTED, lw=.75)
        ax.plot([230, 230, 190], [87, 77, 77], color=MUTED, lw=.75)
        arrow(ax, (190, 77), (190, 70))
        label(ax, 224, 65, '动作', 9, ha='center')
        arrow(ax, (203, 65), (215, 65))
        label(ax, 190, 65, '比较计数', 8.5, ha='center')
        label(ax, 190, 59, '20 个内部步 · 包含平局裁决', 6.5, MUTED, ha='center')

        # Empirical association: raw paired coordinates with no fitted curve.
        ax.plot([7, 233], [52, 52], color=LINE, lw=.6)
        header(ax, 7, 46, 'C', '重接与删除后的功能保持相关')
        p = fig.add_axes([24/W, 13/H, 70/W, 27/H])
        pts = p.scatter(xy[:, 0], xy[:, 1], s=19, color=TEAL,
                        edgecolors='white', linewidths=.35, zorder=3)
        assert np.array_equal(np.asarray(pts.get_offsets()), xy)
        p.set(xlim=(-.04, 1.04), ylim=(-15, 515), xticks=[0, .5, 1], yticks=[0, 250, 500])
        p.set_xticklabels(['0%', '50%', '100%'])
        p.tick_params(labelsize=6.5, length=2, color=MUTED)
        for spine in ('left', 'bottom'):
            p.spines[spine].set_color(MUTED)
            p.spines[spine].set_linewidth(.55)
        p.set_xlabel('背景重接后保留的回报比例', fontsize=7, labelpad=3)
        p.set_ylabel('指定删除后的回报', fontsize=7, labelpad=3)
        label(ax, 105, 35, 'CartPole · 25 个原网络', 8)
        label(ax, 105, 27, f'Spearman ρ = {rho:.3f}', 10, TEAL)
        label(ax, 105, 18, '每点为同一原网络的配对测量', 7, MUTED)
        label(ax, 105, 11, '实测关联', 7.5, TEAL)
        arrow(ax, (180, 53), (180, 37), color=MUTED, dash=True)
        label(ax, 184, 44, '待检验的解释联系', 7, MUTED)
        label(ax, 180, 29, '结构改变后的输出活动', 8)
        label(ax, 180, 22, '可能解释网络的耐受差异', 8)
        label(ax, 180, 11, '机制解释', 7.5, MUTED)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fw, fh = fig.canvas.get_width_height()
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_visible() and t.get_text():
                bb = t.get_window_extent(renderer)
                assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= fw+1 and bb.y1 <= fh+1, t.get_text()
        fig.savefig(OUT/'overview_integration_refined.png', dpi=240)
        fig.savefig(OUT/'overview_integration_refined.svg')
        plt.close(fig)
    assert source_hash == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps({
        'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': source_hash,
        'selection': "panel == 'a'; all rows; no modified coordinates", 'n': 25,
        'rho': rho, 'size_mm': [W, H], 'png_dpi': 240,
        'schematic_panels': ['A: conceptual networks', 'B: conceptual waveforms; not recorded or simulated'],
        'empirical_panel': 'C: paired rewire/deletion returns',
        'status': 'Refined Chinese preview; not inserted into manuscript',
        'software': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
