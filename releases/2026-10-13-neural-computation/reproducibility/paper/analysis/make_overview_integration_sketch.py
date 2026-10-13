"""Original vector sketch: interventions, output computation, observed association."""

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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/figures/v5/overview_integration_sketch'
SOURCE = ROOT / 'paper/figures/v5/figure2_source_data.csv'
W, H = 240, 167
INK, GREY, LIGHT = '#253039', '#63727B', '#CBD2D6'
TEAL, RED, BLUE = '#197F78', '#B45C47', '#4C78A8'


def text(ax, x, y, value, size=9, color=INK, ha='left', **kw):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha,
                   va='center', **kw)


def arrow(ax, start, end, color=GREY, curve=0, style='solid', **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                 mutation_scale=8, lw=.9, color=color, linestyle=style,
                 connectionstyle=f'arc3,rad={curve}', **kw))


def network(ax, x, y, mode='original', scale=1):
    base = np.array([[-15, 4], [-15, -4], [-7, 8], [-6, -5],
                     [1, 3], [2, -8], [9, 7], [10, -3], [19, 4], [19, -4]])
    nodes = base * scale + [x, y]
    edges = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
             (6, 8), (7, 9), (4, 7), (5, 3)]
    rewired = {(2, 4), (3, 5)}
    deleted = {(2, 4), (5, 7)}
    for a, b in edges:
        if (a, b) in rewired and mode == 'rewire':
            continue
        p, q = nodes[a], nodes[b]
        is_deleted = mode == 'delete' and (a, b) in deleted
        arrow(ax, p, q, color=RED if is_deleted else LIGHT,
              style=(0, (2, 2)) if is_deleted else 'solid',
              shrinkA=2.2, shrinkB=2.2, curve=.55 if (a, b) == (5, 3) else 0)
    if mode == 'rewire':
        for a, b in [(2, 5), (3, 4)]:
            arrow(ax, nodes[a], nodes[b], color=TEAL, shrinkA=2.2, shrinkB=2.2)
    if mode == 'delete':
        for a, b in sorted(deleted):
            mark = (nodes[a] + nodes[b]) / 2
            ax.add_patch(Circle(mark, 1.8 * scale, fc='white', ec='none', zorder=6))
            offset = 1.05 * scale
            for sign in (-1, 1):
                ax.plot([mark[0] - offset, mark[0] + offset],
                        [mark[1] - sign * offset, mark[1] + sign * offset],
                        color=RED, lw=1.3, zorder=7)
    for i, p in enumerate(nodes):
        ax.add_patch(Circle(p, 1.55 * scale, ec=INK if i >= 8 else GREY,
                           fc='#E5EFF5' if i >= 8 else 'white', lw=.8, zorder=5))


def membrane(ax, left, base, width, events):
    # Pedagogical threshold/reset drawing, not a simulated or recorded trajectory.
    threshold = 9
    ax.plot([left, left + width], [base + threshold] * 2,
            color=LIGHT, lw=.7, ls=(0, (3, 2)))
    xs, ys = [left], [base]
    previous = 0
    for event in events:
        x0 = left + previous * width
        x1 = left + event * width
        xs.extend([x0 + (x1-x0)*.33, x0 + (x1-x0)*.6, x1, x1])
        ys.extend([base + 3.8, base + 2.8, base + threshold, base])
        previous = event
    xs.extend([left + width])
    ys.extend([base + 2.2])
    ax.plot(xs, ys, color=INK, lw=1.05)
    ax.plot([left, left + width], [base-1.5]*2, color=LIGHT, lw=.6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['panel'] == 'a']
    xy = np.array([[float(r['background_idx']), float(r['J24'])] for r in rows])
    rho = float(spearmanr(xy[:, 0], xy[:, 1]).statistic)
    assert len(rows) == 25 and len({r['seed'] for r in rows}) == 25
    assert abs(rho - .6330769230769231) < 1e-12
    with plt.rc_context({'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
                         'svg.fonttype': 'path', 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, H))
        ax.set_aspect('equal')
        ax.axis('off')
        text(ax, 7, 160, 'A', 13, fontweight='bold')
        text(ax, 15, 160, '同一网络，两类结构改变', 11)
        text(ax, 104, 160, 'B', 13, fontweight='bold')
        text(ax, 112, 160, '输出端如何形成控制动作', 11)

        # A: one parent, two separate perturbations. CartPole is an illustrative task.
        ax.plot([8, 25], [130, 130], color=LIGHT, lw=1)
        ax.add_patch(Rectangle((12, 132), 10, 4, fc='white', ec=INK, lw=1))
        for x in [14, 20]:
            ax.add_patch(Circle((x, 131), .8, fc=INK))
        ax.plot([17, 21], [136, 146], color=INK, lw=1.5)
        text(ax, 17, 125, '任务观测', 8, ha='center')
        arrow(ax, (28, 136), (35, 136))
        network(ax, 55, 137, scale=.95)
        text(ax, 57, 151, '固定突触幅值的进化网络', 8, ha='center')
        text(ax, 57, 123, '只改变连接组织', 8, ha='center', color=GREY)
        arrow(ax, (49, 120), (27, 108), color=TEAL)
        arrow(ax, (62, 120), (76, 108), color=RED)
        network(ax, 24, 96, 'rewire', scale=.85)
        network(ax, 73, 96, 'delete', scale=.85)
        text(ax, 25, 81, '背景重接', 10, TEAL, ha='center')
        text(ax, 74, 81, '指定连接删除', 10, RED, ha='center')
        text(ax, 25, 73, '换接法，保持边数和度', 7.5, GREY, ha='center')
        text(ax, 74, 73, '依次移除连接组', 7.5, GREY, ha='center')

        # Magnification from the output neurons to the output computation panel.
        ax.add_patch(Circle((73, 137), 7, fc='none', ec=BLUE, lw=.9))
        ax.plot([80, 101], [141, 147], color=BLUE, lw=.8, ls=(0, (3, 3)))
        text(ax, 106, 149, '计算过程示意，曲线非实验记录', 7.5, GREY)
        for x, label in [(118, '到达的突触输入'), (155, '膜电位积分与复位'),
                         (199, '输出脉冲')]:
            text(ax, x, 140, label, 8, ha='center')
        for row, base, events in [(1, 117, [.22, .48, .72, .93]),
                                   (2, 94, [.35, .8])]:
            text(ax, 107, base+5, 'E', 7.5, RED)
            text(ax, 107, base-2, 'I', 7.5, BLUE)
            for i, yy, color in [(0, base+5, RED), (1, base-2, BLUE)]:
                ax.plot([112, 125], [yy, yy], color=LIGHT, lw=.65)
                event_positions = ([113, 117, 122] if i == 0 else [115, 120]) if row == 1 else ([114, 121] if i == 0 else [113, 117, 123])
                for pos in event_positions:
                    ax.plot([pos, pos], [yy, yy+(3 if i == 0 else -2)], color=color, lw=.85)
                arrow(ax, (126, yy), (133, base+1.5), color=color)
            ax.add_patch(Circle((137, base+1.5), 3.4, fc='white', ec=INK, lw=.85))
            text(ax, 137, base+1.5, str(row), 8, ha='center')
            arrow(ax, (141, base+1.5), (146, base+1.5))
            membrane(ax, 148, base-3, 25, events)
            arrow(ax, (175, base+1.5), (181, base+1.5))
            ax.plot([183, 210], [base-1]*2, color=LIGHT, lw=.7)
            for e in events:
                ax.plot([183+25*e]*2, [base-1, base+5], color=INK, lw=1)
            text(ax, 217, base+1, f'n{row}', 9)
        text(ax, 170, 86, '内部时间', 7, GREY, ha='center')
        text(ax, 159, 131, '虚线：发放阈值', 6.5, GREY, ha='center')
        ax.plot([228, 232, 232, 228], [120, 120, 97, 97], color=GREY, lw=.85)
        arrow(ax, (224, 91), (224, 81))
        text(ax, 180, 76, '计数比较 + 平局规则 → 动作', 8.3, ha='center')
        text(ax, 180, 68, '每个环境步统计 20 个内部步的输出发放', 7.1, GREY, ha='center')

        ax.plot([7, 233], [61, 61], color=LIGHT, lw=.7)
        text(ax, 7, 55, 'C', 13, fontweight='bold')
        text(ax, 15, 55, '两类结构改变后的功能保持相关', 11)
        p = fig.add_axes([24/W, 14/H, 70/W, 33/H])
        p.scatter(xy[:, 0], xy[:, 1], s=22, color=TEAL, edgecolors='white', linewidths=.4)
        p.set(xlim=(-.04, 1.04), ylim=(-15, 515), xticks=[0, .5, 1], yticks=[0, 250, 500])
        p.set_xticklabels(['0%', '50%', '100%'])
        p.tick_params(labelsize=7, length=2)
        p.spines['left'].set_color(LIGHT)
        p.spines['bottom'].set_color(LIGHT)
        p.set_xlabel('背景重接后保留的回报比例', fontsize=8)
        p.set_ylabel('指定删除后的回报', fontsize=8)
        text(ax, 104, 44, 'CartPole 主队列 · 25 个网络', 9)
        text(ax, 104, 35, f'Spearman ρ = {rho:.3f}', 10, TEAL)
        text(ax, 104, 27, '每点对应同一个原网络的两项测量', 7.5, GREY)
        text(ax, 104, 14, '实测关系', 8.5, TEAL)
        arrow(ax, (179, 62), (179, 45), color=GREY, style=(0, (3, 3)))
        text(ax, 184, 51, '解释联系待检验', 7.5, GREY)
        text(ax, 179, 37, '输出端活动如何改变，', 8, ha='left')
        text(ax, 179, 29, '可能解释耐受差异', 8, ha='left')
        text(ax, 179, 14, '机制解释', 8.5, GREY)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        for artist in fig.findobj(matplotlib.text.Text):
            if artist.get_visible() and artist.get_text():
                bounds = artist.get_window_extent(renderer)
                assert bounds.x0 >= 0 and bounds.y0 >= 0
                assert bounds.x1 <= width+1 and bounds.y1 <= height+1, artist.get_text()
        fig.savefig(OUT/'overview_integration_sketch.png', dpi=190)
        fig.savefig(OUT/'overview_integration_sketch.svg')
        plt.close(fig)
    manifest = {'source': str(SOURCE.relative_to(ROOT)),
                'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'selection': "panel == 'a'; all 25 rows, unmodified coordinates",
                'n': len(rows), 'spearman_rho': rho,
                'schematic_only': ['networks', 'E/I inputs', 'membrane traces', 'spike trains'],
                'status': 'Discussion sketch; no causal mediation claim; no manuscript replaced'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
