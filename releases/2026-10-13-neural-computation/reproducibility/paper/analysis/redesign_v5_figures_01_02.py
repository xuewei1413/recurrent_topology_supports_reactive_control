"""Redesign the v5 figures using the frozen source table and statistics."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
import fitz

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/figures/v5"
OUT = SOURCE / "redesign"
PDF = ROOT / "output/pdf/v5_figures_20260921_redesign"
INK = "#26333B"
GREY = "#64717B"
LINE = "#D6DEE1"
BLUE = "#216A9C"
GREEN = "#147D72"
ORANGE = "#B7522F"
EXPORTS = []
COUNTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(ax, x, y, s, size=8, **kwargs):
    return ax.text(x, y, s, fontsize=size, va="center", **kwargs)


def arrow(ax, p, q, color=GREY, curve=0, lw=.85, style="-|>", **kwargs):
    patch = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7,
                            linewidth=lw, color=color,
                            connectionstyle=f"arc3,rad={curve}", **kwargs)
    ax.add_patch(patch)
    return patch


def network(ax, cx, cy, mode="original", scale=1):
    # Identical coordinates and interface edges across illustrative variants.
    pos = np.array([[-19, -7], [-19, 7], [-9, -10], [-9, 0], [-9, 10],
                    [0, -6], [0, 6], [9, -10], [9, 0], [9, 10],
                    [19, -7], [19, 7]]) * scale + [cx, cy]
    interface = {(0, 2), (0, 3), (1, 4), (7, 10), (8, 10), (9, 11)}
    hh = {(2, 5), (3, 5), (4, 6), (5, 7), (5, 8), (6, 8), (6, 9),
          (7, 3), (8, 4), (9, 2)}
    old, new = {(3, 5), (6, 8)}, {(3, 8), (6, 5)}
    removed = {(2, 5), (5, 7), (8, 4), (6, 9)}
    kept = {(4, 6), (5, 8)}
    rw = hh - old | new
    for n in range(2, 10):
        assert sum(a == n for a, b in hh) == sum(a == n for a, b in rw)
        assert sum(b == n for a, b in hh) == sum(b == n for a, b in rw)
    assert kept <= rw
    edges = hh if mode != "rewire" else rw
    for a, b in sorted(interface | edges):
        deleted = mode == "delete" and (a, b) in removed
        color = ORANGE if deleted else (GREEN if mode == "rewire" and (a, b) in new else GREY)
        if (a, b) in kept and mode != "delete":
            color = INK
        p, q = pos[a], pos[b]
        curve = .30 if a > b else (-.15 if (a, b) in new else 0)
        if mode == "rewire" and (a, b) == (3, 8):
            curve = -.12
        patch = arrow(ax, p, q, color=color, curve=curve,
                      lw=1.05 if color in (GREEN, INK) else .65,
                      alpha=.6 if (a, b) in interface else 1,
                      shrinkA=3, shrinkB=3,
                      linestyle=(0, (2, 2)) if deleted else "solid")
        if deleted:
            mid = (p + q) / 2
            ax.plot([mid[0] - 1, mid[0] + 1], [mid[1] - 1.1, mid[1] + 1.1],
                    color=ORANGE, lw=1)
    for n, (x, y) in enumerate(pos):
        face = "white" if n in (0, 1, 10, 11) else "#EAF0F2"
        ax.add_patch(Circle((x, y), 1.15 * scale, fc=face, ec=INK, lw=.8, zorder=5))


def export(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    outside = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        box = t.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > w + 1 or box.y1 > h + 1:
            outside.append(t.get_text())
    assert not outside, outside
    for ext, folder in [("png", OUT), ("svg", OUT), ("pdf", PDF)]:
        path = folder / f"{name}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None)
        EXPORTS.append({"file": str(path.relative_to(ROOT)), "sha256": sha(path)})
    plt.close(fig)


def overview():
    fig = plt.figure(figsize=(180 / 25.4, 123 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 180), ylim=(0, 123))
    ax.set_aspect("equal")
    ax.axis("off")
    label(ax, 5, 117, "a", 11, weight="bold")
    label(ax, 12, 117, "固定突触幅值，进化连接结构", 10, weight="bold")

    # A compact closed-loop controller, without invented activity traces.
    ax.plot([8, 25], [98, 98], color=GREY, lw=.8)
    ax.add_patch(Rectangle((12, 100), 10, 4, fc="#EAF0F2", ec=INK, lw=.9))
    for x in (14, 20):
        ax.add_patch(Circle((x, 99), .8, color=INK))
    ax.plot([17, 22], [104, 112], color=BLUE, lw=1.8)
    ax.add_patch(Circle((17, 104), .65, color=INK))
    label(ax, 17, 94, "控制环境", 7.5, ha="center")
    arrow(ax, (29, 103), (43, 103))
    label(ax, 51, 103, "观测编码", 8, ha="center")
    arrow(ax, (62, 103), (70, 103))
    network(ax, 85, 103, scale=.55)
    label(ax, 85, 94, "脉冲网络", 7.5, ha="center")
    arrow(ax, (99, 103), (109, 103))
    label(ax, 123, 105, "输出脉冲计数", 8, ha="center")
    label(ax, 123, 99, r"$n_1,\;n_2,\;\ldots$", 8.5, ha="center", color=GREY)
    arrow(ax, (140, 103), (151, 103))
    label(ax, 161, 103, "控制动作", 8, ha="center")
    ax.plot([164, 164, 17], [98, 88, 88], color=LINE, lw=.85)
    arrow(ax, (17, 88), (17, 92), color=GREY)
    label(ax, 90, 88, "动作改变后续观测", 6.8, color=GREY, ha="center",
          bbox=dict(fc="white", ec="none", pad=2))
    ax.plot([5, 175], [83, 83], color=LINE, lw=.6)

    label(ax, 5, 77, "b", 11, weight="bold")
    label(ax, 12, 77, "同一原网络，分别构造两类变体", 10, weight="bold")
    network(ax, 29, 45, scale=1.0)
    label(ax, 29, 29, "原网络", 8.5, ha="center", weight="bold")
    label(ax, 29, 23, r"基准回报 $J_0$", 7.5, ha="center", color=GREY)
    arrow(ax, (52, 49), (67, 64), color=GREEN, curve=-.08, lw=1.1)
    arrow(ax, (52, 41), (67, 28), color=ORANGE, curve=.08, lw=1.1)

    network(ax, 91, 61, "rewire", scale=.85)
    label(ax, 91, 73, "重接其余隐藏连接", 8.2, weight="bold", ha="center", color=GREEN)
    label(ax, 91, 48, "保留指定边组 · 保持度数", 7, ha="center", color=GREY)
    arrow(ax, (111, 61), (126, 61), color=GREEN)
    label(ax, 150, 68, "重接后功能保持", 8.5, ha="center", color=GREEN, weight="bold")
    label(ax, 150, 60, r"$B_i = \langle J_{\mathrm{rewire}}/J_0\rangle$", 11, ha="center")
    label(ax, 150, 51, "13 个变体的平均值", 7, ha="center", color=GREY)

    network(ax, 91, 27, "delete", scale=.85)
    label(ax, 91, 40, "依次删除连接组", 8.2, weight="bold", ha="center", color=ORANGE)
    label(ax, 91, 14, "顺序由原网络输出响应确定", 7, ha="center", color=GREY)
    arrow(ax, (111, 27), (126, 27), color=ORANGE)
    label(ax, 150, 35, "删除后控制表现", 8.5, ha="center", color=ORANGE, weight="bold")
    label(ax, 150, 27, r"$J_{24,i}$", 13, ha="center")
    label(ax, 150, 19, "删除 24 个编号组后的回报", 7, ha="center", color=GREY)

    ax.plot([5, 175], [9, 9], color=LINE, lw=.6)
    label(ax, 6, 4.5, "跨网络比较", 8, weight="bold")
    label(ax, 42, 4.5, "重接后的功能保持，能否预测删除后的控制表现？", 8)
    export(fig, "figure1_overview")


def panel(fig, x, y, letter, title, subtitle=None):
    fig.text(x, y, letter, size=11, weight="bold", va="baseline")
    fig.text(x + .026, y, title, size=9, weight="bold", va="baseline")
    if subtitle:
        fig.text(x + .026, y - .029, subtitle, size=7, color=GREY, va="baseline")


def scatter(fig, rect, rows, result, key, color, ylabel=False, xlabel=True, large=False):
    ax = fig.add_axes(rect)
    assert all(-.035 <= float(r["background_idx"]) <= 1.035 and
               -15 <= float(r["J24"]) <= 515 for r in rows)
    for stage, marker, filled in [("main", "o", True), ("planned", "o", True),
                                   ("post_result_added", "D", False)]:
        selected = [r for r in rows if r["stage"] == stage]
        if not selected:
            continue
        ax.scatter([float(r["background_idx"]) for r in selected],
                   [float(r["J24"]) for r in selected],
                   s=26 if large else 16, marker=marker,
                   facecolor=color if filled else "white", edgecolor=color,
                   lw=.85, alpha=.86 if filled else 1, zorder=3)
    COUNTS[key] = sum(len(c.get_offsets()) for c in ax.collections)
    assert COUNTS[key] == len(rows)
    ax.set(xlim=(-.035, 1.035), ylim=(-15, 515),
           xticks=[0, .5, 1], yticks=[0, 250, 500])
    ax.set_xticklabels(["0", "0.5", "1.0"])
    ax.tick_params(length=2.5, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7ECEF", lw=.5)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    if ylabel:
        ax.set_ylabel(r"删除后回报 $J_{24}$", labelpad=5)
    if xlabel:
        ax.set_xlabel(r"重接后功能保持 $B$", labelpad=4)
    low, high = result["ci95"]
    ax.text(.04, .96, rf"$\rho={result['rho']:.3f}$", transform=ax.transAxes,
            va="top", size=10 if large else 8.5, weight="bold", color=INK)
    # Numeric intervals are below the plot titles rather than over the observations.
    return ax, f"95% CI {low:.2f}–{high:.2f}"


def evidence(rows, report):
    fig = plt.figure(figsize=(180 / 25.4, 157 / 25.4))
    groups = {p: [r for r in rows if r["panel"] == p] for p in "abcd"}
    panel(fig, .055, .958, "a", "本地复跑主队列", "25 个网络")
    panel(fig, .555, .958, "b", "本地复跑后续队列", "历史原计划 15 个 + 追加 15 个种子")
    scatter(fig, [.09, .568, .375, .31], groups["a"], report["scatter"]["main"], "a", BLUE,
            ylabel=True, large=True)
    scatter(fig, [.59, .568, .375, .31], groups["b"], report["scatter"]["extended"], "b", GREEN,
            large=True)
    # Separate the stage legend from both the data and the axis labels.
    fig.text(.59, .90, "● 原计划范围", color=GREEN, size=7)
    fig.text(.76, .90, "◇ 追加范围", color=GREEN, size=7)
    fig.text(.09, .90, "每点为一个网络", color=GREY, size=7)
    fig.add_artist(plt.Line2D([.055, .965], [.477, .477], color=LINE, lw=.6,
                             transform=fig.transFigure))
    panel(fig, .055, .433, "c", "新评估段复测", "沿用 b 的 30 个网络")
    panel(fig, .375, .433, "d", "跨评估段配对", "原评估段 B / 新评估段回报")
    panel(fig, .70, .433, "e", "协变量调整", "点估计与 95% CI")
    scatter(fig, [.09, .145, .225, .213], groups["c"], report["scatter"]["repeat"], "c", GREEN,
            ylabel=True)
    scatter(fig, [.41, .145, .225, .213], groups["d"], report["scatter"]["cross"], "d", GREEN)
    ax = fig.add_axes([.795, .145, .17, .213])
    ax.axvline(0, color=GREY, lw=.7, ls=(0, (3, 3)))
    names = ["未调整", "控剂量", "完整调整"]
    for cohort, offset, color, marker in [("v3", 3.6, BLUE, "o"), ("c2", 0, GREEN, "s")]:
        for j, (key, name) in enumerate(zip(["raw", "dose", "all"], names)):
            item = report["adjustment"][cohort][key]
            y = offset + 2 - j
            lo, hi = item["ci95"]
            ax.plot([lo, hi], [y, y], color=color, lw=1.3, solid_capstyle="round")
            ax.plot(item["rho"], y, marker, color=color, ms=3.5, mec="white", mew=.4)
    ax.set(xlim=(-.22, 1), ylim=(-.55, 6.3), xticks=[0, .5, 1],
           yticks=[5.6, 4.6, 3.6, 2, 1, 0], yticklabels=names + names)
    ax.tick_params(axis="y", length=0, pad=4, labelsize=6.5)
    ax.tick_params(axis="x", length=2.5)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.set_xlabel(r"相关系数 $\rho$", labelpad=4)
    ax.text(-.21, 6.18, "主队列", color=BLUE, size=6.5)
    ax.text(-.21, 2.63, "另一队列", color=GREEN, size=6.5)
    fig.text(.055, .048, "复测不增加独立网络数；完整调整包含基准回报、输出敏感性与实际删边比例。",
             size=6.8, color=GREY)
    export(fig, "figure2_cross_perturbation")


def validate():
    result = []
    for name, height in [("figure1_overview", 123), ("figure2_cross_perturbation", 157)]:
        doc = fitz.open(PDF / f"{name}.pdf")
        page = doc[0]
        assert len(doc) == 1
        assert abs(page.rect.width * 25.4 / 72 - 180) < .01
        assert abs(page.rect.height * 25.4 / 72 - height) < .01
        assert not page.get_images()
        for font in page.get_fonts(full=True):
            assert font[2] != "Type3"
            assert doc.extract_font(font[0])[3]
        for word in page.get_text("words"):
            assert page.rect.contains(fitz.Rect(word[:4])), word
        render = ROOT / "tmp/pdfs/v5-redesign" / f"{name}.png"
        render.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(render)
        result.append({"figure": name, "mm": [180, height], "vector_only": True,
                       "fonts_embedded": True, "text_within_page": True})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / "figure2_source_data.csv", SOURCE / "figure2_statistics.json"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    rows = list(csv.DictReader(paths[0].open(encoding="utf-8-sig", newline="")))
    report = json.loads(paths[1].read_text(encoding="utf-8"))
    assert len(rows) == 115 and len({r["seed"] for r in rows}) == 55
    with mpl.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": GREY, "ytick.color": GREY, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "v5-redesign-20260921", "axes.unicode_minus": False,
    }):
        overview()
        evidence(rows, report)
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    audit = {"source_sha256": before, "statistics_recomputed": False,
             "source_data_unchanged": True, "scatter_points": COUNTS,
             "unique_networks": 55, "pdf_checks": validate(), "exports": EXPORTS,
             "journal_format_status": "provisional; not a journal-compliance certification"}
    (OUT / "figure_qa.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"scatter_points": COUNTS, "outputs": len(EXPORTS),
                      "pdf_checks": audit["pdf_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
