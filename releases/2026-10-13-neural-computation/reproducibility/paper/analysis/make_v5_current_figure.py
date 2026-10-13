"""Plot local current-allocation data without changing statistical estimates."""

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/v5/supplement"
COLORS = {"self_exact": "#157A80", "energy_random": "#C16545"}


def make(lang):
    rows = list(csv.DictReader((ROOT / "paper/analysis/2026-10-12_current_lineages.csv").open(encoding="utf-8")))
    audit = json.loads((ROOT / "paper/analysis/2026-10-12_current_figure.json").read_text(encoding="utf-8"))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="880" viewBox="0 0 1200 880">',
           '<rect width="1200" height="880" fill="white"/>',
           '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#233239;font-size:17px}</style>']

    def line(x1, y1, x2, y2, color="#BCC6CB", width=1, dash="", opacity=1):
        svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}" opacity="{opacity}"/>')

    def text(x, y, value, anchor="start", size=17, rotate=None):
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px"{transform}>{html.escape(str(value))}</text>')

    labels = {"en": ["Exact correction", "Random direction", "No correction", "Mean return", "Exact minus control", "Random", "Signed action", "Single channel", "Shift by 4"],
              "zh": ["精确校正", "随机方向", "无校正", "平均回报", "精确校正减对照", "随机方向", "有符号动作", "单通道", "环移 4 步"]}[lang]
    for col, task in enumerate(["CartPole", "Acrobot"]):
        x0, x1 = 90 + 590 * col, 565 + 590 * col
        data = [r for r in rows if r["task"] == task]
        seeds = sorted({r["seed"] for r in data})
        ymin, ymax = (0, 520) if col == 0 else (-260, -60)
        y0, y1 = 95, 355
        y = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)
        x = lambda a: x0 + (a - .25) / .75 * (x1 - x0)
        text(x0 - 38, 49, "AB"[col], size=26)
        text(x0, 49, task, size=23)
        ticks = [0, 100, 200, 300, 400, 500] if col == 0 else [-260, -220, -180, -140, -100, -60]
        for tick in ticks:
            line(x0, y(tick), x1, y(tick), "#E4E9EB")
            text(x0 - 10, y(tick) + 6, tick, "end", 15)
        line(x0, y0, x0, y1)
        line(x0, y1, x1, y1)
        for alpha in [.25, .5, .75, 1.]:
            text(x(alpha), y1 + 25, f"{alpha:.2f}", "middle", 15)
        text((x0 + x1) / 2, y1 + 53, "α", "middle", 19)
        text(x0 - 60, (y0 + y1) / 2, labels[3], "middle", 17, -90)
        for key in ["energy_random", "self_exact"]:
            for seed in seeds:
                subset = sorted([r for r in data if r["seed"] == seed], key=lambda r: float(r["alpha"]))
                for a, b in zip(subset, subset[1:]):
                    line(x(float(a["alpha"])), y(float(a[key])), x(float(b["alpha"])), y(float(b[key])), COLORS[key], 1.2, "4 3" if key == "energy_random" else "", .28)
            points = audit["tasks"][task]["dose_descriptive"]
            for a, b in zip(points, points[1:]):
                field = "exact_mean" if key == "self_exact" else "random_mean"
                line(x(a["alpha"]), y(a[field]), x(b["alpha"]), y(b[field]), COLORS[key], 3.5, "7 4" if key == "energy_random" else "")
            for r in points:
                field = "exact_mean" if key == "self_exact" else "random_mean"
                svg.append(f'<circle cx="{x(r["alpha"]):.2f}" cy="{y(r[field]):.2f}" r="4.5" fill="{COLORS[key]}"/>')
        line(x0, y(audit["tasks"][task]["none_mean"]), x1, y(audit["tasks"][task]["none_mean"]), "#68767D", 1.5, "3 5")
        for i, color in enumerate([COLORS["self_exact"], COLORS["energy_random"], "#68767D"]):
            lx = x0 + i * 162
            line(lx, 76, lx + 22, 76, color, 2.7, "5 3" if i == 1 else "3 5" if i == 2 else "")
            text(lx + 28, 81, labels[i], size=12 if lang == "en" else 15)

        y0, y1 = 500, 765
        ymin, ymax = (-55, 515) if col == 0 else (-22, 125)
        y = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)
        text(x0 - 38, 467, "CD"[col], size=26)
        text(x0, 467, task + " · α = 1", size=23)
        ticks = [0, 100, 200, 300, 400, 500] if col == 0 else [-20, 0, 40, 80, 120]
        for tick in ticks:
            line(x0, y(tick), x1, y(tick), "#E4E9EB")
            text(x0 - 10, y(tick) + 6, tick, "end", 15)
        line(x0, y0, x0, y1)
        line(x0, y1, x1, y1)
        line(x0, y(0), x1, y(0), "#748188", 1.3, "3 4")
        sesoi = 30 if col == 0 else 10
        line(x0, y(sesoi), x1, y(sesoi), "#896A28", 1.3, "7 4")
        text(x1, y(sesoi) - 7, f"SESOI = {sesoi}", "end", 13)
        text(x0 - 60, (y0 + y1) / 2, labels[4], "middle", 17, -90)
        for i, key in enumerate(["energy_random", "action_centered", "action_only", "time_shuffle"]):
            center = x0 + 56 + i * 118
            selected = sorted([r for r in data if float(r["alpha"]) == 1], key=lambda r: int(r["seed"]))
            for j, r in enumerate(selected):
                value = float(r["self_exact"]) - float(r[key])
                px = center + ((j * 7) % 12 - 5.5) * 2.5
                svg.append(f'<circle cx="{px:.2f}" cy="{y(value):.2f}" r="4" fill="#6F8792" opacity=".7"/>')
            mean = audit["tasks"][task]["contrasts_alpha1"][key]["mean"]
            ci = audit["tasks"][task]["contrasts_alpha1"][key]["ci"]
            px = center + 27
            line(px, y(ci[0]), px, y(ci[1]), "#157A80", 2.5)
            for bound in ci:
                line(px - 5, y(bound), px + 5, y(bound), "#157A80", 2)
            py = y(mean)
            svg.append(f'<polygon points="{px},{py-6} {px+6},{py} {px},{py+6} {px-6},{py}" fill="#157A80"/>')
            text(center, y1 + 28, labels[5 + i], "middle", 13 if lang == "en" else 15)
        text(x0, 838, "12 lineages; points = lineages; diamonds = means" if lang == "en" else "12 个进化网络；圆点为网络，菱形为均值", size=15)
    svg.append("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"figure_s1_current_{lang}.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    print(json.dumps([make("zh"), make("en")], ensure_ascii=False))
