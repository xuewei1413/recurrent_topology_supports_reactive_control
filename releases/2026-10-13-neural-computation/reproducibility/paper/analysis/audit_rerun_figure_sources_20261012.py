"""Record current figure provenance while retaining unchanged historical results."""

import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper/figures/v5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = FIG / "historical_20261009/chinese_final"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ["manifest.json", "qa.json"]:
        if not (archive / name).exists():
            shutil.copy2(FIG / "chinese_final" / name, archive / name)
    previous = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    previous_qa = json.loads((archive / "qa.json").read_text(encoding="utf-8"))
    historical = {}
    for name, expected in previous["source_sha256"].items():
        name = name.replace("\\", "/")
        if name in {"_phase8/audit_v9_layer1_route.json", "e5_results_0803.tgz",
                    "e2rv2_results_0803.tgz", "artifact_e1/results_v7_v8_data.tar.gz"}:
            assert sha(ROOT / name) == expected, name
            historical[name] = expected
    current = {}
    for name in ["paper/manuscript_v5_zh_draft.md", "paper/manuscript_v5_en_draft.md",
                 "paper/analysis/2026-10-12_local_comparison.json",
                 "paper/analysis/2026-10-12_current_figure.json",
                 "paper/analysis/2026-10-12_current_lineages.csv",
                 "paper/figures/v5/figure2_source_data.csv", "paper/figures/v5/figure2_statistics.json"]:
        current[name] = sha(ROOT / name)
    media = {}
    for svg in sorted((FIG / "chinese_final").glob("*.svg")):
        ET.parse(svg)
        png = svg.with_suffix(".png")
        with Image.open(png) as bitmap:
            bitmap.load()
            size = list(bitmap.size)
        item = {"svg_sha256": sha(svg), "png_sha256": sha(png), "png_pixels": size}
        if svg.stem in {"00_overview", "01_cross_perturbation"}:
            item["source"] = "Local execution completed 2026-10-12; new matched graphs and measurements"
        else:
            old = previous_qa["media"][svg.stem]
            assert item["svg_sha256"] == old["svg_sha256"]
            assert item["png_sha256"] == old["png_sha256"]
            item["source"] = "Unchanged historical experiment; not rerun in the local execution"
        media[svg.stem] = item
    with (FIG / "figure2_source_data.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = {panel: sum(row["panel"] == panel for row in rows) for panel in "abcd"}
    assert counts == {"a": 25, "b": 30, "c": 30, "d": 30}
    assert len({row["seed"] for row in rows}) == 55
    estimates = json.loads((FIG / "figure2_statistics.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "paper/analysis/2026-10-12_local_comparison.json").read_text(encoding="utf-8"))
    for panel, cohort in [("main", "v3"), ("extended", "c2"), ("repeat", "final"), ("cross", "cross")]:
        assert estimates["scatter"][panel]["rho"] == audit["new"][cohort]["rho"]
    manifest = {"date": "2026-10-12", "status": "Current local-rerun figures and unchanged historical figures separated",
                "current_source_sha256": current, "historical_source_sha256": historical,
                "historical_input_members": previous["input_members"], "media": media,
                "scatter_statistics": estimates["scatter"],
                "historical_results_figures_3_to_6": previous["results"],
                "overview": "Network layouts and waveforms are schematics; correlation panel uses new measured data"}
    qa = {"date": "2026-10-12", "status": "passed", "figure_count": len(media),
          "cross_perturbation_panel_rows": counts, "unique_cross_perturbation_topologies": 55,
          "checks": ["SVG parsed", "PNG decoded", "Unchanged historical artwork byte-identical",
                     "Source hashes recorded and historical sources verified", "Scatter estimates match new analysis"],
          "manifest_sha256": None, "media": media}
    path = FIG / "chinese_final/manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    qa["manifest_sha256"] = sha(path)
    (FIG / "chinese_final/qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "figures": len(media), "panels": counts}))


if __name__ == "__main__":
    main()
