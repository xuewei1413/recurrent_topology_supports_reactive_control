"""Use the new matched execution in the approved figure layouts."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_local_replication_20261009"
FIG = ROOT / "paper/figures/v5"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    audit = json.loads((ROOT / "paper/analysis/2026-10-12_local_comparison.json").read_text())
    archive = FIG / "historical_20261009"
    for path in [FIG / "figure2_source_data.csv", FIG / "figure2_statistics.json", *[
        FIG / folder / f"{stem}.{ext}" for folder in ["chinese_final","english_final"]
        for stem in ["00_overview","01_cross_perturbation"] for ext in ["svg","png"]], *[
        FIG / "supplement" / f"figure_s1_current_{lang}.{ext}" for lang in ["zh","en"] for ext in ["svg","png"]]]:
        destination = archive / path.relative_to(FIG)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path,destination)
    from local_replication.analyze_association import load_rows
    v3, _ = load_rows("v3","full")
    c2, _ = load_rows("c2","full")
    final, _ = load_rows("c2","full",final=True)
    cross = [{**a,"J24":b["J24"]} for a,b in zip(c2,final)]
    with (FIG / "figure2_source_data.csv").open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=["panel","stage","seed","background_idx","J24"])
        writer.writeheader()
        for panel,rows in zip("abcd",[v3,c2,final,cross]):
            for r in rows:
                writer.writerow({"panel":panel,"stage":"main" if panel=="a" else
                    "planned" if r["seed"]<=4274 else "post_result_added",
                    **{k:r[k] for k in ["seed","background_idx","J24"]}})
    report = {"analysis_status":"local_rerun_20261012", "scatter":{}, "adjustment":{},
              "provenance":"2026-10-12_local_comparison.json; newly matched matrices and measurements"}
    for key,name in [("main","v3"),("extended","c2"),("repeat","final"),("cross","cross")]:
        r = audit["new"][name]
        ci = r.get("ci95") or r["bootstrap_95ci"]["interval"]
        report["scatter"][key] = {"rho":r["rho"],"ci95":ci}
    for c, key in [("v3","main"),("c2","extended")]:
        report["adjustment"][c] = {"raw":report["scatter"][key],
            **audit["new"][c]["adjustment_intervals"]}
    dump(FIG / "figure2_statistics.json",report)
    import matplotlib as mpl
    import redesign_v5_figures_01_02 as draw
    draw.OUT.mkdir(parents=True,exist_ok=True)
    draw.PDF.mkdir(parents=True,exist_ok=True)
    with (FIG / "figure2_source_data.csv").open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    with mpl.rc_context({"font.family":"sans-serif", "font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
        "font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,
        "text.color":draw.INK,"axes.labelcolor":draw.INK,"axes.spines.top":False,
        "axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42,
        "axes.unicode_minus":False}):
        draw.evidence(rows,report)
    import make_overview_integration_refined as overview
    overview.main()
    for stem,source in [("00_overview",FIG/"overview_integration_refined/overview_integration_refined"),
                        ("01_cross_perturbation",FIG/"redesign/figure2_cross_perturbation")]:
        for ext in ["svg","png"]:
            shutil.copy2(source.with_suffix("."+ext),FIG / "chinese_final" / f"{stem}.{ext}")
    current = {"tasks":{}}
    lineage_rows=[]
    for task in ["CartPole","Acrobot"]:
        result = audit["current"][task]["new"]
        means={(r["mode"],r["alpha"]):r["mean"] for r in result["mean_returns"]}
        current["tasks"][task] = {"none_mean":means[("none",1.0)],"dose_descriptive":[
            {"alpha":a,"exact_mean":means[("self_exact",a)],"random_mean":means[("energy_random",a)]}
            for a in [.25,.5,.75,1.0]], "contrasts_alpha1":{k:{"mean":r["mean_difference"],
                "ci":r["bootstrap_95ci"]} for k,r in result["contrasts"].items()}}
        grouped={}
        for path in (OUT / "current/full" / task).glob("*.json"):
            p=json.loads(path.read_text())
            assert p["complete"]
            grouped.setdefault((p["seed"],p["alpha"]),{}).setdefault(p["mode"],[]).append(
                np.mean([r["return"] for r in p["rows"]]))
        for (seed,alpha),conditions in sorted(grouped.items()):
            row={"task":task,"seed":seed,"alpha":alpha}
            for mode,values in conditions.items():
                assert len(values)==6
                row[mode]=float(np.mean(values))
            lineage_rows.append(row)
    dump(ROOT / "paper/analysis/2026-10-12_current_figure.json",current)
    fields=["task","seed","alpha","self_exact","energy_random","none","action_centered","action_only","time_shuffle"]
    with (ROOT / "paper/analysis/2026-10-12_current_lineages.csv").open("w",encoding="utf-8",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=fields)
        writer.writeheader()
        writer.writerows(lineage_rows)
    import make_v5_current_figure as current_draw
    current_draw.make("zh")
    current_draw.make("en")
    print(json.dumps({"panels":draw.COUNTS,"current_lineage_rows":len(lineage_rows)}))


if __name__ == "__main__":
    main()
