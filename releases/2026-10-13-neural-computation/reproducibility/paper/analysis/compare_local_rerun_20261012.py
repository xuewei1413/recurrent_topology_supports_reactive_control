"""Version-separated reporting audit for the completed local rerun."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from local_replication.analyze_association import load_rows, summarize, json_finite
from local_replication.run_current import ROOT, OUT, write_json, digest

DEST = ROOT / "paper/analysis/2026-10-12_local_comparison.json"


def correlation(x, y):
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue)}


def bootstrap(data, controls=(), seed=20261012):
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), (10000, len(data)))]

    def estimate(values):
        ranks = stats.rankdata(values, axis=-2)
        if controls:
            design = np.concatenate([np.ones((*ranks.shape[:-1], 1)), ranks[..., 2:]], axis=-1)
            full_rank = np.linalg.matrix_rank(design) == design.shape[-1]
            ranks = ranks[..., :2] - design @ (np.linalg.pinv(design) @ ranks[..., :2])
        else:
            full_rank = np.ones(ranks.shape[:-2], dtype=bool)
        x, y = ranks[..., 0], ranks[..., 1]
        x, y = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (x * y).sum(axis=-1) / np.sqrt((x*x).sum(axis=-1) * (y*y).sum(axis=-1))
        return r, full_rank

    point, valid_point = estimate(data)
    assert valid_point and np.isfinite(point)
    draws, full_rank = estimate(samples)
    valid = full_rank & np.isfinite(draws)
    return {"rho": float(point), "ci95": np.quantile(draws[valid], [.025, .975]).tolist(),
            "valid_bootstrap": int(valid.sum()), "rank_deficient": int((~full_rank).sum()),
            "requested_bootstrap": 10000}


def selection(rows, key, descending):
    k = int(len(rows) * .2)
    values = np.array([r[key] for r in rows]) * (1 if descending else -1)
    cut = np.sort(values)[-k]
    above = np.where(values > cut)[0]
    tied = np.where(values == cut)[0]
    expected = (sum(rows[i]["J24"] for i in above) +
                (k-len(above)) * np.mean([rows[i]["J24"] for i in tied])) / k
    selected = [int(rows[i]["seed"]) for i in np.argsort(-values, kind="stable")[:k]]
    return {"k": k, "expected_mean_J24": float(expected), "boundary_ties": len(tied),
            "selected_seeds_if_unambiguous": selected if len(tied) == k-len(above) else None}


def extended(rows, cohort, sources):
    result = summarize(rows, "full")
    x = np.array([r["background_idx"] for r in rows])
    y = np.array([r["J24"] for r in rows])
    d = np.array([r["dnorm"] for r in rows])
    j = np.array([r["J0"] for r in rows])
    result["rankers"] = {"negative_dnorm": correlation(-d, y), "J0": correlation(j, y),
                         "background_vs_dnorm": correlation(x, d), "ceiling_count": int(sum(j==500))}
    result["ranker_difference"] = {}
    rng = np.random.default_rng(20261012)
    indices = rng.integers(0, len(rows), (10000, len(rows)))
    for name, z in [("negative_dnorm", -d), ("J0", j)]:
        ranks = stats.rankdata(np.stack([x[indices], y[indices], z[indices]], axis=-1), axis=1)
        ranks -= ranks.mean(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pair = lambda a,b: (a*b).sum(axis=1)/np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
            diff = pair(ranks[:,:,0], ranks[:,:,1])-pair(ranks[:,:,2], ranks[:,:,1])
        valid = np.isfinite(diff)
        result["ranker_difference"][name] = {"point": correlation(x,y)["rho"]-correlation(z,y)["rho"],
            "ci95": np.quantile(diff[valid], [.025,.975]).tolist(), "valid_bootstrap": int(valid.sum())}
    result["selection"] = {key: selection(rows, key, descending) for key, descending in
                           [("background_idx", True), ("dnorm", False), ("J0", True)]}
    result["selection"]["random_expectation"] = float(y.mean())
    result["adjustment_intervals"] = {}
    result["dose_sensitivity"] = {}
    for dose_key in ["dose", "covered_dose"]:
        dose = np.array([r[dose_key] for r in rows])
        result["dose_sensitivity"][dose_key] = {"range": [float(dose.min()),float(dose.max())],
            "cv": float(dose.std()/dose.mean()), "vs_J24": correlation(dose,y),
            "vs_background": correlation(dose,x)}
        for name, controls in [("dose", [dose_key]), ("all", ["J0", "dnorm", dose_key])]:
            from paper.analysis.audit_outline_feedback import partial
            result["dose_sensitivity"][dose_key][name] = partial(rows, controls)
            if dose_key == "dose":
                result["adjustment_intervals"][name] = bootstrap(
                    [[r["background_idx"], r["J24"], *[r[c] for c in controls]] for r in rows], controls)
    payloads = []
    for row in rows:
        path = OUT / f"association/full/{cohort}/measurements/local_measurement_seed{row['seed']}.json"
        sources[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    uncovered = [p["dose"]["uncovered_HH_edges"] for p in payloads]
    result["uncovered_edges"] = {"range": [min(uncovered),max(uncovered)], "median": float(np.median(uncovered))}
    result["condition_sensitivity"] = {}
    for name, field in [("path", "J_static"), ("random", "J_rand")]:
        values = np.array([np.mean(p["result"][field])/r["J0"] for r,p in zip(rows,payloads)])
        result["condition_sensitivity"][name] = bootstrap(np.column_stack([values,y]))
    result["stage_sensitivity"] = {s: correlation(x,[p["result"]["arm_J"][s] for p in payloads])["rho"]
                                   for s in payloads[0]["result"]["arm_J"]}
    return result


def main():
    sources, new, panels = {}, {}, {}
    old = {}
    for cohort, filename in [("v3", "regime3_confirmatory.csv"), ("c2", "regime3_c2_confirmatory_full.csv")]:
        path = ROOT / "_phase8" / filename
        sources[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
        with path.open(encoding="utf-8-sig") as handle:
            historical = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(handle)]
        old[cohort] = correlation([r["background_idx"] for r in historical], [r["J24"] for r in historical])
        rows, inputs = load_rows(cohort, "full")
        sources.update(inputs)
        panels[cohort] = rows
        new[cohort] = extended(rows,cohort,sources)
    final, inputs = load_rows("c2", "full", final=True)
    sources.update(inputs)
    new["final"] = summarize(final,"full")
    for key, rows in [("initial", panels["c2"][:15]), ("added",panels["c2"][15:])]:
        new[key] = summarize(rows,"full")
    cross = [{**a,"J24": b["J24"]} for a,b in zip(panels["c2"],final)]
    new["cross"] = {**correlation([r["background_idx"] for r in cross],[r["J24"] for r in cross]),
                    **bootstrap([[r["background_idx"],r["J24"]] for r in cross])}
    old_final = ROOT / "_phase8/regime3_final_verify.csv"
    sources[old_final.relative_to(ROOT).as_posix()] = digest(old_final.read_bytes())
    with old_final.open(encoding="utf-8-sig") as h:
        rows = list(csv.DictReader(h))
    old["final"] = correlation([float(r["background_idx"]) for r in rows],[float(r["J24"]) for r in rows])
    old_current_path = ROOT / "paper/analysis/revise_v5_current_data_20261009.json"
    sources[old_current_path.relative_to(ROOT).as_posix()] = digest(old_current_path.read_bytes())
    old_current = json.loads(old_current_path.read_text(encoding="utf-8"))
    currents = {}
    for task in ["CartPole", "Acrobot"]:
        path = OUT / f"current/full/analysis_{task}.json"
        sources[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
        currents[task] = {"old":old_current["tasks"][task], "new":json.loads(path.read_text())}
    sources[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__).read_bytes())
    report = {"date":"2026-10-12", "old":old, "new":new, "current":currents,
        "sources":sources, "status":"New execution with original seed ranges; not recovery of deleted historical graphs or independent confirmation",
        "bootstrap":"10000 paired network resamples; ranks recomputed; invalid/rank-deficient draws counted, not replaced"}
    write_json(DEST,json_finite(report))
    print(json.dumps({"output":str(DEST), "main":{k:v['rho'] for k,v in new.items() if 'rho' in v},
                      "coverage":{c:new[c]['uncovered_edges'] for c in ['v3','c2']}}))


if __name__ == "__main__":
    main()
