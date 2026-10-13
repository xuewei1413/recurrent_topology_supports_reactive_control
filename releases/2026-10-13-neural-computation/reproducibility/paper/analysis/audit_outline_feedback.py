"""Reproduce numerical checks for the 2026-09-15 outline feedback."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {}


def read_csv(name):
    path = ROOT / "_phase8" / name
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def corr(x, y):
    result = stats.spearmanr(x, y)
    return {"n": len(x), "rho": float(result.statistic), "nominal_p": float(result.pvalue)}


def partial(rows, controls, ranked=True, rerank_residuals=False):
    x = np.array([r["background_idx"] for r in rows])
    y = np.array([r["J24"] for r in rows])
    z = np.array([[r[c] for c in controls] for r in rows])
    if ranked:
        x, y = stats.rankdata(x), stats.rankdata(y)
        z = np.column_stack([stats.rankdata(z[:, i]) for i in range(z.shape[1])])
    z = np.column_stack([np.ones(len(rows)), z])
    rx = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    ry = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    r = float(stats.spearmanr(rx, ry).statistic if rerank_residuals
              else np.corrcoef(rx, ry)[0, 1])
    df = len(rows) - len(controls) - 2
    return {"r": r, "approximate_t_p": float(2 * stats.t.sf(abs(r) * np.sqrt(df / (1-r*r)), df)),
            "df": df, "controls": controls}


def describe(rows):
    return corr([r["background_idx"] for r in rows], [r["J24"] for r in rows])


def main():
    v3 = read_csv("regime3_confirmatory.csv")
    c2 = read_csv("regime3_c2_confirmatory_full.csv")
    final = read_csv("regime3_final_verify.csv")
    acr = read_csv("acr_crossfamily_test.csv")
    assert len(v3) == 25 and len(c2) == len(final) == 30
    old = {int(r["seed"]): r for r in c2}
    new = {int(r["seed"]): r for r in final}
    assert old.keys() == new.keys()
    seeds = sorted(old)
    audit = {
        "definitions": {
            "standard_partial_spearman": "Rank x, y and controls once; OLS with intercept; Pearson correlation of residuals.",
            "partial_p": "Approximate t test, df=n-k-2; not an exact permutation test.",
            "endpoint": "J24 is raw return, not J24/J0.",
            "cross_panel": "Old background index versus new raw J24 on the same 30 topologies; post hoc sensitivity, not new independent replication.",
            "acrobot_index": "Mean over 13 replicas of (J+500)/(J_F0+500).",
        },
        "v3_raw": describe(v3),
        "v3_partial": {},
        "v3_unnormalized_background_vs_raw_J24": corr([r["background_idx"]*r["J0"] for r in v3], [r["J24"] for r in v3]),
        "c2_original": describe([r for r in c2 if r["seed"] <= 4274]),
        "c2_added": describe([r for r in c2 if r["seed"] >= 4275]),
        "c2_combined": describe(c2),
        "c2_terminal_contemporaneous": describe(final),
        "c2_old_background_new_J24": corr([old[s]["background_idx"] for s in seeds], [new[s]["J24"] for s in seeds]),
        "c2_new_background_old_J24": corr([new[s]["background_idx"] for s in seeds], [old[s]["J24"] for s in seeds]),
        "acrobot_normalized": describe(acr),
        "acrobot_unnormalized": corr([r["background_idx"]*(r["J0"]+500)-500 for r in acr], [r["J24"] for r in acr]),
    }
    for controls in (["J0"], ["J0", "dnorm"]):
        audit["v3_partial"]["+".join(controls)] = {
            "standard": partial(v3, controls),
            "raw_pearson": partial(v3, controls, ranked=False),
            "rank_residuals_again": partial(v3, controls, rerank_residuals=True),
            "raw_residuals_then_spearman": partial(v3, controls, ranked=False, rerank_residuals=True),
        }
    errors = []
    for row in acr:
        path = ROOT / "tmp/acr_region" / f"region_acr_seed{int(row['seed'])}.json"
        INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = json.loads(path.read_text(encoding="utf-8"))
        mean_return = np.mean(raw["J_static_per_rep"] + raw["J_rand_per_rep"])
        expected = (mean_return + 500) / (raw["J_F0"] + 500)
        errors.append(abs(expected - row["background_idx"]))
    audit["acrobot_raw_json_max_index_error"] = float(max(errors))
    assert max(errors) < 1e-12
    audit["input_sha256"] = INPUTS
    output = Path(__file__).with_name("audit_outline_feedback_results.json")
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
