"""Check manuscript reporting without modifying experiments or prior outputs."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUTS = []


def read_bytes(path):
    data = path.read_bytes()
    INPUTS.append({"path": str(path.relative_to(ROOT)),
                   "sha256": hashlib.sha256(data).hexdigest()})
    return data


def partial(x, y, controls):
    design = np.column_stack([np.ones(len(x))] +
                             [stats.rankdata(c) for c in controls])
    residuals = []
    for value in (x, y):
        rank = stats.rankdata(value)
        residuals.append(rank - design @ np.linalg.lstsq(design, rank, rcond=None)[0])
    r = float(stats.pearsonr(*residuals).statistic)
    df = len(x) - len(controls) - 2
    p = float(2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df))
    return {"rho": r, "df": df, "p_approx": p}


def main():
    prior = json.loads(read_bytes(HERE / "exploratory_ranker_compare.json"))
    doses = {}
    for path in sorted((HERE / "dose_v2").glob("*.json")):
        record = json.loads(read_bytes(path))
        doses[int(record["seed"])] = record
    output = {"analysis_status": "retrospective_reporting_audit", "cohorts": {}}
    for label, filename, old_key, old_dose_key in [
        ("v3", "regime3_confirmatory.csv", "v3_main", "dose_audit_v3"),
        ("c2", "regime3_c2_confirmatory_full.csv", "c2_full", "dose_audit_c2"),
    ]:
        data = read_bytes(ROOT / "_phase8" / filename).decode("utf-8-sig")
        rows = list(csv.DictReader(data.splitlines()))
        values = {key: np.array([float(r[key]) for r in rows])
                  for key in ("seed", "J24", "J0", "background_idx", "dnorm")}
        x, y, j0, dn = [values[k] for k in ("background_idx", "J24", "J0", "dnorm")]
        records = [doses[int(s)] for s in values["seed"]]
        assert np.max(np.abs(np.array([r["dnorm_recomputed"] for r in records]) - dn)) == 0
        rx = float(stats.spearmanr(x, y).statistic)
        assert np.isclose(rx, prior[old_key]["background_idx"]["rho"])
        report = {"n": len(y), "rho_bidx_dnorm": float(stats.spearmanr(x, dn).statistic),
                  "max_abs_dnorm_reconstruction_diff": 0.0}
        report["comparisons"] = {}
        for name, score in [("neg_dnorm", -dn), ("J0", j0)]:
            saved = prior[old_key]["vs_" + name]
            report["comparisons"][name] = {
                "sample_delta_rho": rx - float(stats.spearmanr(score, y).statistic),
                "bootstrap_median_delta_rho": saved["d_rho_med"],
                "percentile_CI_from_existing_10000_bootstraps": [saved["lo"], saved["hi"]],
            }
        report["dose"] = {}
        for denominator, key in [("adjacency_denominator", "frac_removed_at_24_adj"),
                                 ("covered_denominator", "frac_removed_at_24_covered")]:
            frac = np.array([r[key] for r in records])
            result = {"range": [float(frac.min()), float(frac.max())],
                      "dose_only": partial(x, y, [frac]),
                      "J0_dnorm_dose": partial(x, y, [j0, dn, frac])}
            assert np.isclose(result["J0_dnorm_dose"]["rho"],
                              prior[old_dose_key][denominator]["partial_bidx_given_J0_dn_dose"])
            report["dose"][denominator] = result
        k = round(0.20 * len(y))
        selected = {}
        for name, score in [("background_idx", x), ("neg_dnorm", -dn)]:
            order = np.argsort(-score, kind="stable")
            assert score[order[k - 1]] != score[order[k]]
            selected[name] = {"mean_J24": float(y[order[:k]].mean()),
                              "seeds": values["seed"][order[:k]].astype(int).tolist()}
        # Both cohorts select entirely from the maximum-J0 tie pool.
        tie = j0 == j0.max()
        assert tie.sum() >= k
        report["selection"] = {"k": k, "scores": selected,
                               "random_selection_exact_expected_J24": float(y.mean()),
                               "J0_uniform_tie_exact_expected_J24": float(y[tie].mean()),
                               "J0_tie_pool_size": int(tie.sum())}
        output["cohorts"][label] = report
    output["inputs"] = INPUTS
    output["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target = HERE / "audit_v5_reporting_20260917.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["cohorts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
