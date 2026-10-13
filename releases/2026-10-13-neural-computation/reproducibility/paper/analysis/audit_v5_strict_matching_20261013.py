"""Audit stored equal-score matching records and reproduce historical summaries."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tmp/v78/results_v8/e5b_strict"
SUMMARY = ROOT / "_phase8/v8_summary.json"
OUTPUT = ROOT / "paper/analysis/2026-10-13_strict_matching_audit.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    historical = json.loads(SUMMARY.read_text(encoding="utf-8"))
    result = {
        "analysis": "post-hoc audit of historical exact-matching records",
        "score": "0.25 * pin1 * pout1",
        "matching_key": ["sign", "source_group_8", "target_group_8", "pin1", "pout1"],
        "summary_source_sha256": sha256(SUMMARY),
        "cohorts": {},
    }
    for task, bound in [("CartPole", 30.0), ("Acrobot", 10.0)]:
        files = sorted(SOURCE.glob(f"e5b_strict_main_{task}_seed*.json"))
        assert len(files) == 20
        rows, differences, scores = [], [], []
        for path in files:
            record = json.loads(path.read_text(encoding="utf-8"))
            pairs = [(ast.literal_eval(k), v) for k, v in record["key_counts"].items()]
            assert sum(v[2] for _, v in pairs) == record["n_strict"]
            for key, counts in pairs:
                assert len(key) == 5 and 0 <= key[1] < 8 and 0 <= key[2] < 8
                assert counts[2] == min(counts[:2])
                scores.extend([0.25 * key[3] * key[4]] * counts[2])
            evaluated = "drop_skel_strict" in record and "drop_matched_mean" in record
            rows.append({
                "source": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "n_selected": record["n_skel"],
                "n_matched": record["n_strict"],
                "matched_fraction": record["strict_frac_of_skel"],
                "evaluated": evaluated,
                "skip_reason": record.get("skipped"),
            })
            if evaluated:
                differences.append(record["drop_skel_strict"] - record["drop_matched_mean"])
        values = np.asarray(differences)
        ci = stats.t.interval(0.90, len(values)-1, loc=values.mean(), scale=stats.sem(values))
        p = float(stats.wilcoxon(values).pvalue)
        archived = historical[f"A1_{task}"]["tost_skel_vs_matched"]
        assert len(values) == archived["n"]
        assert np.isclose(values.mean(), archived["mean"])
        assert np.allclose(ci, archived["ci90_t"])
        assert np.isclose(p, archived["wilcoxon_p"])
        result["cohorts"][task] = {
            "network_records": len(rows),
            "evaluated_networks": len(values),
            "median_matched_fraction": float(np.median([r["matched_fraction"] for r in rows])),
            "matched_selected_edges": len(scores),
            "zero_score_matched_selected_edges": sum(s == 0 for s in scores),
            "matched_score_values": sorted(set(scores)),
            "mean_selected_minus_control_loss": float(values.mean()),
            "median_selected_minus_control_loss": float(np.median(values)),
            "mean_difference_90pct_t_interval": [float(x) for x in ci],
            "wilcoxon_p": p,
            "equivalence_bound": bound,
            "historical_tost_p": archived["tost_p"],
            "equivalence_confirmed": archived["equivalent"],
            "archived_summaries_reproduced": True,
            "records": rows,
        }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "records"}
                      for k, v in result["cohorts"].items()}, indent=2))


if __name__ == "__main__":
    main()
