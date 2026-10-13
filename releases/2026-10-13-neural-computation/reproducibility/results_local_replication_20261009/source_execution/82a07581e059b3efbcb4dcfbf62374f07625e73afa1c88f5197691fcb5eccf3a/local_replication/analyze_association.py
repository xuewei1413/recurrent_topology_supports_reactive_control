"""Compare newly matched rewiring/deletion measurements without mixing old outcomes."""
import argparse
import csv
import json
from pathlib import Path
import warnings

from local_replication.run_current import ROOT, OUT, np, digest, canonical, write_json
from local_replication.evolve_association import COHORTS
from paper.analysis.audit_outline_feedback import corr, partial
from scipy import stats


def interval(rows):
    x = np.array([row["background_idx"] for row in rows])
    y = np.array([row["J24"] for row in rows])
    rng = np.random.RandomState(20261009)
    estimates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        for _ in range(10000):
            sample = rng.randint(0, len(rows), len(rows))
            value = stats.spearmanr(x[sample], y[sample]).statistic
            if np.isfinite(value):
                estimates.append(float(value))
    return {"interval": np.percentile(estimates, [2.5, 97.5]).tolist() if estimates else None,
            "valid_resamples": len(estimates), "requested_resamples": 10000,
            "method": "paired network percentile bootstrap"}


def load_rows(cohort, profile, final=False):
    directory = OUT / "association" / profile / cohort
    seeds = COHORTS[cohort]["seeds"] if profile == "full" else COHORTS[cohort]["seeds"][:1]
    rows, sources = [], {}
    folder = "final_measurements" if final else "measurements"
    for seed in seeds:
        path = directory / folder / f"local_measurement_seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete") or payload["profile"] != profile:
            raise ValueError(f"Incomplete or incompatible measurement: {path}")
        topology = directory / "evolution" / f"F0_topology_{COHORTS[cohort]['filename_tag']}_CartPole_seed{seed}.npy"
        if digest(np.load(topology, allow_pickle=False).tobytes()) != payload["topology_sha256"]:
            raise ValueError("Topology differs from the measured network")
        result = payload["result"]
        if result["J_F0"] <= 0 or len(result["J_static"]) != 3 or len(result["J_rand"]) != 10:
            raise ValueError("Invalid baseline or incomplete background variants")
        rows.append({"seed": seed, "J0": result["J_F0"],
                     "background_idx": float(np.mean(result["J_static"] + result["J_rand"]) / result["J_F0"]),
                     "J24": result["J24"] if final else result["arm_J"]["24"],
                     "dnorm": payload["dose"]["dnorm_sum"],
                     "dose": payload["dose"]["fraction_adjacency_denominator"],
                     "covered_dose": payload["dose"]["fraction_covered_denominator"],
                     "topology_sha256": payload["topology_sha256"]})
        sources[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
    return rows, sources


def summarize(rows, profile):
    if profile != "full":
        return {"n": len(rows), "inference": "Not tested: instrument smoke network"}
    report = {**corr([row["background_idx"] for row in rows], [row["J24"] for row in rows]),
              "bootstrap_95ci": interval(rows),
              "raw_background_vs_raw_J24": corr([r["background_idx"] * r["J0"] for r in rows], [r["J24"] for r in rows]),
              "dose_vs_J24": corr([r["dose"] for r in rows], [r["J24"] for r in rows]),
              "partial_rank_correlations": {"+".join(controls): partial(rows, controls) for controls in
                                            (["dose"], ["J0"], ["J0", "dnorm"], ["J0", "dnorm", "dose"])} }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--cohort", choices=("v3", "c2", "both"), default="both")
    args = parser.parse_args()
    reports, sources, all_rows = {}, {}, {}
    for cohort in (["v3", "c2"] if args.cohort == "both" else [args.cohort]):
        rows, inputs = load_rows(cohort, args.profile)
        reports[cohort] = summarize(rows, args.profile)
        sources.update(inputs)
        all_rows[cohort] = rows
        csv_path = OUT / "association" / args.profile / f"new_execution_{cohort}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if cohort == "c2" and args.profile == "full":
            reports["c2_initial_15"] = summarize(rows[:15], args.profile)
            reports["c2_added_15"] = summarize(rows[15:], args.profile)
            final, final_sources = load_rows(cohort, args.profile, final=True)
            sources.update(final_sources)
            reports["c2_final"] = summarize(final, args.profile)
            if any(a["topology_sha256"] != b["topology_sha256"] for a, b in zip(rows, final)):
                raise ValueError("Original and retest panels use different new networks")
            reports["c2_old_panel_new_endpoint"] = corr([r["background_idx"] for r in rows], [r["J24"] for r in final])
            reports["c2_new_panel_old_endpoint"] = corr([r["background_idx"] for r in final], [r["J24"] for r in rows])
    code = [Path(__file__), ROOT / "paper/analysis/audit_outline_feedback.py"]
    hashes = {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in code}
    key = digest(canonical(hashes))
    for path in code:
        destination = OUT / "source_execution" / key / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(path.read_bytes())
    report = {"profile": args.profile, "reports": reports, "inputs": sources, "analysis_sources": hashes,
              "definitions": {"background_idx": "mean of 13 background returns divided by own F0 return",
                              "endpoint": "raw J24 return, not a second normalized ratio",
                              "statistics": "nominal Spearman p; standard rank residual correlation with approximate t p",
                              "c2_overlap": "selection 89000-89009 overlaps measurement; final 89100-89119 does not",
                              "status": "rerun, not new independent confirmation; no claim of an identified integration mediator"}}
    write_json(OUT / "association" / args.profile / f"analysis_{args.cohort}.json", report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
