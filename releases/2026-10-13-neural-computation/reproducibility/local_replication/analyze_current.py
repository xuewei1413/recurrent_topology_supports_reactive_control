"""Lineage-level summaries; no inferential claims from single-lineage pilots."""
import argparse
import json

from local_replication.run_current import OUT, ROOT, np, digest, canonical, write_json, CELLS
from _mediation.mc_stats import paired_test, cluster_bootstrap_ci, holm_correction


def analyze(task, profile):
    seeds = list(range(4000, 4012)) if task == "CartPole" else list(range(4012, 4024))
    if profile == "single":
        seeds = seeds[:1]
    directory = OUT / "current" / profile / task
    config_data = (directory.parent / "config.json").read_bytes()
    config_hash = digest(config_data)
    means = {}
    assertions = 0
    max_error = 0.0
    for seed in seeds:
        for mode, alpha in CELLS:
            values = []
            for d in range(3):
                for rep in range(2):
                    name = f"rescue_{task}_s{seed}_d{d}_r{rep}_{mode}_a{str(alpha).replace('.', 'p')}.json"
                    payload = json.loads((directory / name).read_text(encoding="utf-8"))
                    if (not payload.get("complete") or payload["config_sha256"] != config_hash
                            or [row["ep"] for row in payload["rows"]] != list(range(86400, 86420))):
                        raise ValueError(f"Incomplete or incompatible cell: {name}")
                    for row in payload["rows"]:
                        check = row["energy_assertion"]
                        if mode in ("action_only", "action_centered", "energy_random"):
                            if check["checks"] != row["n_steps"] * 20:
                                raise ValueError(f"Missing per-step assertions: {name}")
                        assertions += check["checks"]
                        max_error = max(max_error, check["max_abs_error"])
                    values.append(float(np.mean([row["return"] for row in payload["rows"]])))
            means[(seed, mode, alpha)] = float(np.mean(values))
    baseline = [means[(seed, "self_exact", 1.0)] for seed in seeds]
    contrasts = {}
    for control in ("energy_random", "time_shuffle", "action_only", "action_centered"):
        effects = {seed: means[(seed, "self_exact", 1.0)] - means[(seed, control, 1.0)] for seed in seeds}
        item = {"mean_difference": float(np.mean(list(effects.values()))),
                "per_lineage_difference": effects, "n_lineages": len(seeds)}
        if profile == "full":
            item["p_one_sided_raw_sum_sign_flip"] = paired_test(baseline, [means[(s, control, 1.0)] for s in seeds])[0]
            item["bootstrap_95ci"] = cluster_bootstrap_ci(effects)
        else:
            item["inference"] = "Not tested: one lineage per task is an instrument pilot"
        contrasts[control] = item
    if profile == "full":
        for family in (("energy_random", "time_shuffle"), ("action_only", "action_centered")):
            adjusted = holm_correction([contrasts[c]["p_one_sided_raw_sum_sign_flip"] for c in family])
            for control, result in zip(family, adjusted):
                contrasts[control]["p_holm"] = result["adjusted_p"]
    report = {"task": task, "profile": profile, "config_sha256": config_hash,
              "n_lineages": len(seeds), "variants_per_lineage": 6, "episodes_per_cell": 20,
              "cell_count": len(seeds) * 6 * len(CELLS), "contrasts": contrasts,
              "step_strength_assertions": assertions, "max_absolute_norm_error": max_error,
              "mean_returns": [{"mode": mode, "alpha": alpha,
                                "mean": float(np.mean([means[(seed, mode, alpha)] for seed in seeds]))}
                               for mode, alpha in CELLS],
              "interpretation": "Comparative rerun, not identification of a common dynamic mediator"}
    sources = [ROOT / "local_replication/analyze_current.py", ROOT / "_mediation/mc_stats.py",
               ROOT / "code/experiments_v2/analysis/statistics.py"]
    hashes = {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in sources}
    source_hash = digest(canonical(hashes))
    for path in sources:
        destination = OUT / "source_execution" / source_hash / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(path.read_bytes())
    report["analysis_sources"] = hashes
    write_json(directory.parent / f"analysis_{task}.json", report)
    print(json.dumps({"task": task, "n_lineages": len(seeds), "assertions": assertions,
                      "contrasts": {key: value["mean_difference"] for key, value in contrasts.items()}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("single", "full"), default="single")
    parser.add_argument("--task", choices=("CartPole", "Acrobot", "both"), default="both")
    args = parser.parse_args()
    for task in (["CartPole", "Acrobot"] if args.task == "both" else [args.task]):
        analyze(task, args.profile)


if __name__ == "__main__":
    main()
