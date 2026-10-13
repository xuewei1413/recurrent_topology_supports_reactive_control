"""Run original association measurements with local paths and added audit files."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys
from uuid import uuid4

from local_replication.run_current import ROOT, OUT, INPUTS, np, digest, canonical, write_json, provenance
from local_replication.evolve_association import COHORTS


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_calibration(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, default=encode, allow_nan=False), encoding="utf-8")


def measure_one(job):
    cohort_name, seed, profile, stage, config_hash = job
    if stage == "final" and cohort_name != "c2":
        raise ValueError("The original final stage uses c2 networks only")
    name = "_regime3_final_verify" if stage == "final" else (
        "_regime3_measure" if cohort_name == "v3" else "_regime3_c2_measure")
    module = load_module(name)
    directory = OUT / "association" / profile / cohort_name
    evolution_meta = json.loads((directory / "evolution" / f"local_execution_seed{seed}.json").read_text(encoding="utf-8"))
    if not evolution_meta["complete"] or evolution_meta["profile"] != profile:
        raise ValueError("Evolution is incomplete or belongs to a different profile")
    if profile == "full" and evolution_meta["actual_training_episodes"] < 20000:
        raise ValueError("Smoke topology cannot enter full measurements")
    topology = directory / "evolution" / f"F0_topology_{COHORTS[cohort_name]['filename_tag']}_CartPole_seed{seed}.npy"
    adjacency = np.load(topology, allow_pickle=False)
    if digest(adjacency.tobytes()) != evolution_meta["topology_sha256"]:
        raise ValueError("Topology identity changed after evolution")
    result_dir = directory / ("final_measurements" if stage == "final" else "measurements")
    final_path = result_dir / f"local_measurement_seed{seed}.json"
    if final_path.exists():
        completed = json.loads(final_path.read_text(encoding="utf-8"))
        if (completed.get("complete") and completed["config_sha256"] == config_hash
                and completed["topology_sha256"] == evolution_meta["topology_sha256"]):
            return completed
        raise ValueError("Existing measurement is incomplete or incompatible")
    attempt = result_dir / "attempts" / f"seed{seed}_{uuid4().hex}"
    attempt.mkdir(parents=True)
    module.ROOT = OUT / "measurement_inputs"
    module.EVO = directory / "evolution"
    module.OUT = attempt
    cache = {"rewires": 0, "evaluations": [], "pairs": []}
    base_controller = module.UnifiedSNNController

    class AuditedController(base_controller):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            adj = np.asarray(args[1] if len(args) > 1 else kwargs["adjacency"])
            self.audit_sha = digest(adj.tobytes())
            artifact = attempt / "evaluated_topologies" / f"{self.audit_sha}.npy"
            artifact.parent.mkdir(exist_ok=True)
            if not artifact.exists():
                np.save(artifact, adj, allow_pickle=False)

        def evaluate(self, env, seeds, max_steps, collect_diagnostics=False):
            result = super().evaluate(env, seeds, max_steps, collect_diagnostics)
            cache["evaluations"].append({"case": len(cache["evaluations"]),
                                         "adjacency_sha256": self.audit_sha,
                                         "episodes": list(seeds), "returns": result["rewards"]})
            return result

    module.UnifiedSNNController = AuditedController
    original_calibration, original_groups = module.collect_calibration, module.fixed_groups
    original_fingerprint, original_replay = module.fingerprint, module.replay_margins
    original_rewire = module.restricted_rewire

    def calibration(*args, **kwargs):
        result = original_calibration(*args, **kwargs)
        save_calibration(attempt / "calibration.json", result)
        return result

    def groups(*args, **kwargs):
        result = original_groups(*args, **kwargs)
        cache["groups"] = result
        write_json(attempt / "groups.json", [group.tolist() for group in result])
        return result

    def fingerprint(*args, **kwargs):
        result = original_fingerprint(*args, **kwargs)
        cache["L"], cache["base"], cache["n_edges"] = result
        np.savez_compressed(attempt / "fingerprint.npz", L=result[0], base_margins=result[1], n_edges=result[2])
        return result

    def replay(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        cache["pairs"].append(float(np.abs(result[:, 1:, :] - cache["base"][:, 1:, :]).sum(axis=2).mean(axis=0).sum()))
        return result

    def rewire(*args, **kwargs):
        child, done = original_rewire(*args, **kwargs)
        parent = args[0]
        np.testing.assert_array_equal(child.sum(axis=0), parent.sum(axis=0))
        np.testing.assert_array_equal(child.sum(axis=1), parent.sum(axis=1))
        assert not np.diag(child).any()
        index = cache["rewires"]
        np.save(attempt / f"background_{index}.npy", child, allow_pickle=False)
        write_json(attempt / f"background_{index}.json", {"swaps_completed": done,
                   "swaps_requested": args[4], "keep_edges_target_source": np.argwhere(args[3]).tolist()})
        cache["rewires"] += 1
        return child, done

    module.collect_calibration, module.fixed_groups = calibration, groups
    module.fingerprint, module.replay_margins, module.restricted_rewire = fingerprint, replay, rewire
    result = module.one_seed(seed)
    if "fluct_std" in result and not np.isfinite(result["fluct_std"]):
        result["fluct_std"] = None
        result["fluct_std_status"] = "Undefined: no positive output-count sample; network retained"
    if cache["rewires"] != 13:
        raise AssertionError("Not all 13 background variants were measured")
    edge_groups = cache["groups"]
    D = cache["L"].sum(axis=1)
    active = [g for g in range(len(edge_groups)) if len(edge_groups[g])]
    pair_indices = list(combinations(active, 2))
    if len(pair_indices) != len(cache["pairs"]):
        raise AssertionError("Incomplete pair-interaction measurements")
    I = np.zeros((module.G, module.G))
    for (g1, g2), pair_D in zip(pair_indices, cache["pairs"]):
        I[g1, g2] = I[g2, g1] = pair_D - float(D[g1]) - float(D[g2])
    S = D[:, None] + D[None, :]
    C = np.where(S > 1e-9, -I / np.where(S > 1e-9, S, 1.0), 0.0)
    np.fill_diagonal(C, 0.0)
    order = np.argsort(-np.maximum(C, 0).sum(axis=1))
    removed_edges = np.vstack([edge_groups[g] for g in order[:24]])
    removed_unique = {tuple(edge) for edge in removed_edges}
    covered = {tuple(edge) for group in edge_groups for edge in group}
    hh = module.hh_edge_list(adjacency, 8, 2)
    dnorm = float(np.sum(D / np.maximum(cache["n_edges"], 1)))
    if stage != "final" and dnorm != result["dnorm_sum"]:
        raise AssertionError("Saved dose audit disagrees with the original measurement")
    audit = {"group_order": order.tolist(), "pair_indices": pair_indices, "pair_D": cache["pairs"],
             "true_HH_edges": len(hh), "covered_HH_edges": len(covered),
             "uncovered_HH_edges": len(hh) - len(covered), "removed_at_step24": len(removed_unique),
             "fraction_adjacency_denominator": len(removed_unique) / len(hh),
             "fraction_covered_denominator": len(removed_unique) / len(covered), "dnorm_sum": dnorm}
    write_json(attempt / "dose_audit.json", audit)
    write_json(attempt / "raw_evaluations.json", cache["evaluations"])
    if len(cache["evaluations"]) != (15 if stage == "final" else 24):
        raise AssertionError("Incomplete primary behavioral evaluations")
    complete = {"complete": True, "config_sha256": config_hash,
                "topology_sha256": evolution_meta["topology_sha256"], "cohort": cohort_name,
                "profile": profile, "stage": stage, "result": result, "dose": audit,
                "attempt": attempt.relative_to(OUT).as_posix(),
                "status": "instrument smoke only" if profile == "smoke" else "new matched topology and outcomes"}
    write_json(final_path, complete)
    print(f"{cohort_name} seed{seed} {stage}: F0={result['J_F0']:.1f}, "
          f"J24={result['J24'] if stage == 'final' else result['arm_J']['24']:.1f}, "
          f"{result['minutes']:.2f}min", flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cohort", choices=("v3", "c2"), default="v3")
    parser.add_argument("--stage", choices=("measurement", "final"), default="measurement")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if args.stage == "final" and args.cohort != "c2":
        parser.error("Final stage requires c2")
    stats = OUT / "measurement_inputs/results_v2/pilot/observation_stats/observation_stats_CartPole.json"
    data = (INPUTS / "observation_stats/observation_stats_CartPole.json").read_bytes()
    if stats.exists() and stats.read_bytes() != data:
        raise ValueError("Measurement encoder changed")
    stats.parent.mkdir(parents=True, exist_ok=True)
    if not stats.exists():
        stats.write_bytes(data)
    for name in ("_regime3_measure", "_regime3_c2_measure", "_regime3_final_verify"):
        load_module(name)
    prov = provenance()
    paths = {Path(module.__file__) for name, module in sys.modules.items()
             if name.startswith("experiments_v2") and getattr(module, "__file__", None)}
    paths.update(ROOT / f"{name}.py" for name in ("_regime3_measure", "_regime3_c2_measure", "_regime3_final_verify"))
    paths.update([Path(__file__), ROOT / "local_replication/evolve_association.py",
                  ROOT / "local_replication/requirements-association.txt"])
    hashes = {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in sorted(paths)}
    source_hash = digest(canonical(hashes))
    for path in paths:
        destination = OUT / "source_execution" / source_hash / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(path.read_bytes())
    config = {"provenance": prov, "additional_sources": hashes, "scipy": version("scipy"),
              "profile": args.profile, "stage": args.stage, "calibration_states": 250,
              "historical_groups": "unchanged, including incomplete coverage",
              "background_variants": 13, "evaluation_episodes": 20, "cohorts": COHORTS}
    config_hash = digest(canonical(config))
    config_path = OUT / "association" / args.profile / f"measurement_config_{args.stage}.json"
    if config_path.exists() and config_path.read_bytes() != canonical(config):
        raise ValueError("Frozen measurement configuration changed")
    write_json(config_path, config)
    seeds = args.seeds or (COHORTS[args.cohort]["seeds"][:1] if args.profile == "smoke" else COHORTS[args.cohort]["seeds"])
    if not set(seeds).issubset(COHORTS[args.cohort]["seeds"]):
        parser.error("Seeds outside original cohort")
    jobs = [(args.cohort, seed, args.profile, args.stage, config_hash) for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(measure_one, jobs))
    write_json(OUT / "association" / args.profile / f"measurement_report_{args.cohort}_{args.stage}.json", results)


if __name__ == "__main__":
    main()
