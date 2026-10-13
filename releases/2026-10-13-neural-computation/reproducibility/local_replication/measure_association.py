"""Run original association measurements with local paths and added audit files."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import importlib.util
from itertools import combinations
import json
from numbers import Integral
from pathlib import Path
import sys
from uuid import uuid4

from local_replication.run_current import ROOT, OUT, INPUTS, np, digest, canonical, write_json, provenance
from local_replication.evolve_association import COHORTS


CALIBRATION_GUARD = {
    "revision": "calibration_guard_v1",
    "tau_max": 5,
    "minimum_valid_states": 250,
    "rule": "Retain only sampled windows with t+5 < episode length; no padding or resampling",
}
LEGACY_WRAPPER_SHA256 = "cdd733b9f2194edbfe4fbfd6243fae95696a633da29d7592fcea592627bd5ab6"
WRAPPER_PATH = "local_replication/measure_association.py"


def filter_calibration(episodes, states, tau_max=5, minimum_states=250):
    """Drop incomplete replay windows without changing any retained sample."""
    if tau_max < 0 or minimum_states < 0:
        raise ValueError("Negative replay horizon or minimum sample count")
    retained, dropped = [], []
    for state in states:
        ei, t = state
        if not isinstance(ei, Integral) or not isinstance(t, Integral):
            raise ValueError("Calibration indices must be integers")
        if not 0 <= ei < len(episodes) or t < 0:
            raise ValueError("Invalid calibration episode or negative time index")
        episode = episodes[ei]
        if t + tau_max >= len(episode["seq"]):
            dropped.append({"episode_index": int(ei), "time_index": int(t),
                            "episode_length": len(episode["seq"]),
                            "reason": "Incomplete replay window"})
            continue
        if t not in episode["snaps"] and str(t) not in episode["snaps"]:
            raise ValueError("Calibration snapshot is missing")
        retained.append(state)
    if len(retained) < minimum_states:
        raise ValueError(f"Only {len(retained)} valid calibration states; require {minimum_states}")
    audit = {"revision": CALIBRATION_GUARD["revision"], "tau_max": tau_max,
             "minimum_valid_states": minimum_states, "sampled_states": len(states),
             "retained_states": len(retained), "dropped_states": len(dropped),
             "dropped_windows": dropped}
    return (episodes, retained if dropped else states), audit


def compatible_legacy_config_hash(legacy, current):
    """Accept only the recorded pre-guard runner, with all scientific inputs fixed."""
    legacy_parameters = {k: v for k, v in legacy.items() if k != "additional_sources"}
    current_parameters = {k: v for k, v in current.items()
                          if k not in ("additional_sources", "calibration_guard")}
    if legacy_parameters != current_parameters or current.get("calibration_guard") != CALIBRATION_GUARD:
        raise ValueError("Calibration amendment cannot change the frozen measurement parameters")
    old_sources, new_sources = legacy["additional_sources"], current["additional_sources"]
    if old_sources.get(WRAPPER_PATH) != LEGACY_WRAPPER_SHA256:
        raise ValueError("Unknown legacy measurement runner")
    if WRAPPER_PATH not in new_sources:
        raise ValueError("Measurement runner missing from execution sources")
    if ({k: v for k, v in old_sources.items() if k != WRAPPER_PATH}
            != {k: v for k, v in new_sources.items() if k != WRAPPER_PATH}):
        raise ValueError("Source changes outside the calibration-guard runner")
    return digest(canonical(legacy))


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
    cohort_name, seed, profile, stage, config_hash, legacy_config_hash = job
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
        if (completed.get("complete") and completed["config_sha256"] in (config_hash, legacy_config_hash)
                and completed["topology_sha256"] == evolution_meta["topology_sha256"]):
            if (completed["cohort"], completed["profile"], completed["stage"], completed["result"]["seed"]) != (
                    cohort_name, profile, stage, seed):
                raise ValueError("Saved measurement belongs to a different job")
            if completed["config_sha256"] != config_hash:
                # Old results keep their original hashes; a separate audit establishes guard equivalence.
                old_attempt = (OUT / completed["attempt"]).resolve()
                if not old_attempt.is_relative_to(OUT.resolve()):
                    raise ValueError("Saved attempt lies outside the result directory")
                calibration_path = old_attempt / "calibration.json"
                episodes, states = json.loads(calibration_path.read_text(encoding="utf-8"))
                _, audit = filter_calibration(episodes, states)
                if audit["dropped_states"]:
                    raise ValueError("Legacy result contains incomplete calibration windows")
                with np.load(old_attempt / "fingerprint.npz", allow_pickle=False) as fingerprint_data:
                    if fingerprint_data["base_margins"].shape[:2] != (len(states), 6):
                        raise ValueError("Saved fingerprint does not match the calibrated samples")
                reuse = {"cohort": cohort_name, "seed": seed, "stage": stage,
                         "legacy_config_sha256": legacy_config_hash, "guard_config_sha256": config_hash,
                         "result_sha256": digest(final_path.read_bytes()),
                         "calibration_sha256": digest(calibration_path.read_bytes()),
                         "calibration_audit": audit, "status": "Retained legacy result; guard is a no-op"}
                reuse_path = OUT / "association" / profile / "measurement_reuse_checks" / config_hash / (
                    f"{cohort_name}_{stage}_seed{seed}.json")
                if reuse_path.exists() and reuse_path.read_bytes() != canonical(reuse):
                    raise ValueError("Saved calibration reuse audit changed")
                write_json(reuse_path, reuse)
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
        original = original_calibration(*args, **kwargs)
        save_calibration(attempt / "calibration_original.json", original)
        result, audit = filter_calibration(*original)
        save_calibration(attempt / "calibration.json", result)
        audit["original_calibration_sha256"] = digest((attempt / "calibration_original.json").read_bytes())
        audit["retained_calibration_sha256"] = digest((attempt / "calibration.json").read_bytes())
        cache["calibration_audit"] = audit
        write_json(attempt / "calibration_window_audit.json", audit)
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
                "calibration_audit": cache["calibration_audit"],
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
              "background_variants": 13, "evaluation_episodes": 20, "cohorts": COHORTS,
              "calibration_guard": CALIBRATION_GUARD}
    config_hash = digest(canonical(config))
    legacy_path = OUT / "association" / args.profile / f"measurement_config_{args.stage}.json"
    legacy_config_hash = None
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy_config_hash = compatible_legacy_config_hash(legacy, config)
        snapshot_path = OUT / "source_execution" / digest(canonical(legacy["additional_sources"])) / WRAPPER_PATH
        if digest(snapshot_path.read_bytes()) != LEGACY_WRAPPER_SHA256:
            raise ValueError("Archived legacy measurement source is missing or changed")
    config_path = OUT / "association" / args.profile / (
        f"measurement_config_{args.stage}_{CALIBRATION_GUARD['revision']}.json")
    if config_path.exists() and config_path.read_bytes() != canonical(config):
        raise ValueError("Frozen measurement configuration changed")
    write_json(config_path, config)
    seeds = args.seeds or (COHORTS[args.cohort]["seeds"][:1] if args.profile == "smoke" else COHORTS[args.cohort]["seeds"])
    if not set(seeds).issubset(COHORTS[args.cohort]["seeds"]):
        parser.error("Seeds outside original cohort")
    jobs = [(args.cohort, seed, args.profile, args.stage, config_hash, legacy_config_hash) for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(measure_one, jobs))
    write_json(OUT / "association" / args.profile / f"measurement_report_{args.cohort}_{args.stage}.json", results)


if __name__ == "__main__":
    main()
