"""Rerun the original GA locally; smoke outputs cannot enter paper analyses."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
from uuid import uuid4

from local_replication.run_current import (
    ROOT, OUT, INPUTS, digest, canonical, write_json, provenance,
    ControllerProtocol, TASK_SPECS, ObservationEncoder, NeuronScaffold, gym, np)
from experiments_v2.topology_search.ga import GAConfig, TopologyGA

COHORTS = {
    "v3": {"seeds": list(range(4230, 4255)), "train": list(range(88600, 88700)),
           "selection": list(range(88700, 88710)), "evaluation": list(range(88800, 88820)),
           "filename_tag": "r3"},
    "c2": {"seeds": list(range(4260, 4290)), "train": list(range(88900, 89000)),
           "selection": list(range(89000, 89010)), "evaluation": list(range(89000, 89020)),
           "final_evaluation": list(range(89100, 89120)), "filename_tag": "c2"},
}


class RecordingEnv(gym.Wrapper):
    def __init__(self, env, callback):
        super().__init__(env)
        self.callback = callback
        self.episode_seed = None
        self.total = 0.0
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.episode_seed, self.total, self.steps = seed, 0.0, 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        result = self.env.step(action)
        self.total += float(result[1])
        self.steps += 1
        if result[2] or result[3]:
            self.callback({"episode_seed": self.episode_seed, "return": self.total,
                           "steps": self.steps, "terminated": bool(result[2]), "truncated": bool(result[3])})
        return result


class RecordingGA(TopologyGA):
    def __init__(self, *args, raw_file, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_file = raw_file
        self.raw_count = 0
        self.raw_steps = 0

    def _evaluate_genome(self, adjacency, eval_seeds, env_factory):
        context = {"generation": self.generation,
                   "phase": "selection" if self.evaluations_done >= self.config.total_eval_budget else "training",
                   "adjacency_sha256": digest(adjacency.tobytes())}

        def record(row):
            self.raw_file.write(json.dumps({**context, **row}, sort_keys=True) + "\n")
            self.raw_count += 1
            self.raw_steps += row["steps"]

        return super()._evaluate_genome(adjacency, eval_seeds,
                                       lambda: RecordingEnv(env_factory(), record))


def make_ga(cohort_name, seed, budget, raw_file=None):
    cohort = COHORTS[cohort_name]
    task, proto = TASK_SPECS["CartPole"], ControllerProtocol()
    scaffold = NeuronScaffold.from_seed(seed, task, proto)
    encoder = ObservationEncoder.load(INPUTS / "observation_stats/observation_stats_CartPole.json")
    cfg = GAConfig(pop_size=30, n_elite=6, tournament_size=3, lam_poisson=2.0,
                   n_eval_episodes=4, total_eval_budget=budget, seed_rotation_batch=5,
                   candidate_archive_size=30, selection_seeds=tuple(cohort["selection"]))
    args = dict(scaffold=scaffold, task=task, encoder=encoder, ctrl_proto=proto,
                ga_config=cfg, train_seeds=cohort["train"])
    return TopologyGA(**args) if raw_file is None else RecordingGA(**args, raw_file=raw_file)


def run_one(job):
    cohort_name, seed, profile, config_hash = job
    cohort = COHORTS[cohort_name]
    directory = OUT / "association" / profile / cohort_name / "evolution"
    metadata_path = directory / f"local_execution_seed{seed}.json"
    topology_path = directory / f"F0_topology_{cohort['filename_tag']}_CartPole_seed{seed}.npy"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (not metadata.get("complete") or metadata["config_sha256"] != config_hash
                or digest(np.load(topology_path, allow_pickle=False).tobytes()) != metadata["topology_sha256"]):
            raise ValueError(f"Incompatible completed result: {metadata_path}")
        return metadata
    budget = 120 if profile == "smoke" else 20000
    attempt = directory / "attempts" / f"seed{seed}_{uuid4().hex}"
    attempt.mkdir(parents=True)
    started = time.perf_counter()
    with (attempt / "episodes.jsonl").open("w", encoding="utf-8") as raw_file:
        ga = make_ga(cohort_name, seed, budget, raw_file)
        result = ga.run(seed, lambda: gym.make("CartPole-v1"), attempt, verbose=False)
    if result["n_candidates"] != result["n_candidates_evaluated"]:
        raise RuntimeError("A selection candidate failed; this attempt is not complete")
    expected_count = result["evaluations_done"] + result["n_candidates_evaluated"] * len(cohort["selection"])
    if ga.raw_count != expected_count:
        raise RuntimeError(f"Missing raw episode records: {ga.raw_count} != {expected_count}")
    equivalent = None
    if profile == "smoke":
        reference = make_ga(cohort_name, seed, budget).run(seed, lambda: gym.make("CartPole-v1"), verbose=False)
        np.testing.assert_array_equal(result["best_adjacency"], reference["best_adjacency"])
        if result["best_fitness"] != reference["best_fitness"]:
            raise AssertionError("Recording wrapper changed the GA result")
        equivalent = True
    np.save(topology_path, result["best_adjacency"], allow_pickle=False)
    metadata = {"complete": True, "config_sha256": config_hash, "cohort": cohort_name,
                "evolution_seed": seed, "profile": profile,
                "status": "smoke test only" if profile == "smoke" else "new execution, not recovered historical identity",
                "topology_sha256": digest(result["best_adjacency"].tobytes()),
                "best_fitness": result["best_fitness"], "requested_training_budget": budget,
                "actual_training_episodes": result["evaluations_done"], "raw_episodes": ga.raw_count,
                "raw_environment_steps": ga.raw_steps,
                "n_generations": result["n_generations"], "n_candidates": result["n_candidates"],
                "elapsed_seconds_including_selection_and_smoke_equivalence": time.perf_counter() - started,
                "recording_wrapper_bit_identical": equivalent,
                "attempt": str(attempt.relative_to(OUT)).replace("\\", "/")}
    write_json(metadata_path, metadata)
    print(f"{cohort_name} seed{seed}: fitness={result['best_fitness']:.1f}, "
          f"train={result['evaluations_done']}, raw={ga.raw_count}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cohort", choices=("v3", "c2", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    cohorts = list(COHORTS) if args.cohort == "both" else [args.cohort]
    if args.seeds and args.cohort == "both":
        parser.error("Specify a cohort when selecting seeds")
    base = provenance()
    files = [Path(__file__)] + list((ROOT / "code/experiments_v2/topology_search").glob("*.py"))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in files}
    source_hash = digest(canonical(hashes))
    for p in files:
        destination = OUT / "source_execution" / source_hash / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(p.read_bytes())
    config = {"provenance": base, "additional_sources": hashes, "cohorts": COHORTS,
              "ga": asdict(make_ga("v3", 4230, 120 if args.profile == "smoke" else 20000).config),
              "profile": args.profile, "selection_eval_overlap_c2": list(range(89000, 89010)),
              "resume": "completed seeds only; incomplete attempts restart with a new attempt directory"}
    config_hash = digest(canonical(config))
    config_path = OUT / "association" / args.profile / "config.json"
    if config_path.exists() and config_path.read_bytes() != canonical(config):
        raise ValueError("Frozen configuration differs; use a separate result root")
    write_json(config_path, config)
    jobs = []
    for cohort in cohorts:
        seeds = args.seeds or (COHORTS[cohort]["seeds"][:1] if args.profile == "smoke" else COHORTS[cohort]["seeds"])
        if not set(seeds).issubset(COHORTS[cohort]["seeds"]):
            parser.error(f"Seeds outside original {cohort} cohort")
        jobs += [(cohort, seed, args.profile, config_hash) for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(run_one, jobs))
    write_json(OUT / "association" / args.profile / f"report_{args.cohort}.json", results)


if __name__ == "__main__":
    main()
