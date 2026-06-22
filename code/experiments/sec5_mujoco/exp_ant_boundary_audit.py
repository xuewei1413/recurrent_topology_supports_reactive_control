"""Ant-v4 boundary audit for fixed-weight topology-only SNN.

This script is intentionally separate from exp_hopper_evolution.py so it does
not overwrite Hopper/Pendulum reports. It records a traceable boundary result:
random baseline, topology-only GA evolution, and strict 20-episode evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.continuous_snn import ContinuousSNNController

OUT_DIR = ROOT / "experiments" / "results" / "embodied_reflex" / "ant_boundary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_NAME = "Ant-v4"
CONFIG = {
    "env_name": ENV_NAME,
    "obs_dim": 27,
    "action_dim": 8,
    "n_total": 128,
    "n_input": 54,
    "n_output": 8,
    "tau": 5.0,
    "threshold": 0.8,
    "weight_scale": 0.6,
    "T_steps": 20,
    "sparsity": 0.10,
    "pop_size": 40,
    "n_generations": 80,
    "n_elite": 3,
    "parent_pool": 10,
    "mutation_rate": 0.03,
    "n_eval_episodes": 5,
    "strict_episodes": 20,
    "max_steps": 1000,
    "inhibitory_fraction_hidden": 0.30,
}


def make_env(seed: int | None = None):
    import gymnasium as gym
    env = gym.make(ENV_NAME)
    if seed is not None:
        env.reset(seed=seed)
    return env


def random_topology(rng: np.random.Generator, n: int, sparsity: float) -> np.ndarray:
    adj = (rng.random((n, n)) < sparsity).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def mutate_topology(rng: np.random.Generator, adj: np.ndarray, rate: float) -> np.ndarray:
    child = adj.copy()
    mask = rng.random(child.shape) < rate
    child[mask] = 1.0 - child[mask]
    np.fill_diagonal(child, 0.0)
    return child


def make_neuron_types(rng: np.random.Generator, config: dict) -> np.ndarray:
    n = config["n_total"]
    types = np.ones(n, dtype=np.float64)
    hidden = np.arange(config["n_input"], n - config["n_output"])
    n_inh = int(len(hidden) * config["inhibitory_fraction_hidden"])
    if n_inh > 0:
        inh_idx = rng.choice(hidden, n_inh, replace=False)
        types[inh_idx] = -1.0
    return types


def evaluate_controller(adj: np.ndarray, types: np.ndarray, config: dict,
                        n_episodes: int, seed: int, max_steps: int | None = None) -> dict:
    ctrl = ContinuousSNNController(config, adj, types)
    env = make_env()
    max_steps = int(max_steps or config["max_steps"])
    rewards = []
    steps = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        ctrl.reset()
        total = 0.0
        n_steps = 0
        for _ in range(max_steps):
            action = ctrl.forward(np.asarray(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            n_steps += 1
            if terminated or truncated:
                break
        rewards.append(total)
        steps.append(n_steps)
    env.close()
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    steps_arr = np.asarray(steps, dtype=np.float64)
    return {
        "mean": float(rewards_arr.mean()),
        "std": float(rewards_arr.std(ddof=1)) if len(rewards_arr) > 1 else 0.0,
        "min": float(rewards_arr.min()),
        "max": float(rewards_arr.max()),
        "rewards": rewards,
        "steps_mean": float(steps_arr.mean()),
        "steps": steps,
        "n_episodes": int(n_episodes),
    }


def evaluate_random_policy(n_episodes: int, seed: int, max_steps: int) -> dict:
    env = make_env()
    rewards = []
    steps = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        total = 0.0
        n_steps = 0
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            n_steps += 1
            if terminated or truncated:
                break
        rewards.append(total)
        steps.append(n_steps)
    env.close()
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    steps_arr = np.asarray(steps, dtype=np.float64)
    return {
        "mean": float(rewards_arr.mean()),
        "std": float(rewards_arr.std(ddof=1)) if len(rewards_arr) > 1 else 0.0,
        "min": float(rewards_arr.min()),
        "max": float(rewards_arr.max()),
        "rewards": rewards,
        "steps_mean": float(steps_arr.mean()),
        "steps": steps,
        "n_episodes": int(n_episodes),
    }


def run_seed(seed: int, config: dict, skip_random: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    types = make_neuron_types(rng, config)
    pop = [random_topology(rng, config["n_total"], config["sparsity"])
           for _ in range(config["pop_size"])]
    best_adj = None
    best_training_fitness = -np.inf
    best_history = []
    mean_history = []
    conn_history = []

    log_path = OUT_DIR / f"ant_evolution_seed{seed}.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        def emit(msg: str) -> None:
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

        emit(f"Ant-v4 boundary audit | seed={seed}")
        emit(f"config={json.dumps(config, sort_keys=True)}")
        emit(f"types: exc={int((types == 1).sum())}, inh={int((types == -1).sum())}")
        emit(f"{'Gen':>4} {'Best':>10} {'Mean':>10} {'Conns':>7}")

        for gen in range(1, config["n_generations"] + 1):
            fitnesses = []
            for adj in pop:
                ev = evaluate_controller(adj, types, config,
                                         n_episodes=config["n_eval_episodes"],
                                         seed=seed * 10000 + gen)
                fitnesses.append(ev["mean"])
            fitnesses = np.asarray(fitnesses, dtype=np.float64)
            order = np.argsort(fitnesses)[::-1]
            gen_best = float(fitnesses[order[0]])
            gen_mean = float(fitnesses.mean())
            best_conn = int(pop[order[0]].sum())

            best_history.append(gen_best)
            mean_history.append(gen_mean)
            conn_history.append(best_conn)
            if gen_best > best_training_fitness:
                best_training_fitness = gen_best
                best_adj = pop[order[0]].copy()

            emit(f"{gen:4d} {gen_best:10.2f} {gen_mean:10.2f} {best_conn:7d}")

            new_pop = [pop[i].copy() for i in order[:config["n_elite"]]]
            parent_pool = order[:config["parent_pool"]]
            while len(new_pop) < config["pop_size"]:
                parent = pop[int(rng.choice(parent_pool))]
                new_pop.append(mutate_topology(rng, parent, config["mutation_rate"]))
            pop = new_pop

    if best_adj is None:
        raise RuntimeError("Evolution did not produce a best topology")

    strict = evaluate_controller(best_adj, types, config,
                                 n_episodes=config["strict_episodes"],
                                 seed=seed * 100000 + 999)
    random_baseline = None if skip_random else evaluate_random_policy(
        n_episodes=config["strict_episodes"], seed=seed * 100000 + 777,
        max_steps=config["max_steps"])

    topo_path = OUT_DIR / f"Ant_v4_best_topology_seed{seed}.npy"
    types_path = OUT_DIR / f"Ant_v4_neuron_types_seed{seed}.npy"
    np.save(topo_path, best_adj)
    np.save(types_path, types)

    result = {
        "env": ENV_NAME,
        "seed": seed,
        "config": config,
        "best_training_fitness": float(best_training_fitness),
        "strict_20ep": strict,
        "random_baseline_20ep": random_baseline,
        "n_connections": int(best_adj.sum()),
        "elapsed_sec": float(time.time() - start),
        "history": {
            "best": best_history,
            "mean": mean_history,
            "connections": conn_history,
        },
        "artifacts": {
            "topology": str(topo_path),
            "types": str(types_path),
            "log": str(log_path),
        },
    }
    result_path = OUT_DIR / f"ant_boundary_seed{seed}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def summarize() -> None:
    result_paths = sorted(OUT_DIR.glob("ant_boundary_seed*.json"))
    if not result_paths:
        return
    rows = []
    for p in result_paths:
        with p.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    strict = np.asarray([r["strict_20ep"]["mean"] for r in rows], dtype=np.float64)
    train = np.asarray([r["best_training_fitness"] for r in rows], dtype=np.float64)
    random_vals = [r.get("random_baseline_20ep") for r in rows if r.get("random_baseline_20ep")]
    summary = {
        "env": ENV_NAME,
        "n_seeds": len(rows),
        "seeds": [r["seed"] for r in rows],
        "best_training_mean": float(train.mean()),
        "best_training_sd": float(train.std(ddof=1)) if len(train) > 1 else 0.0,
        "strict_mean": float(strict.mean()),
        "strict_sd": float(strict.std(ddof=1)) if len(strict) > 1 else 0.0,
        "strict_values": strict.tolist(),
        "random_baseline_values": [r["mean"] for r in random_vals],
        "random_baseline_mean": float(np.mean([r["mean"] for r in random_vals])) if random_vals else None,
        "random_baseline_sd": float(np.std([r["mean"] for r in random_vals], ddof=1)) if len(random_vals) > 1 else 0.0,
        "results": rows,
    }
    with (OUT_DIR / "ant_boundary_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--generations", type=int, default=CONFIG["n_generations"])
    parser.add_argument("--pop-size", type=int, default=CONFIG["pop_size"])
    parser.add_argument("--eval-episodes", type=int, default=CONFIG["n_eval_episodes"])
    parser.add_argument("--strict-episodes", type=int, default=CONFIG["strict_episodes"])
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(CONFIG)
    config["n_generations"] = args.generations
    config["pop_size"] = args.pop_size
    config["n_eval_episodes"] = args.eval_episodes
    config["strict_episodes"] = args.strict_episodes

    for seed in args.seeds:
        result_path = OUT_DIR / f"ant_boundary_seed{seed}.json"
        if args.resume and result_path.exists():
            print(f"Skipping seed={seed}; existing {result_path}", flush=True)
            continue
        run_seed(seed, config, skip_random=args.skip_random)
    summarize()


if __name__ == "__main__":
    main()
