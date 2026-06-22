"""
MountainCar TO/TG/TI co-evolution control.

This experiment is intentionally protocol-matched to the CartPole TO/TG/TI
control in this directory. The task-specific parts are the environment,
observation/action dimensions, episode length, and MountainCar shaping
fitness; the GA budget and the three treatment definitions are kept aligned.

Groups:
  TO: binary adjacency only; global weight_scale fixed at 0.6.
  TG: binary adjacency plus one evolvable global weight_scale.
  TI: binary adjacency plus per-edge independent magnitudes.

Default paper protocol:
  10 seeds per group, pop_size=80, n_elite=16, n_generations=400,
  mutation_rate=0.03, n_eval_episodes=20, strict evaluation=20 episodes.

Outputs:
  code/experiments/results/mountaincar_coevolution/
    result_{MODE}_{SEED}.json
    topology_{MODE}_{SEED}.npz
    mountaincar_coevolution_results.json
    mountaincar_coevolution_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, CODE_DIR)

from src.snn import SNNController, WeightedSNNController  # noqa: E402


MODES = ("TO", "TG", "TI")
DEFAULT_SEEDS = 10
STRICT_EPISODES = 20

# Protocol-matched to CartPole TO/TG/TI, except task-specific dimensions.
BASE_CONFIG: dict[str, Any] = {
    "task": "MountainCar",
    "env_id": "MountainCar-v0",
    "n_total": 128,
    "n_input": 4,
    "n_output": 3,
    "tau": 5.0,
    "threshold": 0.8,
    "T_steps": 20,
    "weight_scale": 0.6,
    "sparsity": 0.10,
    "pop_size": 80,
    "n_generations": 400,
    "n_elite": 16,
    "mutation_rate": 0.03,
    "n_eval_episodes": 20,
    "stagnation_limit": 10,
    "mut_rate_cap": 0.10,
    "tournament_size": 3,
    "crossover_rate": 0.5,
    "max_steps": 200,
    # Same MountainCar shaping threshold used in the cross-task existence run.
    "solve_threshold": 650.0,
    "target_sparsity": 0.15,
    "density_penalty": 200.0,
    # MountainCar observations are much smaller than CartPole; this preserves
    # the existing MountainCar task protocol while keeping all groups identical.
    "input_pre_scale": 4.0,
}


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def random_topology(n: int, sparsity: float) -> np.ndarray:
    adj = (np.random.rand(n, n) < sparsity).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def mutate(adj: np.ndarray, rate: float) -> np.ndarray:
    child = adj.copy()
    mask = np.random.rand(*child.shape) < rate
    child[mask] = 1.0 - child[mask]
    np.fill_diagonal(child, 0.0)
    return child


def crossover_topology(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    mask = np.random.rand(*p1.shape) < 0.5
    child = np.where(mask, p1, p2).astype(np.float64)
    np.fill_diagonal(child, 0.0)
    return child


def tournament_select(pop: list[Any], fitnesses: np.ndarray, size: int = 3) -> Any:
    candidates = np.random.choice(len(pop), size=min(size, len(pop)), replace=False)
    return pop[candidates[np.argmax(fitnesses[candidates])]]


def init_neuron_types(config: dict[str, Any]) -> np.ndarray:
    n = int(config["n_total"])
    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(int(config["n_input"]), n - int(config["n_output"]))
    n_inh = int(len(hidden_range) * 0.3)
    inh_idx = np.random.choice(hidden_range, n_inh, replace=False)
    neuron_types[inh_idx] = -1.0
    return neuron_types


def build_controller(
    config: dict[str, Any],
    adj: np.ndarray,
    neuron_types: np.ndarray,
    mode: str,
    extra: dict[str, Any],
) -> SNNController:
    if mode == "TI":
        return WeightedSNNController(config, adj, neuron_types, extra["weight_matrix"])
    if mode == "TG":
        ws = float(extra["weight_scale"])
        W = adj * neuron_types[:, np.newaxis] * ws
        return WeightedSNNController(config, adj, neuron_types, W)
    return SNNController(config, adj, neuron_types)


def mountaincar_shaped_fitness(max_pos: float, adj: np.ndarray, config: dict[str, Any]) -> float:
    fitness = (max_pos + 1.2) / 1.8 * 1000.0
    if max_pos >= 0.3:
        fitness += 300.0
    if max_pos >= 0.45:
        fitness += 700.0
    if max_pos >= 0.5:
        fitness += 1000.0

    n = adj.shape[0]
    density = adj.sum() / (n * (n - 1))
    over = max(0.0, density - float(config.get("target_sparsity", 0.15)))
    return float(fitness - over * float(config.get("density_penalty", 200.0)))


def evaluate_mountaincar(
    adj: np.ndarray,
    neuron_types: np.ndarray,
    config: dict[str, Any],
    mode: str,
    extra: dict[str, Any],
    eval_seeds: list[int],
    n_episodes: int,
    return_details: bool = False,
) -> float | dict[str, Any]:
    controller = build_controller(config, adj, neuron_types, mode, extra)
    env = gym.make(str(config["env_id"]))
    fitnesses: list[float] = []
    env_returns: list[float] = []
    max_positions: list[float] = []
    steps_list: list[int] = []
    reached_goal: list[bool] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=eval_seeds[ep])
        controller.reset()
        max_pos = float(obs[0])
        total_return = 0.0
        steps = 0
        terminated_final = False

        for step in range(int(config["max_steps"])):
            scaled_obs = np.array(obs, dtype=np.float64) * float(config["input_pre_scale"])
            action = int(controller.forward(scaled_obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            max_pos = max(max_pos, float(obs[0]))
            steps = step + 1
            terminated_final = bool(terminated)
            if terminated or truncated:
                break

        fitnesses.append(mountaincar_shaped_fitness(max_pos, adj, config))
        env_returns.append(total_return)
        max_positions.append(max_pos)
        steps_list.append(steps)
        reached_goal.append(terminated_final and max_pos >= 0.5)

    env.close()

    if not return_details:
        return float(np.mean(fitnesses))

    f = np.array(fitnesses, dtype=np.float64)
    r = np.array(env_returns, dtype=np.float64)
    m = np.array(max_positions, dtype=np.float64)
    steps_arr = np.array(steps_list, dtype=np.float64)
    goals = np.array(reached_goal, dtype=bool)
    return {
        "fitness_mean": float(f.mean()),
        "fitness_std": float(f.std()),
        "fitness_min": float(f.min()),
        "fitness_max": float(f.max()),
        "env_return_mean": float(r.mean()),
        "env_return_std": float(r.std()),
        "max_position_mean": float(m.mean()),
        "max_position_std": float(m.std()),
        "max_position_min": float(m.min()),
        "max_position_max": float(m.max()),
        "steps_mean": float(steps_arr.mean()),
        "goal_episodes": int(goals.sum()),
        "goal_episode_rate": float(goals.mean()),
        "seed_success": bool(f.mean() >= float(config["solve_threshold"])),
        "n_episodes": int(n_episodes),
        "threshold": float(config["solve_threshold"]),
    }


def initialize_population(config: dict[str, Any], mode: str) -> list[Any]:
    n = int(config["n_total"])
    sparsity = float(config["sparsity"])
    if mode == "TO":
        return [random_topology(n, sparsity) for _ in range(int(config["pop_size"]))]
    if mode == "TG":
        return [
            (random_topology(n, sparsity), float(config["weight_scale"]))
            for _ in range(int(config["pop_size"]))
        ]
    if mode == "TI":
        adjs = [random_topology(n, sparsity) for _ in range(int(config["pop_size"]))]
        return [(adj, float(config["weight_scale"]) * adj) for adj in adjs]
    raise ValueError(f"Unknown mode: {mode}")


def population_entry_to_eval(entry: Any, neuron_types: np.ndarray, config: dict[str, Any], mode: str):
    if mode == "TO":
        return entry, {"weight_scale": float(config["weight_scale"])}
    if mode == "TG":
        adj, ws = entry
        return adj, {"weight_scale": float(ws)}
    adj, mag = entry
    return adj, {"weight_matrix": adj * neuron_types[:, np.newaxis] * mag}


def get_adj(entry: Any, mode: str) -> np.ndarray:
    return entry if mode == "TO" else entry[0]


def evolve_one(mode: str, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    np.random.seed(seed)
    n = int(config["n_total"])
    eval_seeds = [seed * 1000 + ep for ep in range(int(config["n_eval_episodes"]))]
    neuron_types = init_neuron_types(config)
    pop = initialize_population(config, mode)

    best_history: list[float] = []
    mean_history: list[float] = []
    best_fitness = -np.inf
    best_adj: np.ndarray | None = None
    best_extra: dict[str, Any] | None = None

    base_mut_rate = float(config["mutation_rate"])
    current_mut_rate = base_mut_rate
    stagnation = 0
    stagnation_limit = int(config["stagnation_limit"])
    mut_rate_cap = float(config["mut_rate_cap"])
    tournament_size = int(config["tournament_size"])
    crossover_rate = float(config["crossover_rate"])

    tag = f"[MountainCar {mode}|S{seed}]"
    n_hidden = n - int(config["n_input"]) - int(config["n_output"])
    n_inh = int(np.sum(neuron_types == -1.0))
    print(f"{tag} n={n}, hidden={n_hidden}, I={n_inh}, E={n_hidden - n_inh}")

    t0 = time.time()
    for gen in range(int(config["n_generations"])):
        fitnesses = []
        for entry in pop:
            adj, extra = population_entry_to_eval(entry, neuron_types, config, mode)
            fitness = evaluate_mountaincar(
                adj,
                neuron_types,
                config,
                mode,
                extra,
                eval_seeds=eval_seeds,
                n_episodes=int(config["n_eval_episodes"]),
            )
            fitnesses.append(float(fitness))
        fitnesses_arr = np.array(fitnesses, dtype=np.float64)
        order = np.argsort(fitnesses_arr)[::-1]
        gen_best = float(fitnesses_arr[order[0]])
        gen_mean = float(fitnesses_arr.mean())
        best_entry = pop[order[0]]
        best_conn = int(get_adj(best_entry, mode).sum())

        best_history.append(gen_best)
        mean_history.append(gen_mean)

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj, best_extra = population_entry_to_eval(best_entry, neuron_types, config, mode)
            best_adj = best_adj.copy()
            if mode == "TI":
                best_extra = {"weight_matrix": best_extra["weight_matrix"].copy()}
            stagnation = 0
            current_mut_rate = base_mut_rate
        else:
            stagnation += 1
            if stagnation >= stagnation_limit:
                current_mut_rate = min(current_mut_rate * 1.5, mut_rate_cap)

        solved = gen_best >= float(config["solve_threshold"])
        if gen % 20 == 0 or solved:
            print(
                f"{tag} Gen {gen + 1:03d}: best={gen_best:.1f} "
                f"mean={gen_mean:.1f} conns={best_conn} mut={current_mut_rate:.3f}"
                + (" *** SOLVED ***" if solved else "")
            )
        if solved:
            break

        elite = [pop[i] for i in order[: int(config["n_elite"])]]
        new_pop = list(elite)
        while len(new_pop) < int(config["pop_size"]):
            parent = tournament_select(pop, fitnesses_arr, tournament_size)

            if mode == "TO":
                if np.random.rand() < crossover_rate:
                    parent2 = tournament_select(pop, fitnesses_arr, tournament_size)
                    child = crossover_topology(parent, parent2)
                else:
                    child = parent.copy()
                new_pop.append(mutate(child, current_mut_rate))

            elif mode == "TG":
                p_adj, p_ws = parent
                if np.random.rand() < crossover_rate:
                    p2_adj, p2_ws = tournament_select(pop, fitnesses_arr, tournament_size)
                    child_adj = crossover_topology(p_adj, p2_adj)
                    child_ws = (float(p_ws) + float(p2_ws)) / 2.0
                else:
                    child_adj = p_adj.copy()
                    child_ws = float(p_ws)
                child_adj = mutate(child_adj, current_mut_rate)
                child_ws = float(np.clip(child_ws + np.random.normal(0, 0.1), 0.1, 2.0))
                new_pop.append((child_adj, child_ws))

            else:
                p_adj, p_mag = parent
                if np.random.rand() < crossover_rate:
                    p2_adj, p2_mag = tournament_select(pop, fitnesses_arr, tournament_size)
                    child_adj = crossover_topology(p_adj, p2_adj)
                    child_mag = float(config["weight_scale"]) * np.ones((n, n))
                    both = (p_adj == 1) & (p2_adj == 1) & (child_adj == 1)
                    only1 = (p_adj == 1) & (p2_adj == 0) & (child_adj == 1)
                    only2 = (p_adj == 0) & (p2_adj == 1) & (child_adj == 1)
                    child_mag[both] = (p_mag[both] + p2_mag[both]) / 2.0
                    child_mag[only1] = p_mag[only1]
                    child_mag[only2] = p2_mag[only2]
                    child_mag[child_adj == 0] = 0.0
                else:
                    child_adj = p_adj.copy()
                    child_mag = p_mag.copy()

                old_adj = child_adj.copy()
                child_adj = mutate(child_adj, current_mut_rate)
                new_mask = (old_adj == 0) & (child_adj == 1)
                del_mask = (old_adj == 1) & (child_adj == 0)
                child_mag[new_mask] = float(config["weight_scale"])
                child_mag[del_mask] = 0.0

                active_mask = child_adj == 1
                mutate_mask = active_mask & (np.random.rand(n, n) < 0.1)
                noise = np.random.normal(0, 0.05, size=(n, n))
                child_mag[mutate_mask] += noise[mutate_mask]
                child_mag = np.clip(child_mag, 0.1, 1.5)
                child_mag[child_adj == 0] = 0.0
                new_pop.append((child_adj, child_mag))

        pop = new_pop

    elapsed = time.time() - t0
    if best_adj is None or best_extra is None:
        raise RuntimeError("Evolution produced no best individual.")

    strict_seeds = [10000 + ep for ep in range(STRICT_EPISODES)]
    strict = evaluate_mountaincar(
        best_adj,
        neuron_types,
        config,
        mode,
        best_extra,
        eval_seeds=strict_seeds,
        n_episodes=STRICT_EPISODES,
        return_details=True,
    )
    assert isinstance(strict, dict)

    result = {
        "task": "MountainCar",
        "mode": mode,
        "seed": seed,
        "generations": len(best_history),
        "best_training_fitness": float(best_fitness),
        "strict_20ep": strict,
        "solved_by_training_threshold": bool(best_fitness >= float(config["solve_threshold"])),
        "solved_by_strict_threshold": bool(strict["seed_success"]),
        "n_connections": int(best_adj.sum()),
        "elapsed_sec": float(elapsed),
        "config": json_ready(config),
        "extra": {
            "weight_scale": float(best_extra.get("weight_scale", config["weight_scale"]))
            if mode != "TI"
            else None,
            "continuous_evolved_dof": 0 if mode == "TO" else (1 if mode == "TG" else n * n),
        },
        "history": {
            "best": best_history,
            "mean": mean_history,
        },
    }

    result["_arrays"] = {
        "adj": best_adj,
        "types": neuron_types,
        "weight_matrix": best_extra.get("weight_matrix") if mode == "TI" else None,
    }
    print(
        f"{tag} Done: gen={len(best_history)}, train_best={best_fitness:.1f}, "
        f"strict={strict['fitness_mean']:.1f}+/-{strict['fitness_std']:.1f}, "
        f"success={strict['seed_success']}, time={elapsed:.1f}s"
    )
    return result


def anova_test(groups: dict[str, list[float]]) -> dict[str, Any] | None:
    if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
        return None
    try:
        from scipy import stats
    except ImportError:
        return None

    ordered = [np.array(groups[m], dtype=np.float64) for m in groups]
    f_stat, p_value = stats.f_oneway(*ordered)
    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "df_between": int(len(groups) - 1),
        "df_within": int(sum(len(v) for v in groups.values()) - len(groups)),
    }


def summarize(results: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in modes:
        rows = [r for r in results if r["mode"] == mode]
        vals = np.array([r["strict_20ep"]["fitness_mean"] for r in rows], dtype=np.float64)
        seed_success = np.array([r["strict_20ep"]["seed_success"] for r in rows], dtype=bool)
        goal_rates = np.array([r["strict_20ep"]["goal_episode_rate"] for r in rows], dtype=np.float64)
        max_pos = np.array([r["strict_20ep"]["max_position_mean"] for r in rows], dtype=np.float64)
        summary[mode] = {
            "n": int(len(rows)),
            "strict_fitness_mean": float(vals.mean()) if len(vals) else None,
            "strict_fitness_std": float(vals.std()) if len(vals) else None,
            "strict_fitness_min": float(vals.min()) if len(vals) else None,
            "strict_fitness_max": float(vals.max()) if len(vals) else None,
            "seed_success_count": int(seed_success.sum()),
            "seed_success_rate": float(seed_success.mean()) if len(seed_success) else None,
            "episode_goal_rate_mean": float(goal_rates.mean()) if len(goal_rates) else None,
            "max_position_mean": float(max_pos.mean()) if len(max_pos) else None,
        }
    groups = {
        mode: [r["strict_20ep"]["fitness_mean"] for r in results if r["mode"] == mode]
        for mode in modes
    }
    return {"by_mode": summary, "anova": anova_test(groups)}


def save_result(result: dict[str, Any], results_dir: str) -> dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)
    arrays = result.pop("_arrays")
    mode = result["mode"]
    seed = result["seed"]
    npz_path = os.path.join(results_dir, f"topology_{mode}_{seed}.npz")
    np.savez(
        npz_path,
        adj=arrays["adj"],
        types=arrays["types"],
        weight_matrix=arrays["weight_matrix"] if arrays["weight_matrix"] is not None else np.array([]),
        best_hist=np.array(result["history"]["best"], dtype=np.float64),
        mean_hist=np.array(result["history"]["mean"], dtype=np.float64),
        mode=mode,
        seed=seed,
    )
    result["artifact"] = os.path.relpath(npz_path, results_dir)
    out_path = os.path.join(results_dir, f"result_{mode}_{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, indent=2, ensure_ascii=False)
    return result


def load_existing(mode: str, seed: int, results_dir: str) -> dict[str, Any] | None:
    path = os.path.join(results_dir, f"result_{mode}_{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_summary_files(results: list[dict[str, Any]], summary: dict[str, Any], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "mountaincar_coevolution_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_seed": json_ready(results)}, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(results_dir, "mountaincar_coevolution_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "mode",
            "seed",
            "strict_fitness_mean",
            "strict_fitness_std",
            "seed_success",
            "goal_episode_rate",
            "max_position_mean",
            "env_return_mean",
            "generations",
            "best_training_fitness",
            "n_connections",
            "elapsed_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["mode"], x["seed"])):
            s = r["strict_20ep"]
            writer.writerow(
                {
                    "mode": r["mode"],
                    "seed": r["seed"],
                    "strict_fitness_mean": s["fitness_mean"],
                    "strict_fitness_std": s["fitness_std"],
                    "seed_success": s["seed_success"],
                    "goal_episode_rate": s["goal_episode_rate"],
                    "max_position_mean": s["max_position_mean"],
                    "env_return_mean": s["env_return_mean"],
                    "generations": r["generations"],
                    "best_training_fitness": r["best_training_fitness"],
                    "n_connections": r["n_connections"],
                    "elapsed_sec": r["elapsed_sec"],
                }
            )
    print(f"Saved summary JSON: {json_path}")
    print(f"Saved summary CSV:  {csv_path}")


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("MountainCar TO/TG/TI strict 20-episode summary")
    print("=" * 78)
    for mode, row in summary["by_mode"].items():
        print(
            f"{mode}: n={row['n']}, strict={row['strict_fitness_mean']:.1f} "
            f"+/- {row['strict_fitness_std']:.1f}, "
            f"seed_success={row['seed_success_count']}/{row['n']} "
            f"({row['seed_success_rate']:.2%}), "
            f"goal_ep_rate={row['episode_goal_rate_mean']:.2%}, "
            f"max_pos={row['max_position_mean']:.3f}"
        )
    anova = summary.get("anova")
    if anova:
        print(
            f"ANOVA: F({anova['df_between']},{anova['df_within']}) = "
            f"{anova['f_stat']:.3f}, p = {anova['p_value']:.4f}"
        )
    else:
        print("ANOVA: unavailable (need scipy and at least two seeds per mode).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MountainCar TO/TG/TI co-evolution control")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed", type=int, help="Run a single seed for each selected mode")
    parser.add_argument("--resume", action="store_true", help="Skip existing result_{MODE}_{SEED}.json")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(SCRIPT_DIR, "..", "results", "mountaincar_coevolution"),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Debug only: tiny budget, writes to mountaincar_coevolution_smoke.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(BASE_CONFIG)
    results_dir = os.path.abspath(args.results_dir)

    if args.smoke_test:
        config.update(
            {
                "pop_size": 6,
                "n_elite": 2,
                "n_generations": 2,
                "n_eval_episodes": 2,
            }
        )
        global STRICT_EPISODES
        STRICT_EPISODES = 2
        results_dir = os.path.abspath(
            os.path.join(SCRIPT_DIR, "..", "results", "mountaincar_coevolution_smoke")
        )
        print("SMOKE TEST MODE: results are not paper-valid.")

    seed_list = [args.seed] if args.seed is not None else list(range(args.seeds))

    print("=" * 78)
    print("MountainCar TO/TG/TI co-evolution control")
    print(f"Modes: {args.modes}")
    print(f"Seeds: {seed_list}")
    print(f"Results: {results_dir}")
    print(
        "Key protocol: "
        f"pop={config['pop_size']}, elite={config['n_elite']}, "
        f"generations={config['n_generations']}, "
        f"eval={config['n_eval_episodes']}, strict={STRICT_EPISODES}"
    )
    print("=" * 78)

    results: list[dict[str, Any]] = []
    for mode in args.modes:
        for seed in seed_list:
            if args.resume:
                existing = load_existing(mode, seed, results_dir)
                if existing is not None:
                    print(f"[MountainCar {mode}|S{seed}] Existing result found, skipping.")
                    results.append(existing)
                    continue
            result = evolve_one(mode, int(seed), config)
            result = save_result(result, results_dir)
            results.append(result)

    summary = summarize(results, args.modes)
    print_summary(summary)
    write_summary_files(results, summary, results_dir)


if __name__ == "__main__":
    main()
