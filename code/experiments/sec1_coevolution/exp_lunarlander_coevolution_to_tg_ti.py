"""
LunarLander TO/TG/TI co-evolution control.

This is the discriminative, non-ceiling reactive-control control requested for
the manuscript revision. It is protocol-matched to the CartPole/MountainCar
TO/TG/TI controls where possible, but uses raw LunarLander return as the
training and strict-evaluation endpoint.

Groups:
  TO: binary adjacency only; global weight_scale fixed at 0.6.
  TG: binary adjacency plus one evolvable global weight_scale.
  TI: binary adjacency plus per-edge independent magnitudes.

Default revision protocol:
  20 seeds per group, pop_size=80, n_elite=16, n_generations=400,
  mutation_rate=0.03, n_eval_episodes=20, strict evaluation=20 episodes.

Primary endpoint:
  strict 20-episode mean raw LunarLander reward per seed.

Pre-specified equivalence rule:
  The primary TOST margin is delta_fraction * abs(TI strict mean), with
  delta_fraction=0.10 by default. Sensitivity margins are also reported.

Outputs:
  code/experiments/results/lunarlander_coevolution/
    result_{MODE}_{SEED}.json
    topology_{MODE}_{SEED}.npz
    lunarlander_coevolution_results.json
    lunarlander_coevolution_summary.csv
    lunarlander_tost_summary.csv
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
DEFAULT_SEEDS = 20
STRICT_EPISODES = 20

# Protocol-matched to the existing TO/TG/TI controls, except for the
# task-specific observation/action dimensions and LunarLander horizon.
BASE_CONFIG: dict[str, Any] = {
    "task": "LunarLander",
    "env_id": "LunarLander-v3",
    "fallback_env_ids": ["LunarLander-v2"],
    "n_total": 128,
    "n_input": 16,  # 8D state x 2 half-wave input channels
    "n_output": 4,
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
    "max_steps": 1000,
    "solve_threshold": 200.0,
    "early_stop": False,
    "ceiling_proxy": 300.0,
    "ceiling_margin": 20.0,
    "delta_fraction": 0.10,
    "alpha": 0.05,
    "equivalence_margin_rule": "delta = delta_fraction * abs(TI strict seed-level mean)",
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
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    return value


def make_env(env_id: str | None = None) -> gym.Env:
    candidates = [env_id or str(BASE_CONFIG["env_id"])]
    candidates.extend(str(x) for x in BASE_CONFIG.get("fallback_env_ids", []))
    errors: list[str] = []
    for candidate in candidates:
        try:
            return gym.make(candidate)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not create LunarLander environment. Install Box2D support with "
        "`pip install swig \"gymnasium[box2d]\"`. Tried: " + " | ".join(errors)
    )


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


def evaluate_lunarlander(
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
    env = make_env(str(config["env_id"]))
    returns: list[float] = []
    steps_list: list[int] = []
    terminated_list: list[bool] = []
    truncated_list: list[bool] = []
    successful_episodes: list[bool] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=eval_seeds[ep])
        controller.reset()
        total_return = 0.0
        steps = 0
        terminated_final = False
        truncated_final = False

        for step in range(int(config["max_steps"])):
            action = int(controller.forward(np.array(obs, dtype=np.float64)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            steps = step + 1
            terminated_final = bool(terminated)
            truncated_final = bool(truncated)
            if terminated or truncated:
                break

        returns.append(total_return)
        steps_list.append(steps)
        terminated_list.append(terminated_final)
        truncated_list.append(truncated_final)
        successful_episodes.append(total_return >= float(config["solve_threshold"]))

    env.close()

    if not return_details:
        return float(np.mean(returns))

    r = np.array(returns, dtype=np.float64)
    steps_arr = np.array(steps_list, dtype=np.float64)
    successes = np.array(successful_episodes, dtype=bool)
    terminated_arr = np.array(terminated_list, dtype=bool)
    truncated_arr = np.array(truncated_list, dtype=bool)
    return {
        "return_mean": float(r.mean()),
        "return_std": float(r.std()),
        "return_min": float(r.min()),
        "return_max": float(r.max()),
        "episode_returns": r.tolist(),
        "steps_mean": float(steps_arr.mean()),
        "steps_std": float(steps_arr.std()),
        "steps_min": int(steps_arr.min()),
        "steps_max": int(steps_arr.max()),
        "terminated_episode_rate": float(terminated_arr.mean()),
        "truncated_episode_rate": float(truncated_arr.mean()),
        "episode_success_count": int(successes.sum()),
        "episode_success_rate": float(successes.mean()),
        "seed_success": bool(r.mean() >= float(config["solve_threshold"])),
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


def population_entry_to_eval(
    entry: Any,
    neuron_types: np.ndarray,
    config: dict[str, Any],
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
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

    tag = f"[LunarLander {mode}|S{seed}]"
    n_hidden = n - int(config["n_input"]) - int(config["n_output"])
    n_inh = int(np.sum(neuron_types == -1.0))
    print(f"{tag} n={n}, hidden={n_hidden}, I={n_inh}, E={n_hidden - n_inh}")

    t0 = time.time()
    for gen in range(int(config["n_generations"])):
        fitnesses = []
        for entry in pop:
            adj, extra = population_entry_to_eval(entry, neuron_types, config, mode)
            fitness = evaluate_lunarlander(
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
        if solved and bool(config.get("early_stop", False)):
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
    strict = evaluate_lunarlander(
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
        "task": "LunarLander",
        "mode": mode,
        "seed": seed,
        "generations": len(best_history),
        "best_training_return": float(best_fitness),
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
        f"strict={strict['return_mean']:.1f}+/-{strict['return_std']:.1f}, "
        f"success={strict['seed_success']}, time={elapsed:.1f}s"
    )
    return result


def welch_anova(groups: dict[str, list[float]]) -> dict[str, Any] | None:
    if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
        return None
    try:
        from scipy import stats
    except ImportError:
        return None

    ordered_names = list(groups)
    ordered = [np.array(groups[m], dtype=np.float64) for m in ordered_names]
    try:
        res = stats.f_oneway(*ordered, equal_var=False)
        method = "Welch ANOVA"
    except TypeError:
        res = stats.f_oneway(*ordered)
        method = "ordinary one-way ANOVA (SciPy lacks equal_var)"
    return {
        "method": method,
        "groups": ordered_names,
        "f_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "df_between": int(len(groups) - 1),
        "df_within": int(sum(len(v) for v in groups.values()) - len(groups)),
    }


def welch_tost(
    x: list[float],
    y: list[float],
    delta: float,
    alpha: float = 0.05,
    label_x: str = "x",
    label_y: str = "y",
) -> dict[str, Any] | None:
    if len(x) < 2 or len(y) < 2:
        return None
    try:
        from scipy import stats
    except ImportError:
        return None

    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    nx = len(xa)
    ny = len(ya)
    mx = float(xa.mean())
    my = float(ya.mean())
    vx = float(xa.var(ddof=1))
    vy = float(ya.var(ddof=1))
    se2 = vx / nx + vy / ny
    if se2 <= 0.0:
        return None
    se = float(np.sqrt(se2))
    num = se2 * se2
    den = ((vx / nx) ** 2) / (nx - 1) + ((vy / ny) ** 2) / (ny - 1)
    df = float(num / den) if den > 0 else float(nx + ny - 2)
    diff = mx - my

    t_lower = (diff + delta) / se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))
    t_upper = (diff - delta) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)
    tcrit = float(stats.t.ppf(1.0 - alpha, df))
    ci_low = float(diff - tcrit * se)
    ci_high = float(diff + tcrit * se)
    min_margin = float(max(abs(ci_low), abs(ci_high)))
    return {
        "comparison": f"{label_x}-{label_y}",
        "label_x": label_x,
        "label_y": label_y,
        "n_x": int(nx),
        "n_y": int(ny),
        "mean_x": mx,
        "mean_y": my,
        "diff": float(diff),
        "delta": float(delta),
        "alpha": float(alpha),
        "se": se,
        "df": df,
        "t_lower": float(t_lower),
        "p_lower": p_lower,
        "t_upper": float(t_upper),
        "p_upper": p_upper,
        "p_tost": float(p_tost),
        "equivalent": bool(p_lower < alpha and p_upper < alpha),
        "ci_level": float(1.0 - 2.0 * alpha),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min_supported_symmetric_margin": min_margin,
    }


def summarize(
    results: list[dict[str, Any]],
    modes: list[str],
    config: dict[str, Any],
    delta_fraction: float,
    alpha: float,
    sensitivity_margins: list[float],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    ceiling_proxy = float(config["ceiling_proxy"])
    ceiling_cutoff = ceiling_proxy - float(config["ceiling_margin"])

    for mode in modes:
        rows = [r for r in results if r["mode"] == mode]
        vals = np.array([r["strict_20ep"]["return_mean"] for r in rows], dtype=np.float64)
        seed_success = np.array([r["strict_20ep"]["seed_success"] for r in rows], dtype=bool)
        ep_success = np.array([r["strict_20ep"]["episode_success_rate"] for r in rows], dtype=np.float64)
        steps = np.array([r["strict_20ep"]["steps_mean"] for r in rows], dtype=np.float64)
        near_ceiling = vals >= ceiling_cutoff if len(vals) else np.array([], dtype=bool)
        summary[mode] = {
            "n": int(len(rows)),
            "strict_return_mean": float(vals.mean()) if len(vals) else None,
            "strict_return_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) else None,
            "strict_return_min": float(vals.min()) if len(vals) else None,
            "strict_return_max": float(vals.max()) if len(vals) else None,
            "seed_success_count": int(seed_success.sum()),
            "seed_success_rate": float(seed_success.mean()) if len(seed_success) else None,
            "episode_success_rate_mean": float(ep_success.mean()) if len(ep_success) else None,
            "steps_mean": float(steps.mean()) if len(steps) else None,
            "near_ceiling_count": int(near_ceiling.sum()),
            "near_ceiling_rate": float(near_ceiling.mean()) if len(near_ceiling) else None,
            "ceiling_proxy": ceiling_proxy,
            "near_ceiling_cutoff": ceiling_cutoff,
        }

    groups = {
        mode: [r["strict_20ep"]["return_mean"] for r in results if r["mode"] == mode]
        for mode in modes
    }

    ti_vals = groups.get("TI", [])
    ti_mean = float(np.mean(ti_vals)) if ti_vals else float("nan")
    primary_delta = float(delta_fraction * abs(ti_mean)) if ti_vals else float("nan")

    tost_primary: list[dict[str, Any]] = []
    for other in ("TO", "TG"):
        if other in groups and "TI" in groups:
            row = welch_tost(groups[other], groups["TI"], primary_delta, alpha, other, "TI")
            if row is not None:
                row["margin_type"] = "primary_fraction_of_TI"
                tost_primary.append(row)
    if "TO" in groups and "TG" in groups:
        row = welch_tost(groups["TO"], groups["TG"], primary_delta, alpha, "TO", "TG")
        if row is not None:
            row["margin_type"] = "primary_fraction_of_TI"
            tost_primary.append(row)

    sensitivity: list[dict[str, Any]] = []
    for delta in sensitivity_margins:
        for a, b in (("TO", "TI"), ("TO", "TG"), ("TG", "TI")):
            if a in groups and b in groups:
                row = welch_tost(groups[a], groups[b], float(delta), alpha, a, b)
                if row is not None:
                    row["margin_type"] = "absolute_sensitivity"
                    sensitivity.append(row)

    return {
        "by_mode": summary,
        "anova": welch_anova(groups),
        "equivalence_plan": {
            "primary_endpoint": "strict 20-episode mean raw LunarLander reward per seed",
            "alpha": float(alpha),
            "primary_margin_rule": str(config["equivalence_margin_rule"]),
            "delta_fraction": float(delta_fraction),
            "ti_reference_mean": ti_mean,
            "primary_delta": primary_delta,
            "sensitivity_margins": [float(x) for x in sensitivity_margins],
        },
        "tost_primary": tost_primary,
        "tost_sensitivity": sensitivity,
        "headroom_diagnostic": {
            "note": "LunarLander is not hard-capped like CartPole; 300 is used only as a practical near-ceiling proxy.",
            "solve_threshold": float(config["solve_threshold"]),
            "ceiling_proxy": ceiling_proxy,
            "near_ceiling_cutoff": ceiling_cutoff,
        },
    }


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
    json_path = os.path.join(results_dir, "lunarlander_coevolution_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_seed": json_ready(results)}, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(results_dir, "lunarlander_coevolution_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "mode",
            "seed",
            "strict_return_mean",
            "strict_return_std",
            "seed_success",
            "episode_success_rate",
            "steps_mean",
            "generations",
            "best_training_return",
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
                    "strict_return_mean": s["return_mean"],
                    "strict_return_std": s["return_std"],
                    "seed_success": s["seed_success"],
                    "episode_success_rate": s["episode_success_rate"],
                    "steps_mean": s["steps_mean"],
                    "generations": r["generations"],
                    "best_training_return": r["best_training_return"],
                    "n_connections": r["n_connections"],
                    "elapsed_sec": r["elapsed_sec"],
                }
            )

    tost_path = os.path.join(results_dir, "lunarlander_tost_summary.csv")
    with open(tost_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "margin_type",
            "comparison",
            "delta",
            "diff",
            "ci_low",
            "ci_high",
            "p_lower",
            "p_upper",
            "p_tost",
            "equivalent",
            "min_supported_symmetric_margin",
            "n_x",
            "n_y",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary.get("tost_primary", []) + summary.get("tost_sensitivity", []):
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"Saved summary JSON: {json_path}")
    print(f"Saved summary CSV:  {csv_path}")
    print(f"Saved TOST CSV:     {tost_path}")


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("LunarLander TO/TG/TI strict 20-episode summary")
    print("=" * 78)
    for mode, row in summary["by_mode"].items():
        if row["n"] == 0:
            print(f"{mode}: n=0")
            continue
        print(
            f"{mode}: n={row['n']}, strict={row['strict_return_mean']:.1f} "
            f"+/- {row['strict_return_std']:.1f}, "
            f"seed_success={row['seed_success_count']}/{row['n']} "
            f"({row['seed_success_rate']:.2%}), "
            f"episode_success={row['episode_success_rate_mean']:.2%}, "
            f"near_ceiling={row['near_ceiling_count']}/{row['n']}"
        )
    anova = summary.get("anova")
    if anova:
        print(
            f"{anova['method']}: F({anova['df_between']},{anova['df_within']}) = "
            f"{anova['f_stat']:.3f}, p = {anova['p_value']:.4f}"
        )
    else:
        print("ANOVA: unavailable (need scipy and at least two seeds per mode).")

    plan = summary["equivalence_plan"]
    print(
        "Primary TOST margin: "
        f"delta={plan['primary_delta']:.3f} "
        f"({plan['delta_fraction']:.1%} x abs(TI mean={plan['ti_reference_mean']:.3f}))"
    )
    if summary.get("tost_primary"):
        for row in summary["tost_primary"]:
            print(
                f"TOST {row['comparison']}: diff={row['diff']:.2f}, "
                f"90% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}], "
                f"pTOST={row['p_tost']:.4f}, equivalent={row['equivalent']}, "
                f"min_margin={row['min_supported_symmetric_margin']:.2f}"
            )
    else:
        print("TOST: unavailable until at least two seeds per compared group exist.")


def parse_margins(text: str) -> list[float]:
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LunarLander TO/TG/TI co-evolution control")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed", type=int, help="Run a single seed for each selected mode")
    parser.add_argument("--resume", action="store_true", help="Skip existing result_{MODE}_{SEED}.json")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing results")
    parser.add_argument("--early-stop", action="store_true", help="Stop a seed once training mean reaches 200")
    parser.add_argument("--env-id", default=str(BASE_CONFIG["env_id"]))
    parser.add_argument("--delta-fraction", type=float, default=float(BASE_CONFIG["delta_fraction"]))
    parser.add_argument("--alpha", type=float, default=float(BASE_CONFIG["alpha"]))
    parser.add_argument("--sensitivity-margins", default="25,50,75")
    parser.add_argument("--n-generations", type=int, help="Override generation count")
    parser.add_argument("--pop-size", type=int, help="Override population size")
    parser.add_argument("--n-eval-episodes", type=int, help="Override training evaluation episodes")
    parser.add_argument("--strict-episodes", type=int, help="Override strict evaluation episodes")
    parser.add_argument("--max-steps", type=int, help="Override max steps per episode")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(SCRIPT_DIR, "..", "results", "lunarlander_coevolution"),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Debug only: tiny budget, writes to lunarlander_coevolution_smoke.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(BASE_CONFIG)
    config["env_id"] = args.env_id
    config["delta_fraction"] = float(args.delta_fraction)
    config["alpha"] = float(args.alpha)
    config["early_stop"] = bool(args.early_stop)
    if args.n_generations is not None:
        config["n_generations"] = int(args.n_generations)
    if args.pop_size is not None:
        config["pop_size"] = int(args.pop_size)
    if args.n_eval_episodes is not None:
        config["n_eval_episodes"] = int(args.n_eval_episodes)
    if args.max_steps is not None:
        config["max_steps"] = int(args.max_steps)

    results_dir = os.path.abspath(args.results_dir)

    global STRICT_EPISODES
    if args.strict_episodes is not None:
        STRICT_EPISODES = int(args.strict_episodes)

    if args.smoke_test:
        config.update(
            {
                "pop_size": 4,
                "n_elite": 1,
                "n_generations": 1,
                "n_eval_episodes": 1,
                "max_steps": min(int(config["max_steps"]), 200),
            }
        )
        STRICT_EPISODES = 1
        results_dir = os.path.abspath(
            os.path.join(SCRIPT_DIR, "..", "results", "lunarlander_coevolution_smoke")
        )
        print("SMOKE TEST MODE: results are not paper-valid.")

    sensitivity_margins = parse_margins(str(args.sensitivity_margins))
    seed_list = [args.seed] if args.seed is not None else list(range(args.seeds))

    print("=" * 78)
    print("LunarLander TO/TG/TI co-evolution control")
    print(f"Modes: {args.modes}")
    print(f"Seeds: {seed_list}")
    print(f"Results: {results_dir}")
    print(
        "Key protocol: "
        f"pop={config['pop_size']}, elite={config['n_elite']}, "
        f"generations={config['n_generations']}, "
        f"train_eval={config['n_eval_episodes']}, strict={STRICT_EPISODES}, "
        f"max_steps={config['max_steps']}, early_stop={config['early_stop']}"
    )
    print(
        "Equivalence plan: "
        f"delta={config['delta_fraction']:.1%} x abs(TI strict mean), "
        f"alpha={config['alpha']}, sensitivity={sensitivity_margins}"
    )
    print("=" * 78)

    # Fail fast before launching a long batch if Box2D support is missing.
    if not args.aggregate_only:
        env = make_env(str(config["env_id"]))
        print(f"Environment OK: {config['env_id']} obs={env.observation_space} action={env.action_space}")
        env.close()

    results: list[dict[str, Any]] = []
    for mode in args.modes:
        for seed in seed_list:
            existing = load_existing(mode, seed, results_dir)
            if existing is not None:
                if args.resume or args.aggregate_only:
                    print(f"[LunarLander {mode}|S{seed}] Existing result found, loading.")
                    results.append(existing)
                    continue
                raise FileExistsError(
                    f"Refusing to overwrite existing result without --resume: {mode} seed {seed}"
                )
            if args.aggregate_only:
                print(f"[LunarLander {mode}|S{seed}] Missing result, skipping aggregate input.")
                continue
            result = evolve_one(mode, int(seed), config)
            result = save_result(result, results_dir)
            results.append(result)

    summary = summarize(
        results,
        list(args.modes),
        config,
        delta_fraction=float(args.delta_fraction),
        alpha=float(args.alpha),
        sensitivity_margins=sensitivity_margins,
    )
    print_summary(summary)
    write_summary_files(results, summary, results_dir)


if __name__ == "__main__":
    main()
