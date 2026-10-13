"""
固定边预算拓扑 GA（Phase 2）

设计文档 §6：
  - 使用 (mu + lambda) 或 steady-state 精英策略，避免整代大幅替换
  - 同一代所有个体使用相同的环境 episode seeds；训练 seed batch 按固定日程轮换
  - 使用固定总环境评估预算，不按训练 fitness 提前停止
  - mutation-only，取消逐边 uniform crossover

关键区别于旧代码：
  1. 无 early stopping（旧代码 `if solved: break`）
  2. mutation-only，无 crossover（旧代码有 crossover_rate=0.5）
  3. 固定边数（旧代码用 sparsity 比例 + 逐位翻转）
  4. common random numbers（同一代同 seed）
  5. 固定评估预算（按真实环境 episodes 计数）
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.connection_mask import build_allowed_mask
from ..core.controller import UnifiedSNNController
from ..core.encoder import ObservationEncoder
from ..core.protocol import ControllerProtocol, TaskSpec
from ..core.scaffold import NeuronScaffold
from ..core.weight_construction import F0
from .topology_ops import (
    init_population,
    mutate_edge_swap,
    sample_k,
    tournament_select,
    verify_topology_invariants,
)


@dataclass
class GAConfig:
    """拓扑 GA 的配置（pilot 阶段调整，确认性阶段冻结）。"""

    pop_size: int = 50
    n_elite: int = 10                 # (mu + lambda) 的 mu
    tournament_size: int = 3
    lam_poisson: float = 2.0          # k = 1 + Poisson(lam)
    n_eval_episodes: int = 8          # 每个 genome 的训练评估 episode 数
    total_eval_budget: int = 200_000  # 总环境评估 episode 数（固定预算）
    seed_rotation_batch: int = 5      # 每 batch 代轮换一次训练 seeds
    # v3 修复：候选档案 + 固定 selection seeds
    candidate_archive_size: int = 30  # 每代保留的候选数量
    selection_seeds: tuple = ()       # 固定 selection seeds（空=用全部 train_seeds）

    def __post_init__(self):
        if self.n_elite > self.pop_size:
            raise ValueError(f"n_elite {self.n_elite} > pop_size {self.pop_size}")
        if self.n_elite < 1:
            raise ValueError("n_elite must be >= 1")


@dataclass
class GACheckpoint:
    """GA 的 checkpoint 状态，支持 resume。"""

    generation: int
    evaluations_done: int
    best_fitness: float
    best_adjacency: list  # nested list (JSON 可序列化)
    population: list       # list of nested lists
    fitness_history: list  # [(gen, best, mean, median), ...]
    elapsed_sec: float
    seed_batch_index: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f)

    @staticmethod
    def load(path: str | Path) -> GACheckpoint:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return GACheckpoint(**data)


class TopologyGA:
    """固定边预算的 mutation-only (mu + lambda) 拓扑 GA。

    用法：
        ga = TopologyGA(scaffold, task, encoder, ctrl_proto, ga_config, train_seeds)
        result = ga.run(evolution_seed=0, output_dir=..., resume=...)
    """

    def __init__(
        self,
        scaffold: NeuronScaffold,
        task: TaskSpec,
        encoder: ObservationEncoder,
        ctrl_proto: ControllerProtocol,
        ga_config: GAConfig,
        train_seeds: list[int],
        allowed_mask: np.ndarray | None = None,
    ):
        self.scaffold = scaffold
        self.task = task
        self.encoder = encoder
        self.ctrl_proto = ctrl_proto
        self.config = ga_config
        self.train_seeds = list(train_seeds)

        if allowed_mask is None:
            allowed_mask = build_allowed_mask(
                ctrl_proto.n_total, task.n_input, task.n_output
            )
        self.allowed_mask = allowed_mask

        self.edge_count = int(
            0.10 * ctrl_proto.n_total * (ctrl_proto.n_total - 1)
        )

        # 运行时状态
        self.evaluations_done = 0
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_adjacency = None
        self.fitness_history = []
        # v3: 候选档案——每代 top-N 个体去重后保留
        self.candidate_archive: list[np.ndarray] = []
        self.candidate_hashes: set = set()

    def _current_train_seeds(self) -> list[int]:
        """按固定日程轮换训练 seed batch。

        设计文档 §6：训练 seed batch 按固定日程轮换。
        每 seed_rotation_batch 代轮换到下一个 seed 子集。
        """
        batch_idx = self.generation // self.config.seed_rotation_batch
        n = self.config.n_eval_episodes
        # 确定性映射：从 train_seeds 中选 n 个，按 batch_idx 轮换
        total = len(self.train_seeds)
        start = (batch_idx * n) % total
        if start + n <= total:
            return self.train_seeds[start:start + n]
        else:
            # wrap around
            part1 = self.train_seeds[start:]
            part2 = self.train_seeds[:n - len(part1)]
            return part1 + part2

    def _evaluate_genome(
        self,
        adjacency: np.ndarray,
        eval_seeds: list[int],
        env_factory: Callable,
    ) -> float:
        """评估单个拓扑，返回平均 raw return。"""
        import gymnasium as gym

        ctrl = UnifiedSNNController(
            self.scaffold, adjacency, self.encoder, self.ctrl_proto, condition=F0
        )
        env = env_factory()
        rewards = []
        for seed in eval_seeds:
            result = ctrl.rollout(env, self.task.max_steps, seed=seed)
            rewards.append(result["total_reward"])
        env.close()
        return float(np.mean(rewards))

    def run(
        self,
        evolution_seed: int,
        env_factory: Callable,
        output_dir: str | Path | None = None,
        resume_ckpt: GACheckpoint | None = None,
        verbose: bool = True,
    ) -> dict:
        """运行 GA 直到总评估预算耗尽。

        Args:
            evolution_seed: 进化随机种子
            env_factory:    返回新 env 实例的 callable
            output_dir:     输出目录（保存 checkpoint 和结果）
            resume_ckpt:    恢复点（None 则从头开始）
            verbose:        是否打印进度

        Returns:
            {
                'best_adjacency': np.ndarray,
                'best_fitness': float,
                'fitness_history': list,
                'n_generations': int,
                'evaluations_done': int,
                'elapsed_sec': float,
            }
        """
        rng = np.random.RandomState(evolution_seed)
        t0 = time.time()

        # 恢复或初始化
        if resume_ckpt is not None:
            self.generation = resume_ckpt.generation
            self.evaluations_done = resume_ckpt.evaluations_done
            self.best_fitness = resume_ckpt.best_fitness
            self.best_adjacency = np.array(resume_ckpt.best_adjacency)
            population = [np.array(adj) for adj in resume_ckpt.population]
            self.fitness_history = resume_ckpt.fitness_history
            self._seed_batch_index = resume_ckpt.seed_batch_index
            if verbose:
                print(f"[GA resume] gen={self.generation}, evals={self.evaluations_done}, "
                      f"best={self.best_fitness:.1f}")
        else:
            population = init_population(
                self.config.pop_size, self.allowed_mask, self.edge_count, rng
            )
            self._seed_batch_index = 0

        # 主循环：按固定总评估预算运行
        while self.evaluations_done < self.config.total_eval_budget:
            self.generation += 1

            # 当前代的训练 seeds（common random numbers）
            current_seeds = self._current_train_seeds()

            # 评估整代
            fitnesses = np.array([
                self._evaluate_genome(adj, current_seeds, env_factory)
                for adj in population
            ])
            self.evaluations_done += len(population) * len(current_seeds)

            # 排序
            order = np.argsort(fitnesses)[::-1]
            gen_best = float(fitnesses[order[0]])
            gen_mean = float(fitnesses.mean())
            gen_median = float(np.median(fitnesses))

            # 更新最优（仅用于日志显示，跨代不可比）
            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                self.best_adjacency = population[order[0]].copy()

            # v3: 维护候选档案——每代只加当前代 best（去重）
            # 不加 top-30，否则档案膨胀到数千个，selection 阶段太慢
            best_candidate = population[order[0]]
            h = hash(best_candidate.tobytes())
            if h not in self.candidate_hashes:
                self.candidate_hashes.add(h)
                self.candidate_archive.append(best_candidate.copy())
                # 限制档案大小：超过上限时移除最早的
                max_archive = self.config.candidate_archive_size
                if len(self.candidate_archive) > max_archive:
                    removed = self.candidate_archive.pop(0)
                    self.candidate_hashes.discard(hash(removed.tobytes()))

            self.fitness_history.append({
                "gen": self.generation,
                "best": gen_best,
                "mean": gen_mean,
                "median": gen_median,
                "evals_done": self.evaluations_done,
            })

            if verbose and (self.generation % 10 == 0 or self.generation <= 3):
                print(
                    f"[GA] task={self.task.name} seed={evolution_seed} "
                    f"gen={self.generation} best={gen_best:.1f} "
                    f"mean={gen_mean:.1f} evals={self.evaluations_done}/"
                    f"{self.config.total_eval_budget}"
                )

            # (mu + lambda) 精英策略
            elite = [population[i].copy() for i in order[:self.config.n_elite]]
            new_pop = list(elite)

            while len(new_pop) < self.config.pop_size:
                parent = tournament_select(
                    population, fitnesses, self.config.tournament_size, rng
                )
                child = mutate_edge_swap(
                    parent, self.allowed_mask,
                    k=None, rng=rng, lam=self.config.lam_poisson,
                )
                new_pop.append(child)

            population = new_pop

            # 定期 checkpoint
            if output_dir is not None and self.generation % 20 == 0:
                self._save_checkpoint_full(Path(output_dir), evolution_seed, population)

        elapsed = time.time() - t0

        # v3: 在固定 selection seeds 上重新评估全部候选，选择真正的最优
        selection_seeds = list(self.config.selection_seeds) if self.config.selection_seeds else self.train_seeds[:20]

        if verbose:
            print(f"[GA select] Re-evaluating {len(self.candidate_archive)} candidates "
                  f"on {len(selection_seeds)} fixed selection seeds...")

        candidate_scores = []
        for i, adj_c in enumerate(self.candidate_archive):
            try:
                score = self._evaluate_genome(adj_c, selection_seeds, env_factory)
                candidate_scores.append((score, i, adj_c))
            except Exception:
                pass

        if candidate_scores:
            # 按 selection seeds 上的 score 排序
            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            best_sel_score, best_idx, best_adj = candidate_scores[0]

            if verbose:
                print(f"[GA select] Best candidate: selection_score={best_sel_score:.1f} "
                      f"(archive idx={best_idx}, {len(candidate_scores)} evaluated)")
                # 显示 stored best vs selection best 的差异
                stored_sel = None
                for s, i, a in candidate_scores:
                    if np.array_equal(a, self.best_adjacency):
                        stored_sel = s
                        stored_rank = candidate_scores.index((s, i, a)) + 1
                        break
                if stored_sel is not None:
                    print(f"[GA select] Stored best rank: {stored_rank}/{len(candidate_scores)} "
                          f"(selection_score={stored_sel:.1f})")

            self.best_adjacency = best_adj
            self.best_fitness = best_sel_score
        else:
            if verbose:
                print("[GA select] No valid candidates, keeping stored best")

        # 最终验证 best_adjacency 的不变量
        ok, msg = verify_topology_invariants(
            self.best_adjacency, self.allowed_mask, self.edge_count
        )
        if not ok:
            raise RuntimeError(f"Best topology invariant violation: {msg}")

        # 保存最终结果
        if output_dir is not None:
            self._save_final(output_dir, evolution_seed, elapsed)

        if verbose:
            print(f"[GA done] task={self.task.name} seed={evolution_seed} "
                  f"gens={self.generation} best={self.best_fitness:.1f} "
                  f"evals={self.evaluations_done} time={elapsed:.1f}s "
                  f"candidates={len(self.candidate_archive)}")

        return {
            "best_adjacency": self.best_adjacency,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
            "n_generations": self.generation,
            "evaluations_done": self.evaluations_done,
            "elapsed_sec": elapsed,
            "n_candidates": len(self.candidate_archive),
            "n_candidates_evaluated": len(candidate_scores),
        }

    def _save_checkpoint_full(
        self, output_dir: Path, evolution_seed: int, population: list[np.ndarray]
    ) -> None:
        ckpt = GACheckpoint(
            generation=self.generation,
            evaluations_done=self.evaluations_done,
            best_fitness=self.best_fitness,
            best_adjacency=self.best_adjacency.tolist() if self.best_adjacency is not None else None,
            population=[adj.tolist() for adj in population],
            fitness_history=self.fitness_history,
            elapsed_sec=0.0,  # 在 final 中更新
            seed_batch_index=self.generation // self.config.seed_rotation_batch,
        )
        path = output_dir / f"ga_ckpt_{self.task.name}_seed{evolution_seed}.json"
        ckpt.save(path)

    def _save_final(self, output_dir: str | Path, evolution_seed: int, elapsed: float) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "task": self.task.name,
            "evolution_seed": evolution_seed,
            "edge_count": self.edge_count,
            "n_generations": self.generation,
            "evaluations_done": self.evaluations_done,
            "best_fitness": self.best_fitness,
            "elapsed_sec": elapsed,
            "fitness_history": self.fitness_history,
        }
        path = output_dir / f"ga_result_{self.task.name}_seed{evolution_seed}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 保存最优拓扑
        np.save(
            output_dir / f"F0_topology_{self.task.name}_seed{evolution_seed}.npy",
            self.best_adjacency,
        )
