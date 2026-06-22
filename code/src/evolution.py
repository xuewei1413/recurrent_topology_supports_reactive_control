"""
遗传算法 — 进化连接拓扑

进化的对象：二值邻接矩阵（谁连谁）
不进化的：权重幅值、神经元类型、LIF 参数

适应度函数：CartPole-v1 平均存活步数（越高越好，满分=500）
"""

import numpy as np
import gymnasium as gym
from concurrent.futures import ProcessPoolExecutor
from .snn import SNNController, WeightedSNNController


# ── 并行评估辅助 ──────────────────────────────────────────────────────────

def _eval_worker(args):
    """子进程评估单个拓扑，供 ProcessPoolExecutor 调用"""
    adj, neuron_types, config, weight_scale, weight_matrix, eval_seeds = args
    kwargs = dict(eval_seeds=eval_seeds)
    if weight_matrix is not None:
        return evaluate_coevolution(adj, neuron_types, config, weight_matrix=weight_matrix, **kwargs)
    elif weight_scale is not None:
        return evaluate_coevolution(adj, neuron_types, config, weight_scale=weight_scale, **kwargs)
    else:
        return evaluate_coevolution(adj, neuron_types, config, **kwargs)


def _parallel_eval(tasks, n_workers):
    """并行或串行评估，n_workers>1 时并行"""
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return np.array(list(pool.map(_eval_worker, tasks)))
    else:
        return np.array([_eval_worker(t) for t in tasks])


# ── 拓扑操作 ─────────────────────────────────────────────────────────────

def random_topology(n: int, sparsity: float) -> np.ndarray:
    """生成随机稀疏邻接矩阵，无自连接"""
    adj = (np.random.rand(n, n) < sparsity).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def mutate(adj: np.ndarray, rate: float) -> np.ndarray:
    """随机翻转 rate 比例的连接（0→1 或 1→0）"""
    child = adj.copy()
    mask  = np.random.rand(*child.shape) < rate
    child[mask] = 1.0 - child[mask]
    np.fill_diagonal(child, 0.0)
    return child


def crossover_topology(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """均匀交叉两个邻接矩阵"""
    mask = np.random.rand(*p1.shape) < 0.5
    child = np.where(mask, p1, p2).astype(np.float64)
    np.fill_diagonal(child, 0.0)
    return child


def tournament_select(pop, fitnesses, tournament_size=3):
    """锦标赛选择：随机选 tournament_size 个个体，返回最优的"""
    candidates = np.random.choice(len(pop), size=min(tournament_size, len(pop)), replace=False)
    best = candidates[np.argmax(fitnesses[candidates])]
    return pop[best]


# ── 适应度评估 ────────────────────────────────────────────────────────────

def evaluate_coevolution(adj: np.ndarray, neuron_types: np.ndarray, config: dict,
                         weight_scale=None, weight_matrix=None, eval_seeds=None) -> float:
    """
    通用评估函数，支持三种模式：
    - TO: weight_scale=None, weight_matrix=None → 使用config['weight_scale']
    - TG: weight_scale=float, weight_matrix=None → 使用传入的全局weight_scale
    - TI: weight_scale=None, weight_matrix=ndarray → 使用独立权重矩阵
    """
    if weight_matrix is not None:
        controller = WeightedSNNController(config, adj, neuron_types, weight_matrix)
    elif weight_scale is not None:
        W = adj * neuron_types[:, np.newaxis] * weight_scale
        controller = WeightedSNNController(config, adj, neuron_types, W)
    else:
        controller = SNNController(config, adj, neuron_types)

    env   = gym.make('CartPole-v1')
    total = 0.0

    for ep in range(config['n_eval_episodes']):
        seed = eval_seeds[ep] if eval_seeds is not None else None
        obs, _ = env.reset(seed=seed)
        controller.reset()
        for step in range(500):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break

    env.close()
    return total / config['n_eval_episodes']


def evaluate(adj: np.ndarray, neuron_types: np.ndarray, config: dict) -> float:
    """
    让一个拓扑跑 n_eval_episodes 局 CartPole，返回平均步数。
    权重由神经元类型决定，不做任何学习/梯度更新。
    """
    return evaluate_coevolution(adj, neuron_types, config)


# ── 基线测试（实验前对照） ──────────────────────────────────────────────────

def baseline_test(neuron_types: np.ndarray, config: dict, n_samples: int = 20) -> None:
    """评估随机拓扑的 fitness 分布，作为进化前基线"""
    print(f"\n[Baseline] sampling {n_samples} random topologies...")
    scores = []
    for i in range(n_samples):
        adj = random_topology(config['n_total'], config['sparsity'])
        f   = evaluate(adj, neuron_types, config)
        scores.append(f)
        print(f"  #{i+1:02d}: {f:.1f} steps")

    scores = np.array(scores)
    print(f"[Baseline] mean={scores.mean():.1f}  max={scores.max():.1f}  min={scores.min():.1f}\n")
    return scores


# ── 进化主循环 ────────────────────────────────────────────────────────────

def run_evolution(config: dict):
    """
    返回：(最优邻接矩阵, 神经元类型, best_history, mean_history)
    """
    n = config['n_total']

    # ── 神经元类型（固定，整个实验不变）─────────────────────────────────
    # 隐藏层 30% 为抑制性，对应生物皮层 E/I 比例
    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(config['n_input'], n - config['n_output'])
    n_inh        = int(len(hidden_range) * 0.3)
    inh_idx      = np.random.choice(hidden_range, n_inh, replace=False)
    neuron_types[inh_idx] = -1.0

    n_exc = len(hidden_range) - n_inh
    print(f"Neuron types: excitatory={n_exc}, inhibitory={n_inh}")

    # ── 基线对照 ─────────────────────────────────────────────────────────
    if config.get('run_baseline', True):
        baseline_test(neuron_types, config, n_samples=10)

    # ── 初始化种群 ────────────────────────────────────────────────────────
    pop = [random_topology(n, config['sparsity'])
           for _ in range(config['pop_size'])]

    best_history = []
    mean_history = []
    best_adj     = None
    best_fitness = -np.inf

    # 自适应突变率参数
    base_mut_rate    = config.get('mutation_rate', 0.03)
    current_mut_rate = base_mut_rate
    stagnation       = 0
    stagnation_limit = config.get('stagnation_limit', 10)
    mut_rate_cap     = config.get('mut_rate_cap', 0.10)

    tournament_size  = config.get('tournament_size', 3)
    n_workers        = config.get('n_workers', 1)

    print("=" * 55)
    print(f"{'Gen':>4}  {'Best':>7}  {'Mean':>7}  {'Conns':>6}  {'MutR':>5}  Status")
    print("=" * 55)

    for gen in range(config['n_generations']):
        # 评估整代（支持并行）
        tasks = [(adj, neuron_types, config, None, None, None) for adj in pop]
        fitnesses = _parallel_eval(tasks, n_workers)

        order  = np.argsort(fitnesses)[::-1]
        gen_best  = fitnesses[order[0]]
        gen_mean  = fitnesses.mean()
        best_conn = int(pop[order[0]].sum())

        best_history.append(gen_best)
        mean_history.append(gen_mean)

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj     = pop[order[0]].copy()
            stagnation   = 0
            current_mut_rate = base_mut_rate
        else:
            stagnation += 1
            if stagnation >= stagnation_limit:
                current_mut_rate = min(current_mut_rate * 1.5, mut_rate_cap)

        # 稳健通关条件：8局均值 ≥ 490（不接受"运气型"拓扑）
        solved = gen_best >= 490
        status = "*** SOLVED ***" if solved else ""
        print(f"{gen+1:4d}  {gen_best:7.1f}  {gen_mean:7.1f}  {best_conn:6d}  {current_mut_rate:.3f}  {status}")

        if solved:
            break

        # 精英保留 + 锦标赛选择生成下一代
        elite   = [pop[i] for i in order[:config['n_elite']]]
        new_pop = list(elite)
        while len(new_pop) < config['pop_size']:
            parent = tournament_select(pop, fitnesses, tournament_size)
            new_pop.append(mutate(parent, current_mut_rate))
        pop = new_pop

    print("=" * 55)
    return best_adj, neuron_types, best_history, mean_history


def run_evolution_coevolution(config: dict, mode='TO', seed=None):
    """
    拓扑-权重共进化主循环

    mode:
        'TO' -> Topology-Only (原方法，只进化邻接矩阵)
        'TG' -> Topology + Global weight_scale 共进化
        'TI' -> Topology + Independent weights 共进化

    基因组编码：
        - TO: 邻接矩阵 (N×N bits)
        - TG: 邻接矩阵 + 1个float (weight_scale)
        - TI: 邻接矩阵 + 权重幅值矩阵 (N×N floats，符号由neuron_types决定)

    变异策略：
        - 拓扑变异：与原代码相同，flip mutation rate=0.03
        - weight_scale变异：高斯噪声 N(0, 0.1)，clip到[0.1, 2.0]
        - 独立权重变异：高斯噪声 N(0, 0.05)，以0.1概率对每个非零连接变异，clip到[0.1, 1.5]

    返回：
        (best_adj, neuron_types, best_history, mean_history, extra)
        其中extra:
            - TO: {'weight_scale': config['weight_scale']}
            - TG: {'weight_scale': best_global_scale}
            - TI: {'weight_matrix': best_weight_matrix}
    """
    if seed is not None:
        np.random.seed(seed)

    n = config['n_total']

    # 固定评估种子：每个 seed 运行使用不同的确定性初始条件
    eval_seeds = [seed * 1000 + ep for ep in range(config['n_eval_episodes'])] if seed is not None else None

    # ── 神经元类型（固定，整个实验不变）─────────────────────────────────
    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(config['n_input'], n - config['n_output'])
    n_inh        = int(len(hidden_range) * 0.3)
    inh_idx      = np.random.choice(hidden_range, n_inh, replace=False)
    neuron_types[inh_idx] = -1.0

    n_exc = len(hidden_range) - n_inh
    print(f"Neuron types: excitatory={n_exc}, inhibitory={n_inh}")
    print(f"Mode: {mode}")

    # ── 基线对照 ─────────────────────────────────────────────────────────
    if config.get('run_baseline', True):
        baseline_test(neuron_types, config, n_samples=10)

    # ── 初始化种群 ────────────────────────────────────────────────────────
    if mode == 'TO':
        pop = [random_topology(n, config['sparsity'])
               for _ in range(config['pop_size'])]
    elif mode == 'TG':
        pop = [(random_topology(n, config['sparsity']), config['weight_scale'])
               for _ in range(config['pop_size'])]
    elif mode == 'TI':
        adjs = [random_topology(n, config['sparsity'])
                for _ in range(config['pop_size'])]
        pop = [(adj, config['weight_scale'] * adj) for adj in adjs]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    best_history = []
    mean_history = []
    best_adj     = None
    best_fitness = -np.inf
    best_extra   = None

    # 自适应突变率参数
    base_mut_rate    = config.get('mutation_rate', 0.03)
    current_mut_rate = base_mut_rate
    stagnation       = 0
    stagnation_limit = config.get('stagnation_limit', 10)
    mut_rate_cap     = config.get('mut_rate_cap', 0.10)

    tournament_size  = config.get('tournament_size', 3)
    n_workers        = config.get('n_workers', 1)

    print("=" * 55)
    print(f"{'Gen':>4}  {'Best':>7}  {'Mean':>7}  {'Conns':>6}  {'MutR':>5}  Status")
    print("=" * 55)

    for gen in range(config['n_generations']):
        # 评估整代（支持并行）
        if mode == 'TO':
            tasks = [(adj, neuron_types, config, None, None, eval_seeds) for adj in pop]
        elif mode == 'TG':
            tasks = [(adj, neuron_types, config, ws, None, eval_seeds) for adj, ws in pop]
        else:  # TI
            tasks = [(adj, neuron_types, config, None,
                      adj * neuron_types[:, np.newaxis] * mag, eval_seeds) for adj, mag in pop]
        fitnesses = _parallel_eval(tasks, n_workers)

        order     = np.argsort(fitnesses)[::-1]
        gen_best  = fitnesses[order[0]]
        gen_mean  = fitnesses.mean()

        if mode == 'TO':
            best_conn = int(pop[order[0]].sum())
        else:
            best_conn = int(pop[order[0]][0].sum())

        best_history.append(gen_best)
        mean_history.append(gen_mean)

        if gen_best > best_fitness:
            best_fitness = gen_best
            if mode == 'TO':
                best_adj   = pop[order[0]].copy()
                best_extra = {'weight_scale': config['weight_scale']}
            elif mode == 'TG':
                best_adj   = pop[order[0]][0].copy()
                best_extra = {'weight_scale': float(pop[order[0]][1])}
            else:  # TI
                best_adj   = pop[order[0]][0].copy()
                best_mag   = pop[order[0]][1].copy()
                best_extra = {'weight_matrix': best_adj * neuron_types[:, np.newaxis] * best_mag}
            stagnation   = 0
            current_mut_rate = base_mut_rate
        else:
            stagnation += 1
            if stagnation >= stagnation_limit:
                current_mut_rate = min(current_mut_rate * 1.5, mut_rate_cap)

        # 稳健通关条件：8局均值 ≥ 490
        solved = gen_best >= 490
        status = "*** SOLVED ***" if solved else ""
        print(f"{gen+1:4d}  {gen_best:7.1f}  {gen_mean:7.1f}  {best_conn:6d}  {current_mut_rate:.3f}  {status}")

        if solved:
            break

        # 精英保留 + 锦标赛选择生成下一代
        elite = [pop[i] for i in order[:config['n_elite']]]
        new_pop = list(elite)

        crossover_rate = config.get('crossover_rate', 0.5)

        while len(new_pop) < config['pop_size']:
            parent = tournament_select(pop, fitnesses, tournament_size)

            if mode == 'TO':
                if np.random.rand() < crossover_rate:
                    parent2 = tournament_select(pop, fitnesses, tournament_size)
                    child = crossover_topology(parent, parent2)
                else:
                    child = parent.copy()
                child = mutate(child, current_mut_rate)
                new_pop.append(child)

            elif mode == 'TG':
                p_adj, p_ws = parent
                if np.random.rand() < crossover_rate:
                    p2_adj, p2_ws = tournament_select(pop, fitnesses, tournament_size)
                    child_adj = crossover_topology(p_adj, p2_adj)
                    child_ws = (p_ws + p2_ws) / 2.0
                else:
                    child_adj = p_adj.copy()
                    child_ws = p_ws
                child_adj = mutate(child_adj, current_mut_rate)
                child_ws = float(np.clip(child_ws + np.random.normal(0, 0.1), 0.1, 2.0))
                new_pop.append((child_adj, child_ws))

            else:  # TI
                p_adj, p_mag = parent
                if np.random.rand() < crossover_rate:
                    p2_adj, p2_mag = tournament_select(pop, fitnesses, tournament_size)
                    child_adj = crossover_topology(p_adj, p2_adj)
                    # 混合权重幅值
                    child_mag = config['weight_scale'] * np.ones((n, n))
                    both  = (p_adj == 1) & (p2_adj == 1) & (child_adj == 1)
                    only1 = (p_adj == 1) & (p2_adj == 0) & (child_adj == 1)
                    only2 = (p_adj == 0) & (p2_adj == 1) & (child_adj == 1)
                    child_mag[both]  = (p_mag[both] + p2_mag[both]) / 2.0
                    child_mag[only1] = p_mag[only1]
                    child_mag[only2] = p2_mag[only2]
                    child_mag[child_adj == 0] = 0.0
                else:
                    child_adj = p_adj.copy()
                    child_mag = p_mag.copy()

                old_adj = child_adj.copy()
                child_adj = mutate(child_adj, current_mut_rate)

                # 新连接：幅值初始化为 weight_scale
                new_mask = (old_adj == 0) & (child_adj == 1)
                child_mag[new_mask] = config['weight_scale']

                # 删除连接：幅值清零
                del_mask = (old_adj == 1) & (child_adj == 0)
                child_mag[del_mask] = 0.0

                # 已有连接幅值变异（高斯噪声 N(0, 0.05)，以0.1概率触发）
                active_mask = child_adj == 1
                mutate_mask = active_mask & (np.random.rand(n, n) < 0.1)
                noise = np.random.normal(0, 0.05, size=(n, n))
                child_mag[mutate_mask] += noise[mutate_mask]
                child_mag = np.clip(child_mag, 0.1, 1.5)

                new_pop.append((child_adj, child_mag))

        pop = new_pop

    print("=" * 55)
    return best_adj, neuron_types, best_history, mean_history, best_extra
