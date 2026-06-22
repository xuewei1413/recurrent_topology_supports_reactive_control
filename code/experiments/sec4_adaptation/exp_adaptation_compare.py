"""
实验 C3: 环境突变Adaptation Comparison
结构适应(1参数) vs 权重适应(10⁴参数) vs 完全再进化

平台：
  1. CartPole-v1 (gravity 突变)
  2. FlappyBird   (pipe_speed 突变)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import csv
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from src.snn import SNNController
from src.evolution import random_topology, mutate
from src.flappy_env_parametric import FlappyEnvParametric

# ── 依赖检查 ──────────────────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# 中文支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 环境配置 ──────────────────────────────────────────────────────────
CARTPOLE_STANDARD = {'gravity': 9.8, 'length': 0.5}
CARTPOLE_MUTANT = {'gravity': 15.0, 'length': 0.5}

FLAPPY_STANDARD = {'pipe_speed': 4.0, 'gravity': 1.25, 'pipe_gap': 100, 'pipe_frequency': 150}
FLAPPY_MUTANT = {'pipe_speed': 6.0, 'gravity': 1.25, 'pipe_gap': 100, 'pipe_frequency': 150}

# ── SNN 配置 ──────────────────────────────────────────────────────────
SNN_CONFIG_CP = {
    'n_total': 128, 'n_input': 8, 'n_output': 2,
    'sparsity': 0.41, 'weight_scale': 0.6,
    'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
    'pop_size': 40, 'n_generations': 400, 'n_elite': 8,
    'mutation_rate': 0.03, 'n_eval_episodes': 8,
    'run_baseline': False,
}

SNN_CONFIG_FB = {
    'n_total': 64, 'n_input': 8, 'n_output': 2,
    'sparsity': 0.20, 'weight_scale': 0.6,
    'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
    'pop_size': 40, 'n_generations': 400, 'n_elite': 12,
    'mutation_rate': 0.025, 'n_eval_episodes': 3,
    'max_frames': 3000, 'gap_reward_w': 0.8, 'pipe_bonus': 500,
    'target_sparsity': 0.15, 'density_penalty': 300,
    'run_baseline': False, 'solve_score': 10,
}

# ── PPO 配置 (与 D1 相同) ────────────────────────────────────────────
PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'verbose': 0,
    'policy_kwargs': dict(net_arch=[128, 128])
}


# ═══════════════════════════════════════════════════════════════════════
#  CartPole 相关函数
# ═══════════════════════════════════════════════════════════════════════

def _set_cartpole_params(env, env_params):
    """通过unwrapped修改CartPole环境参数"""
    if 'gravity' in env_params:
        env.unwrapped.gravity = env_params['gravity']
    if 'length' in env_params:
        env.unwrapped.length = env_params['length']


def evaluate_snn_cartpole(adj, types, config, env_params, n_episodes=20):
    """用指定环境参数评估 SNN 在 CartPole 上的性能"""
    controller = SNNController(config, adj, types)
    env = gym.make('CartPole-v1')
    _set_cartpole_params(env, env_params)
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        controller.reset()
        total = 0
        for _ in range(500):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
    env.close()
    return np.array(scores)


def ppo_adaptation_cartpole(env_params, total_timesteps=200_000, seed=0):
    """PPO 在突变 CartPole 环境重新训练"""
    def make_env(rank):
        def _init():
            env = gym.make('CartPole-v1')
            _set_cartpole_params(env, env_params)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([make_env(i) for i in range(8)])
    
    model = PPO('MlpPolicy', vec_env, seed=seed, device='cpu', **PPO_CONFIG)
    model.learn(total_timesteps=total_timesteps)

    eval_env = gym.make('CartPole-v1')
    _set_cartpole_params(eval_env, env_params)
    scores = []
    for ep in range(20):
        obs, _ = eval_env.reset(seed=seed + ep)
        total = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
    eval_env.close()
    vec_env.close()

    total_params = sum(p.numel() for p in model.policy.parameters())
    return np.array(scores), total_params


def reevolve_cartpole(env_params, config, seed=0):
    """在突变 CartPole 环境从头再进化（TO 模式）"""
    if seed is not None:
        np.random.seed(seed)
    n = config['n_total']

    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(config['n_input'], n - config['n_output'])
    n_inh = int(len(hidden_range) * 0.3)
    inh_idx = np.random.choice(hidden_range, n_inh, replace=False)
    neuron_types[inh_idx] = -1.0

    def _eval(adj):
        controller = SNNController(config, adj, neuron_types)
        env = gym.make('CartPole-v1')
        _set_cartpole_params(env, env_params)
        total = 0.0
        for _ in range(config['n_eval_episodes']):
            obs, _ = env.reset()
            controller.reset()
            for _ in range(500):
                action = controller.forward(np.array(obs, dtype=np.float64))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
        env.close()
        return total / config['n_eval_episodes']

    pop = [random_topology(n, config['sparsity']) for _ in range(config['pop_size'])]
    best_adj = None
    best_fitness = -np.inf

    print(f"[Re-evolve CartPole] env={env_params}")
    for gen in range(config['n_generations']):
        fitnesses = np.array([_eval(adj) for adj in pop])
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        gen_mean = fitnesses.mean()

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        solved = gen_best >= 490
        status = "*** SOLVED ***" if solved else ""
        print(f"  Gen {gen+1:4d}: best={gen_best:7.1f} mean={gen_mean:7.1f} {status}")

        if solved:
            break

        elite = [pop[i] for i in order[:config['n_elite']]]
        new_pop = list(elite)
        while len(new_pop) < config['pop_size']:
            parent = elite[np.random.randint(len(elite))]
            child = mutate(parent, config['mutation_rate'])
            new_pop.append(child)
        pop = new_pop

    return best_adj, neuron_types, best_fitness


# ═══════════════════════════════════════════════════════════════════════
#  FlappyBird 相关函数
# ═══════════════════════════════════════════════════════════════════════

def evaluate_snn_flappy(adj, types, config, env_params, n_episodes=20):
    """用指定环境参数评估 SNN 在 FlappyBird 上的性能（存活帧数）"""
    controller = SNNController(config, adj, types)
    scores = []
    for _ in range(n_episodes):
        env = FlappyEnvParametric(**env_params)
        state = env.reset()
        controller.reset()
        frames = 0
        while True:
            action = controller.forward(state)
            state, reward, done, info = env.step(action)
            frames += 1
            if done:
                break
        scores.append(frames)
    return np.array(scores)


class FlappyBirdGymWrapper(gym.Env):
    """将 FlappyEnvParametric 包装为 gymnasium 环境，供 PPO 使用"""
    metadata = {'render_modes': []}

    def __init__(self, env_params):
        super().__init__()
        self.env_params = env_params
        self.env = FlappyEnvParametric(**env_params)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        state = self.env.reset()
        return state, {}

    def step(self, action):
        state, reward, done, info = self.env.step(int(action))
        return state, float(reward), bool(done), False, info


def ppo_adaptation_flappy(env_params, total_timesteps=200_000, seed=0):
    """PPO 在突变 FlappyBird 环境重新训练"""
    def _make_env(rank):
        def _init():
            env = FlappyBirdGymWrapper(env_params)
            env.reset(seed=seed + rank)
            return env
        return _init

    vec_env = DummyVecEnv([_make_env(i) for i in range(8)])
    model = PPO('MlpPolicy', vec_env, seed=seed, device='cpu', **PPO_CONFIG)
    model.learn(total_timesteps=total_timesteps)

    eval_env = FlappyBirdGymWrapper(env_params)
    scores = []
    for ep in range(20):
        obs, _ = eval_env.reset(seed=seed + ep)
        frames = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            frames += 1
            if terminated or truncated:
                break
        scores.append(frames)
    eval_env.close()
    vec_env.close()

    total_params = sum(p.numel() for p in model.policy.parameters())
    return np.array(scores), total_params


def mutate_density_aware(adj, rate, target):
    """密度感知变异（来自 flappy_experiment.py）"""
    child = adj.copy()
    n = child.shape[0]
    max_conn = n * (n - 1)
    density = child.sum() / max_conn
    ratio = density / target if target > 0 else 1.0

    if ratio > 1.3:
        p_remove = min(rate * (1 + (ratio - 1.3) * 3), 0.3)
        p_add = rate * 0.2
    elif ratio < 0.7:
        p_add = min(rate * (1 + (0.7 - ratio) * 3), 0.3)
        p_remove = rate * 0.2
    else:
        p_add = p_remove = rate

    rnd = np.random.rand(*child.shape)
    child[(child == 1) & (rnd < p_remove)] = 0.0
    child[(child == 0) & (rnd < p_add)] = 1.0
    np.fill_diagonal(child, 0.0)
    return child


def evaluate_flappy_fitness(adj, neuron_types, config, env_params):
    """FlappyBird 进化用适应度函数（含密度惩罚，来自 flappy_experiment.py）"""
    controller = SNNController(config, adj, neuron_types)
    gap_w = config.get('gap_reward_w', 0.8)
    pipe_bonus = config.get('pipe_bonus', 500)
    total = 0.0

    for _ in range(config['n_eval_episodes']):
        env = FlappyEnvParametric(**env_params)
        state = env.reset()
        controller.reset()
        for _ in range(config['max_frames']):
            action = controller.forward(state)
            state, reward, done, info = env.step(action)
            total += 1.0
            total += info['gap_align'] * gap_w
            if done:
                break
        total += info['score'] * pipe_bonus

    raw = total / config['n_eval_episodes']
    n = adj.shape[0]
    density = adj.sum() / (n * (n - 1))
    target = config.get('target_sparsity', 0.15)
    penalty = config.get('density_penalty', 300)
    over = max(0.0, density - target)
    raw -= over * penalty
    return raw


def reevolve_flappy(env_params, config, seed=0):
    """在突变 FlappyBird 环境从头再进化"""
    if seed is not None:
        np.random.seed(seed)
    n = config['n_total']

    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(config['n_input'], n - config['n_output'])
    n_inh = int(len(hidden_range) * 0.3)
    inh_idx = np.random.choice(hidden_range, n_inh, replace=False)
    neuron_types[inh_idx] = -1.0

    def _eval(adj):
        return evaluate_flappy_fitness(adj, neuron_types, config, env_params)

    pop = [random_topology(n, config['sparsity']) for _ in range(config['pop_size'])]
    best_adj = None
    best_fitness = -np.inf

    print(f"[Re-evolve FlappyBird] env={env_params}")
    for gen in range(config['n_generations']):
        fitnesses = np.array([_eval(adj) for adj in pop])
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        gen_mean = fitnesses.mean()
        conns = int(pop[order[0]].sum())

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        est_pipes = max(0, int((gen_best - config['max_frames']) / 100)) \
            if gen_best > config['max_frames'] else 0
        solved = gen_best >= config['max_frames'] * (1 + config['gap_reward_w']) \
                           + config['solve_score'] * config['pipe_bonus']
        status = "*** SOLVED ***" if solved else ""
        print(f"  Gen {gen+1:4d}: best={gen_best:8.1f} mean={gen_mean:8.1f} "
              f"conns={conns:6d} pipes~={est_pipes:5d} {status}")

        if solved:
            break

        elite = [pop[i] for i in order[:config['n_elite']]]
        new_pop = list(elite)
        while len(new_pop) < config['pop_size']:
            parent = elite[np.random.randint(len(elite))]
            new_pop.append(mutate_density_aware(
                parent, config['mutation_rate'], config['target_sparsity']))
        pop = new_pop

    return best_adj, neuron_types, best_fitness


# ═══════════════════════════════════════════════════════════════════════
#  通用结构适应
# ═══════════════════════════════════════════════════════════════════════

def structure_adaptation(eval_fn, adj, types, base_config, env_params, param_name, param_range):
    """固定拓扑，网格搜索一个全局参数"""
    best_score, best_param = -np.inf, None
    all_results = []
    for p in param_range:
        cfg = base_config.copy()
        cfg[param_name] = p
        scores = eval_fn(adj, types, cfg, env_params, n_episodes=20)
        mean_score = scores.mean()
        all_results.append({'param': p, 'mean': mean_score, 'std': scores.std()})
        if mean_score > best_score:
            best_score, best_param = mean_score, p
    return best_param, best_score, all_results


# ═══════════════════════════════════════════════════════════════════════
#  实验执行与绘图
# ═══════════════════════════════════════════════════════════════════════

def plot_platform_results(results, platform_name, save_path):
    """绘制单个平台的 1×3 对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: weight_scale 扫描曲线
    ax = axes[0]
    ws_curve = results['ws_curve']
    means = [r['mean'] for r in ws_curve]
    ax.plot([r['param'] for r in ws_curve], means, 'o-', color='steelblue')
    ax.axhline(results['baseline_mean'], color='g', ls='--', label='Standard Env Baseline')
    ax.axhline(results['zero_shot_mean'], color='r', ls='--', label='Zero-shot')
    ax.set_xlabel('weight_scale')
    ax.set_ylabel('Score')
    ax.set_title(f'{platform_name} — Structural Adaptation: weight_scale')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图2: T_steps 扫描曲线
    ax = axes[1]
    ts_curve = results['ts_curve']
    means = [r['mean'] for r in ts_curve]
    ax.plot([r['param'] for r in ts_curve], means, 'o-', color='darkorange')
    ax.axhline(results['baseline_mean'], color='g', ls='--', label='Standard Env Baseline')
    ax.axhline(results['zero_shot_mean'], color='r', ls='--', label='Zero-shot')
    ax.set_xlabel('T_steps')
    ax.set_ylabel('Score')
    ax.set_title(f'{platform_name} — Structural Adaptation: T_steps')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3: 策略对比柱状图
    ax = axes[2]
    strategies = ['Zero-shot', 'Struct-A\n(weight)', 'Struct-A\n(T_steps)', 'PPO\nRetrain', 'Full\nRe-evolve']
    scores = [results['zero_shot_mean'], results['best_ws_score'],
              results['best_ts_score'], results['ppo_mean'], results['reevolve_mean']]
    colors = ['red', 'orange', 'gold', 'blue', 'purple']
    bars = ax.bar(strategies, scores, color=colors, edgecolor='black')
    ax.axhline(results['baseline_mean'], color='green', ls='--', alpha=0.5, label='Standard Env Baseline')
    ax.set_ylabel('Score (20-ep mean)')
    ax.set_title(f'{platform_name} — Adaptation Strategy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ymax = max(scores) if max(scores) > 0 else 1
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01,
                f'{s:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图表已保存: {save_path}')


def run_cartpole_section():
    """运行 CartPole 平台的全部实验"""
    print("\n" + "=" * 70)
    print("  平台 1: CartPole-v1")
    print("=" * 70)

    topo_path = '../../../data/key_npy/best_topology.npy'
    types_path = '../../../data/key_npy/neuron_types.npy'
    if not os.path.exists(topo_path) or not os.path.exists(types_path):
        print(f"警告: 未找到 {topo_path} 或 {types_path}")
        print("请先运行 CartPole 进化实验以生成标准环境最优拓扑。")
        return None

    adj = np.load(topo_path)
    types = np.load(types_path)
    config = SNN_CONFIG_CP.copy()

    # 1. Standard Env Baseline
    baseline = evaluate_snn_cartpole(adj, types, config, CARTPOLE_STANDARD, n_episodes=20)
    print('标准环境: %.1f ± %.1f' % (baseline.mean(), baseline.std()))

    # 2. Zero-shot泛化
    zero_shot = evaluate_snn_cartpole(adj, types, config, CARTPOLE_MUTANT, n_episodes=20)
    print('突变环境 (Zero-shot): %.1f ± %.1f' % (zero_shot.mean(), zero_shot.std()))

    # 3. 结构适应：调 weight_scale
    ws_range = np.arange(0.1, 2.0, 0.1)
    best_ws, best_score_ws, ws_curve = structure_adaptation(
        evaluate_snn_cartpole, adj, types, config, CARTPOLE_MUTANT,
        'weight_scale', ws_range)
    print('结构适应 (weight_scale): 最优=%.2f, Score=%.1f' % (best_ws, best_score_ws))

    # 4. 结构适应：调 T_steps
    ts_range = [5, 10, 15, 20, 25, 30, 40, 50]
    best_ts, best_score_ts, ts_curve = structure_adaptation(
        evaluate_snn_cartpole, adj, types, config, CARTPOLE_MUTANT,
        'T_steps', ts_range)
    print('结构适应 (T_steps): 最优=%d, Score=%.1f' % (best_ts, best_score_ts))

    # 5. PPO 重训练
    ppo_scores, ppo_params = ppo_adaptation_cartpole(CARTPOLE_MUTANT, seed=0)
    print('PPO 重训练: %.1f ± %.1f (参数量=%d)' % (ppo_scores.mean(), ppo_scores.std(), ppo_params))

    # 6. 完全再进化
    print('开始完全再进化 (CartPole 突变环境)...')
    t0 = time.time()
    re_adj, re_types, re_fitness = reevolve_cartpole(CARTPOLE_MUTANT, config, seed=0)
    re_scores = evaluate_snn_cartpole(re_adj, re_types, config, CARTPOLE_MUTANT, n_episodes=20)
    print('完全再进化: %.1f ± %.1f (耗时 %.1fs)' % (re_scores.mean(), re_scores.std(), time.time() - t0))

    return {
        'platform': 'CartPole',
        'baseline_mean': baseline.mean(),
        'zero_shot_mean': zero_shot.mean(),
        'zero_shot_std': zero_shot.std(),
        'best_ws': best_ws, 'best_ws_score': best_score_ws, 'ws_curve': ws_curve,
        'best_ts': best_ts, 'best_ts_score': best_score_ts, 'ts_curve': ts_curve,
        'ppo_mean': ppo_scores.mean(), 'ppo_std': ppo_scores.std(), 'ppo_params': ppo_params,
        'reevolve_mean': re_scores.mean(), 'reevolve_std': re_scores.std(),
    }


def run_flappy_section():
    """运行 FlappyBird 平台的全部实验"""
    print("\n" + "=" * 70)
    print("  平台 2: FlappyBird")
    print("=" * 70)

    topo_path = '../../../data/key_npy/flappy_best_topology.npy'
    types_path = '../../../data/key_npy/flappy_neuron_types.npy'
    config = SNN_CONFIG_FB.copy()

    adj, types = None, None
    if os.path.exists(topo_path) and os.path.exists(types_path):
        adj = np.load(topo_path)
        types = np.load(types_path)
        print(f"已加载现有拓扑: {topo_path}")
    else:
        print(f"未找到 {topo_path}，将在标准环境上从头进化...")
        adj, types, _ = reevolve_flappy(FLAPPY_STANDARD, config, seed=0)
        os.makedirs('results', exist_ok=True)
        np.save(topo_path, adj)
        np.save(types_path, types)
        print(f"已保存标准环境最优拓扑到 {topo_path}")

    # 1. Standard Env Baseline
    baseline = evaluate_snn_flappy(adj, types, config, FLAPPY_STANDARD, n_episodes=20)
    print('标准环境: %.1f ± %.1f' % (baseline.mean(), baseline.std()))

    # 2. Zero-shot泛化
    zero_shot = evaluate_snn_flappy(adj, types, config, FLAPPY_MUTANT, n_episodes=20)
    print('突变环境 (Zero-shot): %.1f ± %.1f' % (zero_shot.mean(), zero_shot.std()))

    # 3. 结构适应：调 weight_scale
    ws_range = np.arange(0.1, 2.0, 0.1)
    best_ws, best_score_ws, ws_curve = structure_adaptation(
        evaluate_snn_flappy, adj, types, config, FLAPPY_MUTANT,
        'weight_scale', ws_range)
    print('结构适应 (weight_scale): 最优=%.2f, Score=%.1f' % (best_ws, best_score_ws))

    # 4. 结构适应：调 T_steps
    ts_range = [5, 10, 15, 20, 25, 30, 40, 50]
    best_ts, best_score_ts, ts_curve = structure_adaptation(
        evaluate_snn_flappy, adj, types, config, FLAPPY_MUTANT,
        'T_steps', ts_range)
    print('结构适应 (T_steps): 最优=%d, Score=%.1f' % (best_ts, best_score_ts))

    # 5. PPO 重训练
    ppo_scores, ppo_params = ppo_adaptation_flappy(FLAPPY_MUTANT, seed=0)
    print('PPO 重训练: %.1f ± %.1f (参数量=%d)' % (ppo_scores.mean(), ppo_scores.std(), ppo_params))

    # 6. 完全再进化
    print('开始完全再进化 (FlappyBird 突变环境)...')
    t0 = time.time()
    re_adj, re_types, re_fitness = reevolve_flappy(FLAPPY_MUTANT, config, seed=0)
    re_scores = evaluate_snn_flappy(re_adj, re_types, config, FLAPPY_MUTANT, n_episodes=20)
    print('完全再进化: %.1f ± %.1f (耗时 %.1fs)' % (re_scores.mean(), re_scores.std(), time.time() - t0))

    return {
        'platform': 'FlappyBird',
        'baseline_mean': baseline.mean(),
        'zero_shot_mean': zero_shot.mean(),
        'zero_shot_std': zero_shot.std(),
        'best_ws': best_ws, 'best_ws_score': best_score_ws, 'ws_curve': ws_curve,
        'best_ts': best_ts, 'best_ts_score': best_score_ts, 'ts_curve': ts_curve,
        'ppo_mean': ppo_scores.mean(), 'ppo_std': ppo_scores.std(), 'ppo_params': ppo_params,
        'reevolve_mean': re_scores.mean(), 'reevolve_std': re_scores.std(),
    }


def save_summary(cp_res, fb_res):
    """保存 CSV 摘要与综合对比图"""
    os.makedirs('results/adaptation', exist_ok=True)

    csv_path = '../results/adaptation/adaptation_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['platform', 'strategy', 'mean', 'std', 'n_eval', 'params_adjusted'])

        if cp_res:
            w.writerow(['CartPole', 'baseline', cp_res['baseline_mean'], None, 20, 0])
            w.writerow(['CartPole', 'zero_shot', cp_res['zero_shot_mean'], cp_res['zero_shot_std'], 20, 0])
            w.writerow(['CartPole', 'struct_weight_scale', cp_res['best_ws_score'], None, 20, 1])
            w.writerow(['CartPole', 'struct_T_steps', cp_res['best_ts_score'], None, 20, 1])
            w.writerow(['CartPole', 'ppo_retrain', cp_res['ppo_mean'], cp_res['ppo_std'], 20, cp_res['ppo_params']])
            w.writerow(['CartPole', 'reevolve', cp_res['reevolve_mean'], cp_res['reevolve_std'], 20, 'N/A'])

        if fb_res:
            w.writerow(['FlappyBird', 'baseline', fb_res['baseline_mean'], None, 20, 0])
            w.writerow(['FlappyBird', 'zero_shot', fb_res['zero_shot_mean'], fb_res['zero_shot_std'], 20, 0])
            w.writerow(['FlappyBird', 'struct_weight_scale', fb_res['best_ws_score'], None, 20, 1])
            w.writerow(['FlappyBird', 'struct_T_steps', fb_res['best_ts_score'], None, 20, 1])
            w.writerow(['FlappyBird', 'ppo_retrain', fb_res['ppo_mean'], fb_res['ppo_std'], 20, fb_res['ppo_params']])
            w.writerow(['FlappyBird', 'reevolve', fb_res['reevolve_mean'], fb_res['reevolve_std'], 20, 'N/A'])

    print(f'CSV 已保存: {csv_path}')

    # 综合对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, name in zip(axes, [cp_res, fb_res], ['CartPole-v1', 'FlappyBird']):
        if res is None:
            ax.text(0.5, 0.5, '数据缺失', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
            continue

        strategies = ['Zero-shot', 'Struct-A\n(weight)', 'Struct-A\n(T_steps)', 'PPO\nRetrain', 'Full\nRe-evolve']
        scores = [res['zero_shot_mean'], res['best_ws_score'],
                  res['best_ts_score'], res['ppo_mean'], res['reevolve_mean']]
        colors = ['red', 'orange', 'gold', 'blue', 'purple']
        bars = ax.bar(strategies, scores, color=colors, edgecolor='black')
        ax.axhline(res['baseline_mean'], color='green', ls='--', alpha=0.5, label='Standard Env Baseline')
        ax.set_ylabel('Score (20-ep mean)')
        ax.set_title(f'{name} — Adaptation Strategy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ymax = max(scores) if max(scores) > 0 else 1
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01,
                    f'{s:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    overview_path = '../results/adaptation/adaptation_overview.png'
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'综合图表已保存: {overview_path}')


def main():
    os.makedirs('results/adaptation', exist_ok=True)

    print("=" * 70)
    print("  实验 C3: 环境突变Adaptation Comparison")
    print("  结构适应(1参数) vs 权重适应(10^4参数) vs 完全再进化")
    print("=" * 70)

    cp_res = run_cartpole_section()
    if cp_res:
        plot_platform_results(cp_res, 'CartPole-v1', '../results/adaptation/adaptation_cartpole.png')

    fb_res = run_flappy_section()
    if fb_res:
        plot_platform_results(fb_res, 'FlappyBird', '../results/adaptation/adaptation_flappy.png')

    save_summary(cp_res, fb_res)
    print("\n实验 C3 完成。")


if __name__ == '__main__':
    main()
