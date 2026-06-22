"""
实验 5A：MuJoCo 具身连续控制（Section 3.5, Fig 14a-b）

在 MuJoCo 仿真机器人上验证纯拓扑进化在连续高维控制任务上的能力。

测试环境：
  Hopper-v4  — 单腿跳跃（11D 状态, 3D 连续动作, 64 神经元）
  HalfCheetah-v4 — 半身猎豹（17D 状态, 6D 连续动作, 64 神经元）

对比维度：
  - Struct-Dyn SNN（纯拓扑, 1 标量参数）vs PPO+MLP（34,691 参数）
  - 极端参数效率：1 参数 vs 34,691 参数，达到 PPO 的 48%

论文关键结果（Section 3.5）：
  Hopper-v4: SNN 1009.1 分（20-ep strict），PPO 1M-step 3-seed reference 2083.7 分
  进化过程：10 代随机 -> 3 代快速跃升（637->823->1013）-> 稳定平台
  Gravity 适应：1.5x 重力下保持 ~70% 性能，零重训练

用法：
  python exp_hopper_evolution.py                    # 默认 Hopper-v4
  python exp_hopper_evolution.py --env Hopper-v4    # 指定环境
  python exp_hopper_evolution.py --skip-ppo         # 跳过 PPO 训练（耗时长）

输出：
  results/embodied_reflex/Hopper_v4_evolution.png   <- 进化曲线
  results/embodied_reflex/Hopper_v4_adaptation.png  <- gravity 扰动
  results/embodied_reflex/hopper_*.json             <- 结果数据
"""

import numpy as np
import os, sys, time, json

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.continuous_snn import ContinuousSNNController, evaluate_continuous

# ── 配置 ──────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'embodied_reflex')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 环境配置表
ENV_CONFIGS = {
    'Hopper-v4': {
        'obs_dim': 11, 'action_dim': 3,
        'level': 0, 'desc': 'Sense-Act Reflex',
        'n_total': 64, 'T_steps': 15, 'sparsity': 0.15,
    },
    'HalfCheetah-v4': {
        'obs_dim': 17, 'action_dim': 6,
        'level': 1, 'desc': 'Dynamic Balancing',
        'n_total': 96, 'T_steps': 15, 'sparsity': 0.12,
    },
    'Ant-v4': {
        'obs_dim': 27, 'action_dim': 8,
        'level': 2, 'desc': 'Force Reflex',
        'n_total': 128, 'T_steps': 20, 'sparsity': 0.10,
    },
}

# 通用 SNN 超参
SNN_COMMON = {
    'tau': 5.0, 'threshold': 0.8, 'weight_scale': 0.6,
}

# 进化配置（小规模快速验证）
EVO_CONFIG = {
    'pop_size': 40,
    'n_generations': 80,
    'mutate_rate': 0.03,
    'n_eval_episodes': 5,      # Hopper评估噪声大，每基因组5局
    'max_steps_per_ep': 1000,
}


def make_env(env_name, gravity_scale=1.0):
    """创建环境，支持重力扰动。"""
    import gymnasium as gym
    env = gym.make(env_name)
    # MuJoCo 环境可通过 model.opt.gravity 修改重力
    if gravity_scale != 1.0:
        try:
            env.unwrapped.model.opt.gravity[2] = -9.81 * gravity_scale
        except Exception as e:
            print(f"  [Warning] Could not modify gravity: {e}")
    return env


# ── SNN 进化（连续控制） ──────────────────────────────────────────────────

def random_topology(n, sparsity):
    adj = (np.random.rand(n, n) < sparsity).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def mutate_topology(adj, rate):
    child = adj.copy()
    mask = np.random.rand(*child.shape) < rate
    child[mask] = 1.0 - child[mask]
    np.fill_diagonal(child, 0.0)
    return child


def evaluate_snn_fitness(adj, neuron_types, config, env_name, n_eps=3, max_steps=1000, gravity_scale=1.0):
    """评估一个 SNN 拓扑在具身环境中的适应度。"""
    controller = ContinuousSNNController(config, adj, neuron_types)
    env = make_env(env_name, gravity_scale)
    total_reward = 0.0
    for _ in range(n_eps):
        obs, _ = env.reset()
        controller.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_reward += ep_reward
    env.close()
    return total_reward / n_eps


def run_evolution_continuous(env_name, seed=0):
    """为指定具身环境运行小规模 GA，返回最优拓扑。"""
    env_cfg = ENV_CONFIGS[env_name]
    np.random.seed(seed)

    n = env_cfg['n_total']
    n_in = env_cfg['obs_dim'] * 2
    n_out = env_cfg['action_dim']

    config = {
        'n_total': n, 'n_input': n_in, 'n_output': n_out,
        'T_steps': env_cfg['T_steps'],
        **SNN_COMMON,
        'n_eval_episodes': EVO_CONFIG['n_eval_episodes'],
    }

    # 神经元类型：隐藏层 30% 抑制
    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(n_in, n - n_out)
    if len(hidden_range) > 0:
        n_inh = int(len(hidden_range) * 0.3)
        inh_idx = np.random.choice(hidden_range, n_inh, replace=False)
        neuron_types[inh_idx] = -1.0

    print(f"\n[Evolution] {env_name} | n={n} (in={n_in}, out={n_out}) | seed={seed}")
    print(f"  Types: exc={np.sum(neuron_types==1)}, inh={np.sum(neuron_types==-1)}")

    # 初始化种群
    pop = [random_topology(n, env_cfg['sparsity']) for _ in range(EVO_CONFIG['pop_size'])]
    best_hist = []
    mean_hist = []
    best_adj = None
    best_fitness = -np.inf

    print(f"{'Gen':>4}  {'Best':>9}  {'Mean':>9}  {'Conns':>6}")
    for gen in range(EVO_CONFIG['n_generations']):
        fitnesses = []
        for adj in pop:
            f = evaluate_snn_fitness(adj, neuron_types, config, env_name,
                                     n_eps=EVO_CONFIG['n_eval_episodes'],
                                     max_steps=EVO_CONFIG['max_steps_per_ep'])
            fitnesses.append(f)
        fitnesses = np.array(fitnesses)

        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        gen_mean = fitnesses.mean()
        best_conn = int(pop[order[0]].sum())

        best_hist.append(float(gen_best))
        mean_hist.append(float(gen_mean))

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        print(f"{gen+1:4d}  {gen_best:9.2f}  {gen_mean:9.2f}  {best_conn:6d}")

        # 精英 + 变异
        elite_size = 3
        new_pop = [pop[i].copy() for i in order[:elite_size]]
        while len(new_pop) < EVO_CONFIG['pop_size']:
            parent = pop[np.random.choice(order[:10])]
            child = mutate_topology(parent, EVO_CONFIG['mutate_rate'])
            new_pop.append(child)
        pop = new_pop

    print(f"  Best fitness: {best_fitness:.2f}")
    return best_adj, neuron_types, config, best_hist, mean_hist, best_fitness


# ── PPO 基线 ──────────────────────────────────────────────────────────────

def train_ppo_baseline(env_name, total_timesteps=200_000, seed=0):
    """训练 PPO 基线，返回模型和训练日志。"""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("[Error] stable-baselines3 not installed. Run: uv pip install stable-baselines3")
        return None, None

    def _make():
        return make_env(env_name)

    vec_env = DummyVecEnv([_make])
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=seed,
                n_steps=2048, batch_size=64, n_epochs=10,
                learning_rate=3e-4, ent_coef=0.0)

    print(f"\n[PPO] Training {env_name} for {total_timesteps} timesteps...")
    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - start
    print(f"  Done in {elapsed/60:.1f} min")

    # 评估
    eval_env = make_env(env_name)
    rewards = []
    for _ in range(10):
        obs, _ = eval_env.reset()
        total = 0.0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    eval_env.close()
    print(f"  PPO eval: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    return model, {'rewards': rewards, 'time': elapsed}


# ── 适应实验：重力扰动 ────────────────────────────────────────────────────

def adaptation_experiment(best_adj, neuron_types, config, env_name, gravity_scales=[1.0, 1.3, 1.5, 2.0]):
    """
    结构适应：固定拓扑，只调 weight_scale。
    对比 PPO 在突变环境中的零样本表现。
    """
    print(f"\n[Adaptation] {env_name}")
    results = {'structural': {}, 'ppo': {}}

    # 结构适应：网格搜索 weight_scale
    for g in gravity_scales:
        print(f"\n  Gravity scale = {g:.1f}")
        best_ws = None
        best_score = -np.inf
        scores = []
        for ws in np.arange(0.1, 2.0, 0.1):
            cfg = config.copy()
            cfg['weight_scale'] = ws
            score = evaluate_snn_fitness(best_adj, neuron_types, cfg, env_name,
                                         n_eps=3, max_steps=1000, gravity_scale=g)
            scores.append((ws, score))
            if score > best_score:
                best_score = score
                best_ws = ws
        results['structural'][g] = {
            'best_ws': best_ws,
            'best_score': best_score,
            'all_scores': scores,
        }
        print(f"    Struct adapt: ws={best_ws:.1f}, score={best_score:.1f}")

    return results


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_evolution_curves(env_name, snn_hist, ppo_rewards, out_dir):
    if not HAS_MPL:
        print("[Warning] matplotlib not available, skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(snn_hist['best'], label='SNN Best', color='steelblue')
    ax.plot(snn_hist['mean'], label='SNN Mean', color='lightblue', alpha=0.7)
    if ppo_rewards:
        ax.axhline(np.mean(ppo_rewards), color='coral', linestyle='--', label=f'PPO Final ({np.mean(ppo_rewards):.0f})')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Mean Reward)')
    ax.set_title(f'{env_name} — SNN Evolution vs PPO Baseline')
    ax.legend()
    fig_path = os.path.join(out_dir, f'{env_name.replace("-", "_")}_evolution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_adaptation_comparison(env_name, adaptation_results, out_dir):
    if not HAS_MPL or not adaptation_results or 'structural' not in adaptation_results:
        return
    gravities = sorted(adaptation_results['structural'].keys())
    struct_scores = [adaptation_results['structural'][g]['best_score'] for g in gravities]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gravities, struct_scores, 'o-', color='steelblue', label='Structural Adapt (grid search weight_scale)')
    ax.set_xlabel('Gravity Scale')
    ax.set_ylabel('Best Mean Reward')
    ax.set_title(f'{env_name} — Structural Adaptation under Gravity Perturbation')
    ax.legend()
    fig_path = os.path.join(out_dir, f'{env_name.replace("-", "_")}_adaptation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_level_summary(all_results, out_dir):
    """三层实验的汇总对比图。"""
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (env_name, res) in zip(axes, all_results.items()):
        env_cfg = ENV_CONFIGS[env_name]
        labels = ['SNN Best', 'PPO']
        snn_best = res.get('snn_best_fitness', 0)
        ppo_mean = np.mean(res.get('ppo_rewards', [0])) if res.get('ppo_rewards') else 0
        colors = ['steelblue', 'coral']
        ax.bar(labels, [snn_best, ppo_mean], color=colors)
        ax.set_ylabel('Mean Reward')
        ax.set_title(f"Level {env_cfg['level']}: {env_cfg['desc']}\n{env_name}")
        # 标注比例
        if ppo_mean > 0:
            ax.text(0, snn_best * 0.5, f"{snn_best:.0f}", ha='center', va='center', color='white', fontweight='bold')
            ax.text(1, ppo_mean * 0.5, f"{ppo_mean:.0f}", ha='center', va='center', color='white', fontweight='bold')
            ratio = snn_best / ppo_mean * 100
            ax.text(0.5, max(snn_best, ppo_mean) * 1.05, f"SNN = {ratio:.0f}% PPO",
                    ha='center', fontsize=10, color='black')
    fig_path = os.path.join(out_dir, 'level_summary.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v4', choices=list(ENV_CONFIGS.keys()))
    parser.add_argument('--skip-ppo', action='store_true', default=False)
    args = parser.parse_args()

    print("="*70)
    print(f"  Experiment 1C: {args.env} — {ENV_CONFIGS[args.env]['desc']}")
    print("="*70)

    all_results = {}
    envs_to_run = [args.env]

    for env_name in envs_to_run:
        print(f"\n{'='*70}")
        print(f"  Environment: {env_name} — {ENV_CONFIGS[env_name]['desc']}")
        print(f"{'='*70}")

        # 1. SNN 进化
        best_adj, neuron_types, config, best_hist, mean_hist, best_fitness = \
            run_evolution_continuous(env_name, seed=42)

        # 保存拓扑
        topo_path = os.path.join(RESULTS_DIR, f'{env_name.replace("-", "_")}_best_topology.npy')
        np.save(topo_path, best_adj)
        np.save(topo_path.replace('best_topology', 'neuron_types'), neuron_types)
        print(f"  Saved topology: {topo_path}")

        # 2. PPO baseline (optional, time-consuming; skip with --skip-ppo)
        ppo_model = None
        ppo_info = None
        if not args.skip_ppo:
            try:
                ppo_model, ppo_info = train_ppo_baseline(env_name, total_timesteps=100_000, seed=42)
            except Exception as e:
                print(f"  [Warning] PPO training failed: {e}")
        else:
            print("\n[PPO] Skipped (--skip-ppo)")

        # 3. 适应实验
        adaptation_res = adaptation_experiment(best_adj, neuron_types, config, env_name,
                                                gravity_scales=[1.0, 1.3, 1.5])

        # 4. 可视化
        snn_hist = {'best': best_hist, 'mean': mean_hist}
        ppo_rewards = ppo_info['rewards'] if ppo_info else None
        plot_evolution_curves(env_name, snn_hist, ppo_rewards, RESULTS_DIR)
        plot_adaptation_comparison(env_name, adaptation_res, RESULTS_DIR)

        # 记录结果
        all_results[env_name] = {
            'snn_best_fitness': best_fitness,
            'ppo_rewards': ppo_rewards,
            'adaptation': adaptation_res,
        }

    # 5. 汇总图
    if len(all_results) > 1:
        plot_level_summary(all_results, RESULTS_DIR)

    # 6. 保存 JSON 报告
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {},
    }
    for env_name, res in all_results.items():
        report['results'][env_name] = {
            'snn_best_fitness': float(res['snn_best_fitness']),
            'ppo_mean_reward': float(np.mean(res['ppo_rewards'])) if res.get('ppo_rewards') else None,
            'ppo_std_reward': float(np.std(res['ppo_rewards'])) if res.get('ppo_rewards') else None,
        }
    report_path = os.path.join(RESULTS_DIR, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")
    print("Done.")


if __name__ == '__main__':
    main()
