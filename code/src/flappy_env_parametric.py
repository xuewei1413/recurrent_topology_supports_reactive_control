"""
Flappy Bird 环境 — 参数化版本
基于 flappy_env.py 的 FlappyBirdEnv 修改，所有核心环境常量可通过构造函数参数调整。

状态空间（4维）：
  [bird_y_norm, bird_vy_norm, pipe_dx_norm, pipe_gap_cy_norm]

动作空间：
  0 = 不扇翅（自由落体）
  1 = 扇翅（向上跳）
"""

import numpy as np


class FlappyEnvParametric:
    """
    参数化 Flappy Bird 无头环境。

    用法：
        env = FlappyEnvParametric(pipe_speed=4.0, gravity=1.25,
                                   pipe_gap=100, pipe_frequency=150)
        state = env.reset()
        while True:
            action = controller(state)   # 0 或 1
            state, reward, done, info = env.step(action)
            if done:
                break
    """

    def __init__(self, pipe_speed=4.0, gravity=1.25, pipe_gap=100,
                 pipe_frequency=150, seed=None):
        # ── 可调节参数 ──────────────────────────────────────────────────
        self.pipe_speed = float(pipe_speed)
        self.gravity = float(gravity)
        self.pipe_gap = int(pipe_gap)
        self.pipe_frequency = int(pipe_frequency)

        # ── 固定渲染/物理常量（通常不需要调整）───────────────────────────
        self.screen_w = 288
        self.screen_h = 512
        self.flap_vel = -8.5
        self.max_vel = 10.0
        self.min_vel = -10.0
        self.bird_x = 60
        self.bird_r = 10
        self.pipe_w = 52
        self.margin = 20

        self.rng = np.random.default_rng(seed)
        self.reset()

    # ── 核心接口 ──────────────────────────────────────────────────────────

    def reset(self):
        self.bird_y = float(self.screen_h // 2)
        self.bird_vy = 0.0
        self.frame = 0
        self.score = 0
        self.pipes = []
        self._spawn_pipe()
        return self._get_state()

    def step(self, action: int):
        """
        返回 (state, reward, done, info)
        reward = 1.0 每存活一帧；通过管道额外 +50
        """
        # 扇翅
        if action == 1:
            self.bird_vy = self.flap_vel

        # 物理更新
        self.bird_vy = min(max(self.bird_vy + self.gravity, self.min_vel), self.max_vel)
        self.bird_y += self.bird_vy

        # 移动管道
        for p in self.pipes:
            p['x'] -= self.pipe_speed

        # 生成新管道
        self.frame += 1
        if self.frame % self.pipe_frequency == 0:
            self._spawn_pipe()

        # 检测得分（飞过管道）
        bonus = 0
        for p in self.pipes:
            if not p.get('passed', False) and p['x'] + self.pipe_w < self.bird_x:
                p['passed'] = True
                self.score += 1
                bonus = 50

        # 移除屏幕外管道
        self.pipes = [p for p in self.pipes if p['x'] > -self.pipe_w - 20]

        # 碰撞检测
        done = self._check_collision()

        # 间隙对齐奖励
        upcoming = [p for p in self.pipes if p['x'] + self.pipe_w > self.bird_x - self.bird_r]
        if upcoming:
            nearest = min(upcoming, key=lambda p: p['x'])
            gap_dist_norm = abs(self.bird_y - nearest['gap_cy']) / (self.screen_h / 2)
        else:
            gap_dist_norm = 1.0
        gap_align = max(0.0, 1.0 - gap_dist_norm)

        reward = 1.0 + bonus
        return self._get_state(), reward, done, {
            'score': self.score,
            'frame': self.frame,
            'gap_align': gap_align,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _spawn_pipe(self):
        half_gap = self.pipe_gap // 2
        gap_cy = int(self.rng.integers(
            self.margin + half_gap,
            self.screen_h - self.margin - half_gap
        ))
        self.pipes.append({'x': self.screen_w + self.pipe_w, 'gap_cy': gap_cy})

    def _get_state(self) -> np.ndarray:
        """
        归一化 4 维状态，范围约 [-1, 1]
        """
        upcoming = [p for p in self.pipes if p['x'] + self.pipe_w > self.bird_x - self.bird_r]
        if upcoming:
            pipe = min(upcoming, key=lambda p: p['x'])
        else:
            pipe = {'x': self.screen_w, 'gap_cy': self.screen_h // 2}

        bird_y_norm = (self.bird_y / self.screen_h) * 2.0 - 1.0
        bird_vy_norm = self.bird_vy / self.max_vel
        pipe_dx_norm = (pipe['x'] - self.bird_x) / self.screen_w
        gap_offset_norm = (self.bird_y - pipe['gap_cy']) / (self.screen_h / 2)

        return np.array([bird_y_norm, bird_vy_norm, pipe_dx_norm, gap_offset_norm],
                        dtype=np.float64)

    def _check_collision(self) -> bool:
        if self.bird_y - self.bird_r <= 0 or self.bird_y + self.bird_r >= self.screen_h:
            return True
        for p in self.pipes:
            pipe_left = p['x']
            pipe_right = p['x'] + self.pipe_w
            half_gap = self.pipe_gap // 2
            gap_top = p['gap_cy'] - half_gap
            gap_bottom = p['gap_cy'] + half_gap
            if (self.bird_x + self.bird_r > pipe_left and self.bird_x - self.bird_r < pipe_right):
                if (self.bird_y - self.bird_r < gap_top or
                        self.bird_y + self.bird_r > gap_bottom):
                    return True
        return False

    # ── 渲染辅助（供 demo 使用）──────────────────────────────────────────

    def get_render_state(self) -> dict:
        """返回供 pygame 渲染的原始游戏状态"""
        return {
            'bird_y': self.bird_y,
            'bird_x': self.bird_x,
            'bird_r': self.bird_r,
            'pipes': [(p['x'], p['gap_cy']) for p in self.pipes],
            'pipe_w': self.pipe_w,
            'pipe_gap': self.pipe_gap,
            'screen_w': self.screen_w,
            'screen_h': self.screen_h,
            'score': self.score,
            'frame': self.frame,
        }
