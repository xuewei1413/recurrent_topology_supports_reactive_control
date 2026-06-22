"""
Flappy Bird 环境 — 纯 Python 实现，无需 gymnasium 依赖

状态空间（4维，对应 SNNController.encode_state 的输入）：
  [bird_y_norm, bird_vy_norm, pipe_dx_norm, pipe_gap_cy_norm]

动作空间：
  0 = 不扇翅（自由落体）
  1 = 扇翅（向上跳）
"""

import numpy as np

# ── 游戏常量 ──────────────────────────────────────────────────────────────
SCREEN_W      = 288
SCREEN_H      = 512
GRAVITY       = 0.45
FLAP_VEL      = -8.5
MAX_VEL       = 10.0
MIN_VEL       = -10.0
BIRD_X        = 60
BIRD_R        = 10       # 碰撞半径（像素）
PIPE_W        = 52
PIPE_GAP      = 140      # 管道上下间隙高度
PIPE_SPEED    = 3        # 管道移动速度（像素/帧）
PIPE_INTERVAL = 100      # 每隔多少帧生成新管道
MARGIN        = 20       # 管道顶/底距边界最小距离


class FlappyBirdEnv:
    """
    Flappy Bird 无头环境。

    用法：
        env = FlappyBirdEnv()
        state = env.reset()
        while True:
            action = controller(state)   # 0 或 1
            state, reward, done, info = env.step(action)
            if done: break
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ── 核心接口 ──────────────────────────────────────────────────────────

    def reset(self):
        self.bird_y  = float(SCREEN_H // 2)
        self.bird_vy = 0.0
        self.frame   = 0
        self.score   = 0
        self.pipes   = []
        self._spawn_pipe()
        return self._get_state()

    def step(self, action: int):
        """
        返回 (state, reward, done, info)
        reward = 1.0 每存活一帧；通过管道额外 +50
        """
        # 扇翅
        if action == 1:
            self.bird_vy = FLAP_VEL

        # 物理更新
        self.bird_vy = min(max(self.bird_vy + GRAVITY, MIN_VEL), MAX_VEL)
        self.bird_y += self.bird_vy

        # 移动管道
        for p in self.pipes:
            p['x'] -= PIPE_SPEED

        # 生成新管道
        self.frame += 1
        if self.frame % PIPE_INTERVAL == 0:
            self._spawn_pipe()

        # 检测得分（飞过管道）
        bonus = 0
        for p in self.pipes:
            if not p.get('passed', False) and p['x'] + PIPE_W < BIRD_X:
                p['passed'] = True
                self.score += 1
                bonus = 50

        # 移除屏幕外管道
        self.pipes = [p for p in self.pipes if p['x'] > -PIPE_W - 20]

        # 碰撞检测
        done = self._check_collision()

        # 间隙对齐奖励：鸟距当前管道间隙中心越近越高（归一化到 [0,1]）
        upcoming = [p for p in self.pipes if p['x'] + PIPE_W > BIRD_X - BIRD_R]
        if upcoming:
            nearest = min(upcoming, key=lambda p: p['x'])
            gap_dist_norm = abs(self.bird_y - nearest['gap_cy']) / (SCREEN_H / 2)
        else:
            gap_dist_norm = 1.0
        gap_align = max(0.0, 1.0 - gap_dist_norm)  # 0=偏离最远, 1=正中间隙

        reward = 1.0 + bonus
        return self._get_state(), reward, done, {
            'score'    : self.score,
            'frame'    : self.frame,
            'gap_align': gap_align,   # 新增：间隙对齐度
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _spawn_pipe(self):
        gap_cy = int(self.rng.integers(
            MARGIN + PIPE_GAP // 2,
            SCREEN_H - MARGIN - PIPE_GAP // 2
        ))
        self.pipes.append({'x': SCREEN_W + PIPE_W, 'gap_cy': gap_cy})

    def _get_state(self) -> np.ndarray:
        """
        归一化 4 维状态，范围约 [-1, 1]

        第4维从"间隙绝对高度"改为"鸟相对间隙中心的偏差"：
          正值 = 鸟在间隙上方（需要少扇翅下落）
          负值 = 鸟在间隙下方（需要扇翅上升）
          0   = 正中间隙
        这让 SNN 无需内部做减法即可直接响应，降低任务难度。
        """
        # 找最近的前方管道
        upcoming = [p for p in self.pipes if p['x'] + PIPE_W > BIRD_X - BIRD_R]
        if upcoming:
            pipe = min(upcoming, key=lambda p: p['x'])
        else:
            pipe = {'x': SCREEN_W, 'gap_cy': SCREEN_H // 2}

        bird_y_norm      = (self.bird_y / SCREEN_H) * 2.0 - 1.0
        bird_vy_norm     = self.bird_vy / MAX_VEL
        pipe_dx_norm     = (pipe['x'] - BIRD_X) / SCREEN_W
        # 关键修改：相对偏差，正=在间隙上方，负=在间隙下方
        gap_offset_norm  = (self.bird_y - pipe['gap_cy']) / (SCREEN_H / 2)

        return np.array([bird_y_norm, bird_vy_norm, pipe_dx_norm, gap_offset_norm],
                        dtype=np.float64)

    def _check_collision(self) -> bool:
        # 边界
        if self.bird_y - BIRD_R <= 0 or self.bird_y + BIRD_R >= SCREEN_H:
            return True
        # 管道
        for p in self.pipes:
            pipe_left  = p['x']
            pipe_right = p['x'] + PIPE_W
            gap_top    = p['gap_cy'] - PIPE_GAP // 2
            gap_bottom = p['gap_cy'] + PIPE_GAP // 2
            if (BIRD_X + BIRD_R > pipe_left and BIRD_X - BIRD_R < pipe_right):
                if (self.bird_y - BIRD_R < gap_top or
                        self.bird_y + BIRD_R > gap_bottom):
                    return True
        return False

    # ── 渲染辅助（供 demo 使用）──────────────────────────────────────────

    def get_render_state(self) -> dict:
        """返回供 pygame 渲染的原始游戏状态"""
        return {
            'bird_y' : self.bird_y,
            'bird_x' : BIRD_X,
            'bird_r' : BIRD_R,
            'pipes'  : [(p['x'], p['gap_cy']) for p in self.pipes],
            'pipe_w' : PIPE_W,
            'pipe_gap': PIPE_GAP,
            'screen_w': SCREEN_W,
            'screen_h': SCREEN_H,
            'score'  : self.score,
            'frame'  : self.frame,
        }
