"""
Minimal Snake environment for topology evolution experiments.

Grid: 10×10. The snake moves in 4 directions. Food spawns randomly.
Observation (8D): head_x, head_y, food_dx, food_dy,
                  tail_dir_x, tail_dir_y, body_len_norm, danger_ahead
                  → encoded as 16 input neurons (half-wave rectification).

Actions: 0=up, 1=right, 2=down, 3=left
Reward: +1 per food eaten, small step penalty to encourage efficiency.
"""
import numpy as np


class SnakeEnv:
    SIZE = 10
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left

    def __init__(self):
        self.reset()

    def reset(self):
        cx, cy = self.SIZE // 2, self.SIZE // 2
        self.snake = [(cx, cy)]
        self.direction = 1  # right
        self.food = self._spawn_food()
        self.done = False
        self.score = 0
        self.steps = 0
        return self._get_state()

    def _spawn_food(self):
        empty = [(x, y) for x in range(self.SIZE) for y in range(self.SIZE)
                 if (x, y) not in self.snake]
        if not empty:
            return self.snake[-1]
        return empty[np.random.randint(len(empty))]

    def _get_state(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        state = [
            hx / self.SIZE,
            hy / self.SIZE,
            (fx - hx) / self.SIZE,
            (fy - hy) / self.SIZE,
        ]
        # tail direction (normalized)
        if len(self.snake) > 1:
            tx, ty = self.snake[-1]
            px, py = self.snake[-2] if len(self.snake) > 2 else self.snake[-1]
            state.extend([(tx - px), (ty - py)])
        else:
            state.extend([0.0, 0.0])
        # body length normalized
        state.append(len(self.snake) / 20.0)
        # danger ahead (1 if wall or body in next step)
        dx, dy = self.DIRECTIONS[self.direction]
        nx, ny = hx + dx, hy + dy
        danger = 1.0 if (nx < 0 or nx >= self.SIZE or ny < 0 or ny >= self.SIZE
                         or (nx, ny) in self.snake[:-1]) else 0.0
        state.append(danger)
        return np.array(state, dtype=np.float64)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {'score': self.score, 'steps': self.steps}
        self.direction = int(action) % 4
        hx, hy = self.snake[0]
        dx, dy = self.DIRECTIONS[self.direction]
        nx, ny = hx + dx, hy + dy
        self.steps += 1

        # wall collision
        if nx < 0 or nx >= self.SIZE or ny < 0 or ny >= self.SIZE:
            self.done = True
            return self._get_state(), -1.0, True, {'score': self.score, 'steps': self.steps}
        # self collision
        if (nx, ny) in self.snake[:-1]:
            self.done = True
            return self._get_state(), -1.0, True, {'score': self.score, 'steps': self.steps}

        self.snake.insert(0, (nx, ny))
        if (nx, ny) == self.food:
            self.score += 1
            self.food = self._spawn_food()
            reward = 1.0
        else:
            self.snake.pop()
            reward = -0.01

        if self.steps >= 1000:
            self.done = True

        return self._get_state(), reward, self.done, {'score': self.score, 'steps': self.steps}

    def render(self):
        grid = [['.' for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        fx, fy = self.food
        grid[fy][fx] = 'F'
        for i, (x, y) in enumerate(self.snake):
            grid[y][x] = 'H' if i == 0 else 'S'
        for row in grid:
            print(''.join(row))
        print()
