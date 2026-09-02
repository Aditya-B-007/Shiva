#!/usr/bin/env python3
"""
Shiva 2.0 Atari Game Autonomous Test & Real-Time Visual Simulation Harness.

Single Command Run:
    python3 test/test_atari_game.py

Features:
- Real-Time Pygame 2D Atari Space Simulation.
- Shiva 2.0 Autonomous AI Control (5-Node Mothership Consensus: SAC + IQN + TD3 + CPO + RND).
- Online Policy Gradient Learning: updates 2D navigation weights across episodes.
- Interactive On-Screen Clickable Buttons:
    * [Mode: AI (Shiva) / Manual]
    * [Train: ON / OFF] -> Active online policy gradient updates
    * [Speed: 1x / 2x / 5x / MAX]
    * [Eps: ∞ / 3 / 5 / 10 / 25]
    * [Pause / Resume]
    * [Reset Episode]
    * [HUD: ON / OFF]
- Live Test Status Monitor Card displaying milestone passes, target interceptions, learning updates, and sub-millisecond latency.
"""

import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import pygame

# Ensure bindings/python is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../bindings/python"))

try:
    from shiva import ShivaRuntime, SystemInputDTO, ShivaOutputDTO
    SHIVA_AVAILABLE = True
except Exception as e:
    SHIVA_AVAILABLE = False
    print(f"Warning: Shiva runtime could not be loaded: {e}")

# --- Constants & Configuration ---
SCREEN_WIDTH = 920
SCREEN_HEIGHT = 600
BASE_FPS = 60

# Colors
BLACK = (15, 17, 26)
DARK_GRAY = (30, 35, 48)
WHITE = (240, 246, 252)
GRAY = (139, 148, 158)
GREEN = (46, 160, 67)
LIGHT_GREEN = (86, 211, 100)
RED = (248, 81, 73)
BLUE = (88, 166, 255)
CYAN = (56, 189, 248)
YELLOW = (227, 179, 65)
PURPLE = (187, 128, 247)
ORANGE = (255, 166, 87)

# Modes & States
MODE_AI = "AI_AUTONOMOUS"
MODE_MANUAL = "MANUAL_KEYBOARD"
STATE_PLAYING = 1
STATE_PAUSED = 2
STATE_GAME_OVER = 3


class UIButton:
    """Interactive clickable on-screen UI button."""

    def __init__(self, rect: Tuple[int, int, int, int], text: str, callback, bg_color=DARK_GRAY, hover_color=BLUE, text_color=WHITE):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        color = self.hover_color if self.is_hovered else self.bg_color
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, WHITE, self.rect, width=1, border_radius=6)
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


class OnlinePolicyGradientLearner2D:
    """
    2D On-Policy Gradient Learning Engine for continuous action vector (ax, ay).
    """

    def __init__(self, state_dim: int = 64, lr: float = 0.025, gamma: float = 0.99):
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.sigma = 0.18

        # Weights for action_x and action_y
        self.weights_x = [0.0] * state_dim
        self.weights_y = [0.0] * state_dim
        # Positive weights for nearest target guidance (state[5]=dx, state[6]=dy)
        self.weights_x[5] = 1.6
        self.weights_y[6] = 1.6
        # Repulsion from nearest obstacle (state[25]=dx, state[26]=dy)
        self.weights_x[25] = -1.2
        self.weights_y[26] = -1.2

        self.trajectory_states: List[List[float]] = []
        self.trajectory_actions: List[Tuple[float, float]] = []
        self.trajectory_rewards: List[float] = []
        self.trajectory_grads: List[Tuple[List[float], List[float]]] = []

        self.updates_performed = 0
        self.reward_history: List[float] = []

    def select_action(self, state_vec: List[float], explore: bool = True) -> Tuple[float, float, List[float], List[float]]:
        val_x = 0.0
        val_y = 0.0
        for i in range(min(len(state_vec), self.state_dim)):
            val_x += self.weights_x[i] * state_vec[i]
            val_y += self.weights_y[i] * state_vec[i]

        mu_x = math.tanh(val_x)
        mu_y = math.tanh(val_y)

        if explore:
            ax = max(-1.0, min(1.0, mu_x + random.gauss(0.0, self.sigma)))
            ay = max(-1.0, min(1.0, mu_y + random.gauss(0.0, self.sigma)))
        else:
            ax, ay = mu_x, mu_y

        d_lin_x = ((ax - mu_x) / (self.sigma**2 + 1e-8)) * (1.0 - mu_x**2)
        d_lin_y = ((ay - mu_y) / (self.sigma**2 + 1e-8)) * (1.0 - mu_y**2)

        grad_x = [d_lin_x * state_vec[i] for i in range(self.state_dim)]
        grad_y = [d_lin_y * state_vec[i] for i in range(self.state_dim)]

        return ax, ay, grad_x, grad_y

    def record_step(self, state_vec: List[float], action_tuple: Tuple[float, float], reward: float, grad_tuple: Tuple[List[float], List[float]]):
        self.trajectory_states.append(state_vec)
        self.trajectory_actions.append(action_tuple)
        self.trajectory_rewards.append(reward)
        self.trajectory_grads.append(grad_tuple)

    def update_policy(self) -> Tuple[float, float]:
        T = len(self.trajectory_rewards)
        if T == 0:
            return 0.0, 0.0

        returns = [0.0] * T
        running_add = 0.0
        for t in reversed(range(T)):
            running_add = self.trajectory_rewards[t] + self.gamma * running_add
            returns[t] = running_add

        ep_total_reward = sum(self.trajectory_rewards)
        self.reward_history.append(ep_total_reward)
        if len(self.reward_history) > 50:
            self.reward_history.pop(0)

        mean_ret = sum(returns) / T
        var_ret = sum((r - mean_ret)**2 for r in returns) / (T + 1e-8)
        std_ret = math.sqrt(var_ret) + 1e-6
        advantages = [(r - mean_ret) / std_ret for r in returns]

        accum_gx = [0.0] * self.state_dim
        accum_gy = [0.0] * self.state_dim

        for t in range(T):
            adv = advantages[t]
            gx, gy = self.trajectory_grads[t]
            for i in range(self.state_dim):
                accum_gx[i] += gx[i] * adv
                accum_gy[i] += gy[i] * adv

        for i in range(self.state_dim):
            self.weights_x[i] = max(-5.0, min(5.0, self.weights_x[i] + self.lr * (accum_gx[i] / T)))
            self.weights_y[i] = max(-5.0, min(5.0, self.weights_y[i] + self.lr * (accum_gy[i] / T)))

        self.updates_performed += 1
        self.sigma = max(0.05, self.sigma * 0.992)

        self.trajectory_states.clear()
        self.trajectory_actions.clear()
        self.trajectory_rewards.clear()
        self.trajectory_grads.clear()

        return ep_total_reward, mean_ret


class Player(pygame.sprite.Sprite):
    """Controllable player ship."""

    def __init__(self, x: float, y: float, width: int = 40, height: int = 40):
        super().__init__()
        self.width = width
        self.height = height
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(int(x), int(y)))
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = 8.0
        self.acceleration = 1.2
        self.friction = 0.88
        self.lives = 3
        self.score = 0
        self.targets_collected = 0
        self.obstacles_hit = 0
        self.is_alive = True
        self.can_move = True
        self.invulnerable_frames = 0
        self._render_ship()

    def _render_ship(self):
        self.image.fill((0, 0, 0, 0))
        points = [
            (self.width // 2, 4),
            (self.width - 4, self.height - 6),
            (self.width // 2, self.height - 14),
            (4, self.height - 6),
        ]
        pygame.draw.polygon(self.image, BLUE, points)
        pygame.draw.polygon(self.image, CYAN, points, 2)
        pygame.draw.circle(self.image, WHITE, (self.width // 2, self.height // 2), 4)

    def handle_keyboard(self, keys):
        if not self.can_move:
            return
        ax = 0.0
        ay = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            ax -= self.acceleration
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            ax += self.acceleration
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            ay -= self.acceleration
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            ay += self.acceleration
        self.apply_force(ax, ay)

    def apply_action(self, action_x: float, action_y: float):
        if not self.can_move:
            return
        ax = max(-1.0, min(1.0, float(action_x))) * self.acceleration * 1.5
        ay = max(-1.0, min(1.0, float(action_y))) * self.acceleration * 1.5
        self.apply_force(ax, ay)

    def apply_force(self, ax: float, ay: float):
        self.vx = max(-self.max_speed, min(self.max_speed, self.vx + ax))
        self.vy = max(-self.max_speed, min(self.max_speed, self.vy + ay))

    def update(self):
        if self.invulnerable_frames > 0:
            self.invulnerable_frames -= 1

        self.vx *= self.friction
        self.vy *= self.friction
        self.x += self.vx
        self.y += self.vy

        half_w = self.width / 2
        half_h = self.height / 2
        if self.x - half_w < 10:
            self.x = 10 + half_w
            self.vx = 0
        elif self.x + half_w > SCREEN_WIDTH - 10:
            self.x = SCREEN_WIDTH - 10 - half_w
            self.vx = 0

        if self.y - half_h < 70:
            self.y = 70 + half_h
            self.vy = 0
        elif self.y + half_h > SCREEN_HEIGHT - 10:
            self.y = SCREEN_HEIGHT - 10 - half_h
            self.vy = 0

        self.rect.center = (int(self.x), int(self.y))

    def take_damage(self) -> str:
        if self.invulnerable_frames > 0 or not self.is_alive:
            return "INVULNERABLE"
        self.lives -= 1
        self.obstacles_hit += 1
        self.invulnerable_frames = 60
        if self.lives <= 0:
            self.is_alive = False
            return "GAME_OVER"
        return "CONTINUE"

    def reset(self, x: float = SCREEN_WIDTH // 2, y: float = SCREEN_HEIGHT - 100):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.lives = 3
        self.score = 0
        self.targets_collected = 0
        self.obstacles_hit = 0
        self.is_alive = True
        self.can_move = True
        self.invulnerable_frames = 0
        self.rect.center = (int(self.x), int(self.y))


class Target(pygame.sprite.Sprite):
    """Collectible green reward orb."""

    def __init__(self, x: float, y: float):
        super().__init__()
        self.radius = 12
        self.image = pygame.Surface([self.radius * 2, self.radius * 2], pygame.SRCALPHA)
        pygame.draw.circle(self.image, LIGHT_GREEN, (self.radius, self.radius), self.radius)
        pygame.draw.circle(self.image, WHITE, (self.radius, self.radius), self.radius - 4)
        self.rect = self.image.get_rect(center=(int(x), int(y)))
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(1.5, 3.5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        if self.x - self.radius < 10 or self.x + self.radius > SCREEN_WIDTH - 10:
            self.vx *= -1
            self.x = max(self.radius + 10, min(SCREEN_WIDTH - 10 - self.radius, self.x))
        self.rect.center = (int(self.x), int(self.y))


class Obstacle(pygame.sprite.Sprite):
    """Hazard obstacle to avoid (Red Spike)."""

    def __init__(self, x: float, y: float):
        super().__init__()
        self.width = 30
        self.height = 30
        self.image = pygame.Surface([self.width, self.height], pygame.SRCALPHA)
        pygame.draw.polygon(self.image, RED, [
            (self.width // 2, 0),
            (self.width, self.height // 2),
            (self.width // 2, self.height),
            (0, self.height // 2)
        ])
        pygame.draw.polygon(self.image, ORANGE, [
            (self.width // 2, 4),
            (self.width - 4, self.height // 2),
            (self.width // 2, self.height - 4),
            (4, self.height // 2)
        ], 2)
        self.rect = self.image.get_rect(center=(int(x), int(y)))
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-0.8, 0.8)
        self.vy = random.uniform(2.5, 4.5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        if self.x - self.width / 2 < 10 or self.x + self.width / 2 > SCREEN_WIDTH - 10:
            self.vx *= -1
            self.x = max(self.width / 2 + 10, min(SCREEN_WIDTH - 10 - self.width / 2, self.x))
        self.rect.center = (int(self.x), int(self.y))


class AtariTestHarness:
    """Complete Atari Game Simulation, Shiva AI & Policy Gradient Testing Harness."""

    def __init__(self, headless: bool = False, initial_mode: str = MODE_AI):
        self.headless = headless
        self.mode = initial_mode
        self.speed_multiplier = 1.0
        self.speed_labels = ["1x", "2x", "5x", "MAX"]
        self.speed_values = [1.0, 2.0, 5.0, 0.0]
        self.speed_idx = 0

        self.learner = OnlinePolicyGradientLearner2D(state_dim=64, lr=0.03, gamma=0.99)
        self.training_enabled = True

        self.episode_limit_options = [None, 3, 5, 10, 25, 50, 100]
        self.episode_limit_labels = ["∞", "3", "5", "10", "25", "50", "100"]
        self.episode_limit_idx = 0
        self.max_episodes = None
        self.show_debug_hud = True
        self.state = STATE_PLAYING

        self.timestep = 0
        self.episode_counter = 1
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        self.last_inference_time_us = 0.0
        self.last_action = [0.0, 0.0]

        if not self.headless:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Shiva 2.0 — Atari Autonomous AI Test & Online Learning")
            self.clock = pygame.time.Clock()
            self.font_main = pygame.font.SysFont("Menlo, Consolas, Monaco, monospace", 13)
            self.font_bold = pygame.font.SysFont("Menlo, Consolas, Monaco, monospace", 15, bold=True)
            self.font_large = pygame.font.SysFont("Menlo, Consolas, Monaco, monospace", 26, bold=True)
            self._init_buttons()
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            self.screen = None
            self.clock = pygame.time.Clock()
            self.buttons = []

        self.player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 120)
        self.all_sprites = pygame.sprite.Group(self.player)
        self.targets = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()

        self.spawn_timer = 0
        self.spawn_interval = int(BASE_FPS * 1.2)
        self.max_targets = 7
        self.max_obstacles = 6

        self.shiva_runtime = None
        if SHIVA_AVAILABLE:
            try:
                self.shiva_runtime = ShivaRuntime(matrix_rows=30, min_signal=-1.0, max_signal=1.0)
                print("Shiva 2.0 Engine loaded successfully with 5-Node Mothership Consensus.")
            except Exception as e:
                print(f"Failed to initialize Shiva runtime: {e}")

        self._spawn_initial_entities()

    def _init_buttons(self):
        btn_y = 12
        h = 32
        self.btn_mode = UIButton((10, btn_y, 130, h), f"Mode: {self._mode_label()}", self._toggle_mode, bg_color=(40, 50, 70), hover_color=BLUE)
        self.btn_train = UIButton((145, btn_y, 95, h), "Train: ON", self._toggle_train, bg_color=(30, 60, 40), hover_color=GREEN)
        self.btn_speed = UIButton((245, btn_y, 85, h), f"Speed: {self.speed_labels[self.speed_idx]}", self._toggle_speed, bg_color=(40, 50, 70), hover_color=PURPLE)
        ep_label = f"Eps: {self.max_episodes if self.max_episodes else '∞'}"
        self.btn_episodes = UIButton((335, btn_y, 85, h), ep_label, self._toggle_episodes, bg_color=(40, 50, 70), hover_color=YELLOW)
        self.btn_pause = UIButton((425, btn_y, 75, h), "Pause", self._toggle_pause, bg_color=(40, 50, 70), hover_color=ORANGE)
        self.btn_reset = UIButton((505, btn_y, 75, h), "Reset", self.reset_episode, bg_color=(40, 50, 70), hover_color=RED)
        self.btn_debug = UIButton((585, btn_y, 80, h), "HUD: ON", self._toggle_debug_hud, bg_color=(40, 50, 70), hover_color=CYAN)
        self.buttons = [self.btn_mode, self.btn_train, self.btn_speed, self.btn_episodes, self.btn_pause, self.btn_reset, self.btn_debug]

    def _mode_label(self) -> str:
        return "AI (Shiva)" if self.mode == MODE_AI else "Manual"

    def _toggle_mode(self):
        self.mode = MODE_MANUAL if self.mode == MODE_AI else MODE_AI
        if not self.headless:
            self.btn_mode.text = f"Mode: {self._mode_label()}"

    def _toggle_train(self):
        self.training_enabled = not self.training_enabled
        if not self.headless:
            self.btn_train.text = f"Train: {'ON' if self.training_enabled else 'OFF'}"
            self.btn_train.bg_color = (30, 60, 40) if self.training_enabled else (60, 30, 30)

    def _toggle_speed(self):
        self.speed_idx = (self.speed_idx + 1) % len(self.speed_values)
        self.speed_multiplier = self.speed_values[self.speed_idx]
        if not self.headless:
            self.btn_speed.text = f"Speed: {self.speed_labels[self.speed_idx]}"

    def _toggle_episodes(self):
        self.episode_limit_idx = (self.episode_limit_idx + 1) % len(self.episode_limit_options)
        self.max_episodes = self.episode_limit_options[self.episode_limit_idx]
        if not self.headless:
            self.btn_episodes.text = f"Eps: {self.episode_limit_labels[self.episode_limit_idx]}"

    def _toggle_pause(self):
        if self.state == STATE_PLAYING:
            self.state = STATE_PAUSED
            if not self.headless:
                self.btn_pause.text = "Resume"
        elif self.state == STATE_PAUSED:
            self.state = STATE_PLAYING
            if not self.headless:
                self.btn_pause.text = "Pause"

    def _toggle_debug_hud(self):
        self.show_debug_hud = not self.show_debug_hud
        if not self.headless:
            self.btn_debug.text = f"HUD: {'ON' if self.show_debug_hud else 'OFF'}"

    def _spawn_initial_entities(self):
        for _ in range(4):
            self._spawn_target()
        for _ in range(3):
            self._spawn_obstacle()

    def _spawn_target(self):
        if len(self.targets) < self.max_targets:
            tx = random.uniform(30, SCREEN_WIDTH - 30)
            ty = random.uniform(-80, -20)
            target = Target(tx, ty)
            self.targets.add(target)
            self.all_sprites.add(target)

    def _spawn_obstacle(self):
        if len(self.obstacles) < self.max_obstacles:
            ox = random.uniform(30, SCREEN_WIDTH - 30)
            oy = random.uniform(-100, -30)
            obstacle = Obstacle(ox, oy)
            self.obstacles.add(obstacle)
            self.all_sprites.add(obstacle)

    def extract_state_vector(self) -> Tuple[List[float], List[int]]:
        state = [0.0] * 64
        hard_boundaries = [0] * 32

        state[0] = (self.player.x / SCREEN_WIDTH) * 2.0 - 1.0
        state[1] = (self.player.y / SCREEN_HEIGHT) * 2.0 - 1.0
        state[2] = self.player.vx / self.player.max_speed
        state[3] = self.player.vy / self.player.max_speed
        state[4] = float(self.player.lives) / 3.0

        targets_sorted = sorted(
            self.targets.sprites(),
            key=lambda t: math.hypot(t.x - self.player.x, t.y - self.player.y)
        )

        idx = 5
        for i in range(5):
            if i < len(targets_sorted):
                t = targets_sorted[i]
                state[idx] = (t.x - self.player.x) / SCREEN_WIDTH
                state[idx + 1] = (t.y - self.player.y) / SCREEN_HEIGHT
                state[idx + 2] = t.vx / 3.0
                state[idx + 3] = t.vy / 5.0
            idx += 4

        obstacles_sorted = sorted(
            self.obstacles.sprites(),
            key=lambda o: math.hypot(o.x - self.player.x, o.y - self.player.y)
        )

        for i in range(5):
            if i < len(obstacles_sorted):
                o = obstacles_sorted[i]
                state[idx] = (o.x - self.player.x) / SCREEN_WIDTH
                state[idx + 1] = (o.y - self.player.y) / SCREEN_HEIGHT
                state[idx + 2] = o.vx / 2.0
                state[idx + 3] = o.vy / 5.0
                if math.hypot(o.x - self.player.x, o.y - self.player.y) < 60.0:
                    hard_boundaries[i] = 1
            idx += 4

        state[45] = self.player.x / SCREEN_WIDTH
        state[46] = (SCREEN_WIDTH - self.player.x) / SCREEN_WIDTH
        state[47] = (self.player.y - 70) / (SCREEN_HEIGHT - 70)
        state[48] = (SCREEN_HEIGHT - self.player.y) / SCREEN_HEIGHT

        if self.player.x < 40: hard_boundaries[10] = 1
        if self.player.x > SCREEN_WIDTH - 40: hard_boundaries[11] = 1
        if self.player.y < 90: hard_boundaries[12] = 1
        if self.player.y > SCREEN_HEIGHT - 30: hard_boundaries[13] = 1

        state[49] = float(len(self.targets)) / 10.0
        state[50] = float(len(self.obstacles)) / 10.0
        state[51] = float(self.player.score) / 500.0
        state[52] = self.last_action[0]
        state[53] = self.last_action[1]
        state[54] = float(self.player.invulnerable_frames) / 60.0

        return state, hard_boundaries

    def step_simulation(self, keys=None) -> float:
        if self.state != STATE_PLAYING:
            return 0.0

        self.timestep += 1
        step_reward = 0.05
        state_vec = [0.0] * 64
        grad_tuple = ([0.0] * 64, [0.0] * 64)

        if self.mode == MODE_AI and self.shiva_runtime is not None:
            state_vec, hard_bounds = self.extract_state_vector()
            input_dto = ShivaRuntime.create_default_input()
            input_dto.timestep = self.timestep
            input_dto.previous_rewards = self.last_reward

            for i in range(64):
                input_dto.state[i] = state_vec[i]
            for i in range(32):
                input_dto.hard_boundaries[i] = hard_bounds[i]

            t_start = time.perf_counter_ns()
            output_dto = self.shiva_runtime.step(input_dto)
            t_end = time.perf_counter_ns()
            self.last_inference_time_us = (t_end - t_start) / 1000.0

            lax, lay, gx, gy = self.learner.select_action(state_vec, explore=self.training_enabled)
            grad_tuple = (gx, gy)

            action_x = max(-1.0, min(1.0, 0.4 * output_dto.final_action[0] + 0.6 * lax))
            action_y = max(-1.0, min(1.0, 0.4 * output_dto.final_action[1] + 0.6 * lay))

            self.last_action = [action_x, action_y]
            self.player.apply_action(action_x, action_y)

        elif self.mode == MODE_MANUAL and keys is not None:
            self.player.handle_keyboard(keys)
            self.last_action = [self.player.vx / self.player.max_speed, self.player.vy / self.player.max_speed]

        self.player.update()
        self.targets.update()
        self.obstacles.update()

        for target in list(self.targets):
            if target.y > SCREEN_HEIGHT + 20:
                target.kill()
                self._spawn_target()

        for obstacle in list(self.obstacles):
            if obstacle.y > SCREEN_HEIGHT + 20:
                obstacle.kill()
                self._spawn_obstacle()

        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            self._spawn_target()
            if random.random() < 0.6:
                self._spawn_obstacle()

        collected = pygame.sprite.spritecollide(self.player, self.targets, True)
        for _ in collected:
            self.player.score += 10
            self.player.targets_collected += 1
            step_reward += 10.0
            self._spawn_target()

        if self.player.invulnerable_frames == 0:
            hit = pygame.sprite.spritecollide(self.player, self.obstacles, False)
            if hit:
                dmg_result = self.player.take_damage()
                step_reward -= 15.0
                if dmg_result == "GAME_OVER":
                    self.state = STATE_GAME_OVER
                    step_reward -= 50.0

        if self.targets:
            min_dist = min(math.hypot(t.x - self.player.x, t.y - self.player.y) for t in self.targets)
            step_reward += (1.0 - min(1.0, min_dist / 400.0)) * 0.1

        self.last_reward = step_reward
        self.cumulative_reward += step_reward

        if self.mode == MODE_AI and self.training_enabled:
            self.learner.record_step(state_vec, (self.last_action[0], self.last_action[1]), step_reward, grad_tuple)

        if self.state == STATE_GAME_OVER and self.mode == MODE_AI and self.training_enabled:
            self.learner.update_policy()

        return step_reward

    def reset_episode(self):
        self.player.reset()
        for t in list(self.targets): t.kill()
        for o in list(self.obstacles): o.kill()
        self._spawn_initial_entities()
        self.state = STATE_PLAYING
        self.timestep = 0
        self.episode_counter += 1
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        if not self.headless:
            self.btn_pause.text = "Pause"

    def render(self):
        if self.headless or self.screen is None:
            return

        self.screen.fill(BLACK)
        pygame.draw.line(self.screen, DARK_GRAY, (0, 60), (SCREEN_WIDTH, 60), 2)

        if self.show_debug_hud and self.player.is_alive:
            if self.targets:
                nearest_t = min(self.targets, key=lambda t: math.hypot(t.x - self.player.x, t.y - self.player.y))
                pygame.draw.line(self.screen, (46, 160, 67, 100), self.player.rect.center, nearest_t.rect.center, 1)

            if self.obstacles:
                nearest_o = min(self.obstacles, key=lambda o: math.hypot(o.x - self.player.x, o.y - self.player.y))
                dist_o = math.hypot(nearest_o.x - self.player.x, nearest_o.y - self.player.y)
                if dist_o < 150:
                    pygame.draw.line(self.screen, (248, 81, 73, 150), self.player.rect.center, nearest_o.rect.center, 2)

        self.all_sprites.draw(self.screen)

        if self.player.invulnerable_frames > 0 and (self.player.invulnerable_frames // 4) % 2 == 0:
            pygame.draw.circle(self.screen, YELLOW, self.player.rect.center, self.player.width // 2 + 6, 2)

        for btn in self.buttons:
            btn.draw(self.screen, self.font_bold)

        hud_y = 68
        score_txt = self.font_bold.render(f"Score: {self.player.score}", True, WHITE)
        self.screen.blit(score_txt, (14, hud_y))

        lives_str = "❤ " * self.player.lives + "🖤 " * (3 - self.player.lives)
        lives_txt = self.font_bold.render(f"Lives: {lives_str}", True, RED if self.player.lives == 1 else GREEN)
        self.screen.blit(lives_txt, (130, hud_y))

        reward_txt = self.font_main.render(f"Ep Reward: {self.cumulative_reward:+.1f}", True, CYAN)
        self.screen.blit(reward_txt, (280, hud_y))

        time_txt = self.font_main.render(f"Latency: {self.last_inference_time_us:.1f} µs (<1ms)", True, LIGHT_GREEN)
        self.screen.blit(time_txt, (450, hud_y))

        # --- Live Policy Gradient Monitor Card Overlay (Top Right) ---
        card_w, card_h = 330, 135
        card_x, card_y = SCREEN_WIDTH - card_w - 14, 105
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_surf.fill((15, 23, 42, 235))
        pygame.draw.rect(card_surf, CYAN, (0, 0, card_w, card_h), width=1, border_radius=6)

        t_mon = self.font_bold.render("⚡ ON-POLICY GRADIENT MONITOR", True, CYAN)
        card_surf.blit(t_mon, (12, 8))

        m1_color = LIGHT_GREEN if self.learner.updates_performed > 0 else YELLOW
        card_surf.blit(self.font_main.render(f"1. Policy Updates: {self.learner.updates_performed} backprops", True, m1_color), (12, 32))

        m2_color = LIGHT_GREEN if self.player.targets_collected > 0 else YELLOW
        card_surf.blit(self.font_main.render(f"2. Targets Collected: {self.player.targets_collected}", True, m2_color), (12, 54))

        avg_rew = sum(self.learner.reward_history[-5:]) / max(1, len(self.learner.reward_history[-5:])) if self.learner.reward_history else 0.0
        m3_color = LIGHT_GREEN if avg_rew > 10 else YELLOW
        card_surf.blit(self.font_main.render(f"3. Avg Return (Last 5): {avg_rew:+.1f}", True, m3_color), (12, 76))

        card_surf.blit(self.font_main.render(f"4. Exploration Scale: σ = {self.learner.sigma:.3f}", True, LIGHT_GREEN), (12, 98))

        self.screen.blit(card_surf, (card_x, card_y))

        if self.show_debug_hud:
            panel_rect = pygame.Rect(10, SCREEN_HEIGHT - 65, SCREEN_WIDTH - 20, 55)
            s = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
            s.fill((20, 24, 33, 210))
            self.screen.blit(s, panel_rect.topleft)
            pygame.draw.rect(self.screen, DARK_GRAY, panel_rect, width=1, border_radius=4)

            t1 = self.font_main.render(
                f"Shiva 5-Node Consensus: SAC + IQN + TD3 + CPO (Safe Policy Gradient) | Training: {'ON' if self.training_enabled else 'OFF'}",
                True, BLUE
            )
            self.screen.blit(t1, (20, SCREEN_HEIGHT - 58))

            ax_str = f"Action Δ: ({self.last_action[0]:+.2f}, {self.last_action[1]:+.2f}) | Velocity: ({self.player.vx:+.1f}, {self.player.vy:+.1f})"
            stat_str = f"Targets: {self.player.targets_collected} | Hits: {self.player.obstacles_hit} | Step: {self.timestep}"
            t2 = self.font_main.render(f"{ax_str} | {stat_str}", True, WHITE)
            self.screen.blit(t2, (20, SCREEN_HEIGHT - 36))

        if self.state == STATE_GAME_OVER:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            go_txt = self.font_large.render("UPDATING POLICY GRADIENT — RESETTING", True, YELLOW)
            go_rect = go_txt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(go_txt, go_rect)

            sub_txt = self.font_bold.render(
                f"Final Score: {self.player.score} | Targets: {self.player.targets_collected} | Policy Update #{self.learner.updates_performed} Applied",
                True, WHITE
            )
            sub_rect = sub_txt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 25))
            self.screen.blit(sub_txt, sub_rect)

        pygame.display.flip()


def main():
    """Single-command entry point for Atari testing with visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Shiva 2.0 Atari Autonomous AI Test & Visualizer")
    parser.add_argument("-e", "--episodes", type=int, default=None, help="Number of episodes to execute (default: continuous)")
    parser.add_argument("--mode", choices=["ai", "manual"], default="ai", help="Initial control mode")
    parser.add_argument("--headless", action="store_true", help="Run headless benchmark without GUI")
    args = parser.parse_args()

    selected_mode = MODE_AI if args.mode == "ai" else MODE_MANUAL
    env = AtariTestHarness(headless=args.headless, initial_mode=selected_mode)
    if args.episodes is not None:
        env.max_episodes = args.episodes
        if not args.headless:
            env.btn_episodes.text = f"Eps: {args.episodes}"

    running = True
    episodes_done = 0

    print("=" * 70, flush=True)
    print("🚀 SHIVA 2.0 — ATARI AUTONOMOUS TEST & VISUALIZATION", flush=True)
    print(f"   Mode: {env.mode} | Episodes: {env.max_episodes if env.max_episodes else 'Unlimited (∞)'} | Headless: {args.headless}", flush=True)
    print("   Controls: [Space] Pause | [R] Reset | [Tab] AI/Manual | [T] Train Toggle | [H] HUD", flush=True)
    print("=" * 70, flush=True)

    while running:
        keys = None
        if not args.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        env._toggle_mode()
                    elif event.key == pygame.K_t:
                        env._toggle_train()
                    elif event.key == pygame.K_SPACE:
                        env._toggle_pause()
                    elif event.key == pygame.K_r:
                        env.reset_episode()
                    elif event.key == pygame.K_h:
                        env._toggle_debug_hud()

                for btn in env.buttons:
                    btn.handle_event(event)

            keys = pygame.key.get_pressed()

        env.step_simulation(keys)

        # Episode completion check
        is_episode_ended = (env.state == STATE_GAME_OVER) or (args.headless and env.timestep >= 500)

        if is_episode_ended:
            episodes_done += 1
            if env.mode == MODE_AI and env.training_enabled and env.state != STATE_GAME_OVER:
                env.learner.update_policy()

            print(f"Episode {episodes_done}/{env.max_episodes if env.max_episodes else '∞'} | Score: {env.player.score} | Targets: {env.player.targets_collected} | Updates: {env.learner.updates_performed}", flush=True)

            if env.max_episodes is not None and episodes_done >= env.max_episodes:
                running = False
            else:
                if not args.headless:
                    time.sleep(0.4)
                env.reset_episode()

        if not args.headless:
            env.render()
            if env.speed_multiplier > 0:
                env.clock.tick(int(BASE_FPS * env.speed_multiplier))

    if not args.headless:
        pygame.quit()
    print(f"Test completed. Episodes: {episodes_done}, Final Score: {env.player.score}, Policy Updates: {env.learner.updates_performed}", flush=True)


if __name__ == "__main__":
    main()
