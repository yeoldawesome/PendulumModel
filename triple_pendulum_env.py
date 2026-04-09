import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry

TRIPLE_PENDULUM_ENV_ID = "InvertedTriplePendulum-v0"


class InvertedTriplePendulumEnv(gym.Env[np.ndarray, np.ndarray]):
    """A lightweight triple-pendulum-on-cart environment with continuous force control.

    The state contains cart position/velocity and three link angles/velocities.
    Observation format follows the MuJoCo inverted pendulum style using sin/cos angles.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int = 1000):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)

        self.dt = 0.02
        self.gravity = 9.81
        self.step_count = 0
        self.survival_streak = 0

        # Balance-first shaping: maximize upright hold time with dense progress reward.
        self.terminal_penalty = 150.0
        self.survival_bonus_interval = 25
        self.survival_bonus_value = 0.12

        # Internal state: x, x_dot, theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot
        self.state = np.zeros(8, dtype=np.float32)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_high = np.array(
            [
                np.inf,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        x, x_dot, t1, t2, t3, t1_dot, t2_dot, t3_dot = self.state
        return np.array(
            [
                x,
                np.sin(t1),
                np.sin(t2),
                np.sin(t3),
                np.cos(t1),
                np.cos(t2),
                np.cos(t3),
                x_dot,
                t1_dot,
                t2_dot,
                t3_dot,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        # Start close to upright with a small randomized perturbation.
        self.state = self.np_random.normal(loc=0.0, scale=0.04, size=(8,)).astype(np.float32)
        self.step_count = 0
        self.survival_streak = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1

        x, x_dot, t1, t2, t3, t1_dot, t2_dot, t3_dot = [float(v) for v in self.state]
        force = float(np.clip(action, -1.0, 1.0).reshape(-1)[0]) * 15.0

        # Simple coupled dynamics to emulate a cart with three connected inverted links.
        cart_acc = 0.6 * force - 0.2 * x_dot - 0.35 * (math.sin(t1) + math.sin(t2) + math.sin(t3))

        t1_acc = 1.5 * self.gravity * math.sin(t1) + 0.25 * cart_acc - 0.12 * t1_dot + 0.20 * (t2 - t1)
        t2_acc = 1.6 * self.gravity * math.sin(t2) + 0.20 * cart_acc - 0.12 * t2_dot + 0.16 * (t1 - 2.0 * t2 + t3)
        t3_acc = 1.7 * self.gravity * math.sin(t3) + 0.15 * cart_acc - 0.12 * t3_dot + 0.20 * (t2 - t3)

        x_dot += cart_acc * self.dt
        t1_dot += t1_acc * self.dt
        t2_dot += t2_acc * self.dt
        t3_dot += t3_acc * self.dt

        x += x_dot * self.dt
        t1 += t1_dot * self.dt
        t2 += t2_dot * self.dt
        t3 += t3_dot * self.dt

        t1 = ((t1 + math.pi) % (2.0 * math.pi)) - math.pi
        t2 = ((t2 + math.pi) % (2.0 * math.pi)) - math.pi
        t3 = ((t3 + math.pi) % (2.0 * math.pi)) - math.pi

        self.state = np.array([x, x_dot, t1, t2, t3, t1_dot, t2_dot, t3_dot], dtype=np.float32)

        # Hybrid reward: dense upright progress + centered cart + small streak milestone bonuses.
        upright_score = (math.cos(t1) + math.cos(t2) + math.cos(t3)) / 3.0
        reward = 0.9 * upright_score
        reward -= 0.04 * abs(x)

        tight_upright = abs(t1) < 0.25 and abs(t2) < 0.25 and abs(t3) < 0.25 and abs(x) < 0.8
        if tight_upright:
            reward += 0.35

        self.survival_streak += 1
        if self.survival_streak % self.survival_bonus_interval == 0:
            reward += self.survival_bonus_value

        terminated = bool(abs(x) > 2.4 or abs(t1) > 2.75 or abs(t2) > 2.75 or abs(t3) > 2.75)
        if terminated:
            reward -= self.terminal_penalty
            self.survival_streak = 0

        truncated = self.step_count >= self.max_episode_steps
        if truncated:
            self.survival_streak = 0
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        return None


def register_triple_pendulum_env() -> None:
    if TRIPLE_PENDULUM_ENV_ID in registry:
        return

    register(
        id=TRIPLE_PENDULUM_ENV_ID,
        entry_point=InvertedTriplePendulumEnv,
        max_episode_steps=1000,
    )
