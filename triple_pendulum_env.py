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

        # Reward shaping tuned for sustained upright balance, even with aggressive corrections.
        self.base_reward = 10.0
        self.terminal_penalty = 100.0
        self.stability_bonus = 1.5

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

        # Use (1 - cos(theta)) so reward is smooth near upright and periodic over angle wrap.
        angle_cost = 8.0 * (1.0 - math.cos(t1)) + 10.0 * (1.0 - math.cos(t2)) + 12.0 * (1.0 - math.cos(t3))
        cart_cost = 1.8 * (x**2)
        # Keep a light velocity penalty to permit high-gain corrections when needed.
        vel_cost = 0.015 * (x_dot**2 + t1_dot**2 + t2_dot**2 + t3_dot**2)
        action_cost = 0.01 * (force**2)

        is_stable = (
            abs(t1) < 0.15
            and abs(t2) < 0.15
            and abs(t3) < 0.15
            and abs(t1_dot) < 1.0
            and abs(t2_dot) < 1.0
            and abs(t3_dot) < 1.0
            and abs(x) < 0.5
            and abs(x_dot) < 1.0
        )
        stable_bonus = self.stability_bonus if is_stable else 0.0

        cost = cart_cost + angle_cost + vel_cost + action_cost
        reward = float(self.base_reward - cost + stable_bonus)

        terminated = bool(abs(x) > 2.4 or abs(t1) > 2.75 or abs(t2) > 2.75 or abs(t3) > 2.75)
        if terminated:
            reward -= self.terminal_penalty

        truncated = self.step_count >= self.max_episode_steps
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
