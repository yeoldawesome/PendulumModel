import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import random
import re
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from triple_pendulum_env import TRIPLE_PENDULUM_ENV_ID, register_triple_pendulum_env

DEFAULT_JOINTS = 3
JOINTS_TO_ENV_ID = {
    1: "Pendulum-v1",
    2: "InvertedDoublePendulum-v5",
    3: TRIPLE_PENDULUM_ENV_ID,
}


@dataclass
class DDPGConfig:
    total_episodes: int = 500
    std_dev_start: float = 0.3
    std_dev_end: float = 0.05
    std_dev_decay_episodes: int = 300
    critic_lr: float = 0.001
    actor_lr: float = 0.0003
    gamma: float = 0.99
    tau: float = 0.002
    buffer_capacity: int = 200000
    batch_size: int = 128
    max_steps_per_episode: int = 2000
    updates_per_step: int = 1
    warmup_steps: int = 0
    checkpoint_interval_episodes: int = 0


class OUActionNoise:
    def __init__(self, mean: np.ndarray, std_deviation: np.ndarray, theta: float = 0.15, dt: float = 1e-2):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_prev = np.zeros_like(self.mean)

    def __call__(self) -> np.ndarray:
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self) -> None:
        self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, buffer_capacity: int, batch_size: int, state_dim: int, action_dim: int):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim), dtype=np.float32)

    def record(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.buffer_counter += 1

    def record_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        batch_size = states.shape[0]
        indices = (np.arange(batch_size) + self.buffer_counter) % self.buffer_capacity
        self.state_buffer[indices] = states
        self.action_buffer[indices] = actions
        self.reward_buffer[indices, 0] = rewards.astype(np.float32)
        self.next_state_buffer[indices] = next_states
        self.buffer_counter += batch_size

    def can_sample(self) -> bool:
        return min(self.buffer_counter, self.buffer_capacity) >= self.batch_size

    def size(self) -> int:
        return min(self.buffer_counter, self.buffer_capacity)

    def sample(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.randint(0, record_range, size=self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        return state_batch, action_batch, reward_batch, next_state_batch


def get_actor(num_states: int, num_actions: int, upper_bound: np.ndarray) -> keras.Model:
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(x)
    outputs = outputs * upper_bound

    return keras.Model(inputs, outputs)


def get_critic(num_states: int, num_actions: int) -> keras.Model:
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    x = layers.Dense(256, activation="relu")(concat)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    return keras.Model([state_input, action_input], outputs)


def choose_action(
    state: np.ndarray,
    noise: OUActionNoise,
    actor_model: keras.Model,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
    sampled_actions = tf.squeeze(actor_model(state_tensor), axis=0).numpy()
    sampled_actions = sampled_actions + noise()
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return np.asarray(legal_action, dtype=np.float32)


def choose_actions_batch(
    states: np.ndarray,
    noise: OUActionNoise,
    actor_model: keras.Model,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    sampled_actions = actor_model(state_tensor, training=False).numpy()
    sampled_actions = sampled_actions + noise()
    legal_actions = np.clip(sampled_actions, lower_bound, upper_bound)
    return legal_actions.astype(np.float32)


def update_targets(target: keras.Model, source: keras.Model, tau: float) -> None:
    tau_tensor = tf.convert_to_tensor(tau, dtype=tf.float32)
    one_minus_tau = tf.convert_to_tensor(1.0, dtype=tf.float32) - tau_tensor
    for target_var, source_var in zip(target.weights, source.weights):
        target_var.assign(source_var * tau_tensor + target_var * one_minus_tau)


def linear_decay(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    alpha = min(max(step / decay_steps, 0.0), 1.0)
    return (1.0 - alpha) * start + alpha * end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG on a continuous-control Gymnasium environment with Keras.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument(
        "--joints",
        type=int,
        default=DEFAULT_JOINTS,
        choices=sorted(JOINTS_TO_ENV_ID.keys()),
        help=(
            "Number of pendulum joints used to auto-select env id "
            f"(1=Pendulum-v1, 2=InvertedDoublePendulum-v5, 3={TRIPLE_PENDULUM_ENV_ID})."
        ),
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="",
        help=(
            "Gymnasium environment id override. If omitted, env is chosen from --joints "
            f"(for example Pendulum-v1, InvertedDoublePendulum-v5, or {TRIPLE_PENDULUM_ENV_ID})."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory for model files and metadata.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--render", action="store_true", help="Render environment while training (slower; not for HPC).")
    parser.add_argument("--require-gpu", type=int, default=0, choices=[0, 1], help="Exit with error when no GPU is detected.")
    parser.add_argument("--max-steps-per-episode", type=int, default=2000, help="Maximum environment steps per episode.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environment instances for faster simulation.")
    parser.add_argument("--log-interval-steps", type=int, default=500, help="Print in-episode progress every N steps.")
    parser.add_argument(
        "--progress-bar",
        type=int,
        default=1,
        choices=[0, 1],
        help="Show a per-episode progress bar (1) or interval-based logging (0).",
    )
    parser.add_argument("--actor-lr", type=float, default=0.0003, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=0.001, help="Critic learning rate.")
    parser.add_argument("--tau", type=float, default=0.002, help="Target network update factor.")
    parser.add_argument("--buffer-capacity", type=int, default=200000, help="Replay buffer capacity.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for replay sampling.")
    parser.add_argument("--noise-start", type=float, default=0.3, help="Initial exploration noise stddev.")
    parser.add_argument("--noise-end", type=float, default=0.05, help="Final exploration noise stddev.")
    parser.add_argument("--noise-decay-episodes", type=int, default=300, help="Episodes over which exploration noise decays.")
    parser.add_argument(
        "--updates-per-step",
        type=int,
        default=4,
        help="Number of gradient updates to run per environment step after warmup.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=4096,
        help="Number of collected transitions before starting gradient updates.",
    )
    parser.add_argument(
        "--save-full-artifacts",
        type=int,
        default=0,
        choices=[0, 1],
        help="Save full training artifacts (1) or only one actor weights file (0, default).",
    )
    parser.add_argument(
        "--checkpoint-interval-episodes",
        type=int,
        default=1,
        help="Save actor checkpoints every N episodes (0 disables periodic checkpoints).",
    )
    parser.add_argument(
        "--eval-every-episodes",
        type=int,
        default=25,
        help="Run deterministic evaluation every N training episodes (0 disables periodic evaluation).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of deterministic evaluation episodes per evaluation event.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=1000,
        help="Max steps per deterministic evaluation episode.",
    )
    parser.add_argument(
        "--resume-actor-weights",
        type=str,
        default="",
        help="Path to actor weights file (.weights.h5) to continue training from.",
    )
    parser.add_argument(
        "--resume-episode-offset",
        type=int,
        default=-1,
        help=(
            "Episode number offset for resumed runs. If negative (default), the script tries to infer it "
            "from the resume filename pattern *_ep<NUM>_*."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _make_env(env_id: str) -> gym.Env:
    register_triple_pendulum_env()
    return gym.make(env_id)


def resolve_env_id(joints: int, env_id_override: str) -> str:
    env_id_override = env_id_override.strip()
    if env_id_override:
        return env_id_override
    if joints not in JOINTS_TO_ENV_ID:
        raise ValueError(f"Unsupported joints value: {joints}. Supported joints are {sorted(JOINTS_TO_ENV_ID.keys())}.")
    return JOINTS_TO_ENV_ID[joints]


def make_env_slug(env_id: str) -> str:
    slug = env_id.lower().replace("-", "_")
    slug = "".join(ch for ch in slug if ch.isalnum() or ch == "_")
    return slug or "unknown_env"


def infer_episode_offset_from_weights(path_str: str) -> int | None:
    file_name = pathlib.Path(path_str).name
    match = re.search(r"_ep(\d+)_", file_name)
    if not match:
        return None
    return int(match.group(1))


def run_deterministic_evaluation(
    actor_model: keras.Model,
    env_id: str,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    eval_env = _make_env(env_id)
    if not isinstance(eval_env.action_space, gym.spaces.Box):
        eval_env.close()
        raise ValueError(f"Environment {env_id} must use a continuous Box action space for evaluation.")

    lower_bound = eval_env.action_space.low.astype(np.float32)
    upper_bound = eval_env.action_space.high.astype(np.float32)

    returns: list[float] = []
    resets_per_episode: list[int] = []
    time_to_failure_steps: list[int] = []
    success_flags: list[int] = []

    try:
        for eval_idx in range(episodes):
            state, _ = eval_env.reset(seed=seed + eval_idx)
            total_reward = 0.0
            resets = 0
            first_failure_step = max_steps

            for step_idx in range(max_steps):
                state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
                action_value = tf.squeeze(actor_model(state_tensor, training=False), axis=0).numpy()
                action = np.clip(action_value, lower_bound, upper_bound).astype(np.float32)

                state, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += float(reward)

                if terminated or truncated:
                    if first_failure_step == max_steps:
                        first_failure_step = step_idx + 1
                    resets += 1
                    state, _ = eval_env.reset(seed=seed + eval_idx * 100000 + step_idx + 1)

            returns.append(total_reward)
            resets_per_episode.append(resets)
            time_to_failure_steps.append(first_failure_step)
            success_flags.append(1 if resets == 0 else 0)
    finally:
        eval_env.close()

    returns_arr = np.asarray(returns, dtype=np.float32)
    resets_arr = np.asarray(resets_per_episode, dtype=np.float32)
    failure_steps_arr = np.asarray(time_to_failure_steps, dtype=np.float32)
    success_arr = np.asarray(success_flags, dtype=np.float32)

    n = max(int(returns_arr.size), 1)
    mean_return = float(np.mean(returns_arr))
    median_return = float(np.median(returns_arr))
    std_return = float(np.std(returns_arr, ddof=1)) if n > 1 else 0.0
    stderr_return = std_return / np.sqrt(n) if n > 1 else 0.0
    ci95_scale = 1.96
    return_ci95_low = mean_return - ci95_scale * stderr_return
    return_ci95_high = mean_return + ci95_scale * stderr_return

    success_rate = float(np.mean(success_arr))
    success_stderr = float(np.sqrt(success_rate * (1.0 - success_rate) / n)) if n > 1 else 0.0
    success_ci95_low = max(0.0, success_rate - ci95_scale * success_stderr)
    success_ci95_high = min(1.0, success_rate + ci95_scale * success_stderr)

    return {
        "eval_episodes": float(episodes),
        "mean_return": mean_return,
        "median_return": median_return,
        "return_std": std_return,
        "return_stderr": stderr_return,
        "return_ci95_low": float(return_ci95_low),
        "return_ci95_high": float(return_ci95_high),
        "success_rate": success_rate,
        "success_ci95_low": float(success_ci95_low),
        "success_ci95_high": float(success_ci95_high),
        "avg_time_to_failure_steps": float(np.mean(failure_steps_arr)),
        "avg_resets_per_episode": float(np.mean(resets_arr)),
    }


@tf.function
def train_step(
    state_batch: tf.Tensor,
    action_batch: tf.Tensor,
    reward_batch: tf.Tensor,
    next_state_batch: tf.Tensor,
    actor_model: keras.Model,
    critic_model: keras.Model,
    target_actor: keras.Model,
    target_critic: keras.Model,
    actor_optimizer: keras.optimizers.Optimizer,
    critic_optimizer: keras.optimizers.Optimizer,
    gamma: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as critic_tape:
        target_actions = target_actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
        critic_value = critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.reduce_mean(tf.square(y - critic_value))

    critic_grad = critic_tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

    with tf.GradientTape() as actor_tape:
        actions = actor_model(state_batch, training=True)
        critic_value_for_actions = critic_model([state_batch, actions], training=True)
        actor_loss = -tf.reduce_mean(critic_value_for_actions)

    actor_grad = actor_tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    return actor_loss, critic_loss


def main() -> None:
    register_triple_pendulum_env()
    args = parse_args()
    resolved_env_id = resolve_env_id(args.joints, args.env_id)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.log_interval_steps < 1:
        raise ValueError("--log-interval-steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.buffer_capacity < args.batch_size:
        raise ValueError("--buffer-capacity must be >= --batch-size")
    if args.updates_per_step < 1:
        raise ValueError("--updates-per-step must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.checkpoint_interval_episodes < 0:
        raise ValueError("--checkpoint-interval-episodes must be >= 0")
    if args.eval_every_episodes < 0:
        raise ValueError("--eval-every-episodes must be >= 0")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be >= 1")
    if args.eval_max_steps < 1:
        raise ValueError("--eval-max-steps must be >= 1")
    if args.resume_episode_offset < -1:
        raise ValueError("--resume-episode-offset must be >= -1")
    if args.noise_start < 0 or args.noise_end < 0:
        raise ValueError("--noise-start and --noise-end must be >= 0")
    if args.render and args.num_envs > 1:
        raise ValueError("--render is only supported with --num-envs=1")

    cfg = DDPGConfig(
        total_episodes=args.episodes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        std_dev_start=args.noise_start,
        std_dev_end=args.noise_end,
        std_dev_decay_episodes=args.noise_decay_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        updates_per_step=args.updates_per_step,
        warmup_steps=args.warmup_steps,
        checkpoint_interval_episodes=args.checkpoint_interval_episodes,
    )

    set_seed(args.seed)

    render_mode = "human" if args.render else None
    if args.num_envs == 1:
        env = _make_env(resolved_env_id) if render_mode is None else gym.make(resolved_env_id, render_mode=render_mode)
        obs_space = env.observation_space
        action_space = env.action_space
    else:
        env_fns = [lambda env_id=resolved_env_id: _make_env(env_id) for _ in range(args.num_envs)]
        env = gym.vector.AsyncVectorEnv(env_fns)
        obs_space = env.single_observation_space
        action_space = env.single_action_space

    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError(f"Environment {resolved_env_id} must use a continuous Box action space.")
    if not isinstance(obs_space, gym.spaces.Box):
        raise ValueError(f"Environment {resolved_env_id} must use a Box observation space.")

    num_states = obs_space.shape[0]
    num_actions = action_space.shape[0]
    upper_bound = action_space.high.astype(np.float32)
    lower_bound = action_space.low.astype(np.float32)

    print(f"State space: {num_states}")
    print(f"Action space: {num_actions}")
    print(f"Environment: {resolved_env_id}")
    print(f"Joints: {args.joints}")
    print(f"Action bounds: low={lower_bound} high={upper_bound}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Log interval steps: {args.log_interval_steps}")
    print(f"Progress bar: {args.progress_bar}")
    print(f"Actor lr: {cfg.actor_lr}")
    print(f"Critic lr: {cfg.critic_lr}")
    print(f"Replay capacity: {cfg.buffer_capacity}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Updates per step: {cfg.updates_per_step}")
    print(f"Warmup steps: {cfg.warmup_steps}")
    print(f"Checkpoint interval episodes: {cfg.checkpoint_interval_episodes}")
    print(f"Eval every episodes: {args.eval_every_episodes}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Eval max steps: {args.eval_max_steps}")
    print(f"Save full artifacts: {args.save_full_artifacts}")
    print(f"Exploration noise: start={cfg.std_dev_start:.3f} end={cfg.std_dev_end:.3f} decay_episodes={cfg.std_dev_decay_episodes}")
    if resolved_env_id.startswith("InvertedDoublePendulum") and args.episodes < 200:
        print("Warning: InvertedDoublePendulum usually needs many more than 200 episodes for clear learning progress.", flush=True)
    if resolved_env_id.startswith("InvertedTriplePendulum") and args.episodes < 300:
        print("Warning: InvertedTriplePendulum usually needs many more than 300 episodes for clear learning progress.", flush=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Visible GPUs: {len(gpus)}")
    if args.require_gpu == 1 and len(gpus) == 0:
        raise RuntimeError("No GPU detected, and --require-gpu=1 was requested.")

    noise_shape = (1,) if args.num_envs == 1 else (args.num_envs, num_actions)
    noise = OUActionNoise(mean=np.zeros(noise_shape), std_deviation=float(cfg.std_dev_start) * np.ones(noise_shape, dtype=np.float32))

    actor_model = get_actor(num_states=num_states, num_actions=num_actions, upper_bound=upper_bound)
    critic_model = get_critic(num_states=num_states, num_actions=num_actions)

    target_actor = get_actor(num_states=num_states, num_actions=num_actions, upper_bound=upper_bound)
    target_critic = get_critic(num_states=num_states, num_actions=num_actions)

    resume_actor_weights = args.resume_actor_weights.strip()
    episode_offset = args.resume_episode_offset
    if resume_actor_weights:
        resume_path = pathlib.Path(resume_actor_weights)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-actor-weights not found: {resume_path}")
        actor_model.load_weights(resume_path)
        print(f"Loaded actor weights from: {resume_path}")

        if episode_offset < 0:
            inferred_offset = infer_episode_offset_from_weights(resume_actor_weights)
            if inferred_offset is not None:
                episode_offset = inferred_offset
                print(f"Inferred resume episode offset from filename: {episode_offset}")
            else:
                episode_offset = 0
                print("Could not infer resume episode offset from filename; defaulting to 0.")
    else:
        if episode_offset == -1:
            episode_offset = 0

    if episode_offset < 0:
        episode_offset = 0

    print(f"Resume actor weights: {resume_actor_weights or '<none>'}")
    print(f"Resume episode offset: {episode_offset}")

    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    actor_optimizer = keras.optimizers.Adam(cfg.actor_lr)
    critic_optimizer = keras.optimizers.Adam(cfg.critic_lr)

    replay_buffer = ReplayBuffer(
        buffer_capacity=cfg.buffer_capacity,
        batch_size=cfg.batch_size,
        state_dim=num_states,
        action_dim=num_actions,
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_slug = make_env_slug(resolved_env_id)

    def save_actor_checkpoint(episode_number: int) -> pathlib.Path:
        checkpoint_path = output_dir / f"model_pendulum_j{args.joints}_ep{episode_number}_{run_stamp}_{env_slug}_actor.weights.h5"
        actor_model.save_weights(checkpoint_path)
        return checkpoint_path

    episodic_rewards: list[float] = []
    rolling_avg_rewards: list[float] = []
    episode_lengths: list[int] = []
    eval_enabled = args.eval_every_episodes > 0
    best_train_avg_reward = float("-inf")
    best_train_episode = -1
    best_eval_mean_return = float("-inf")
    best_eval_episode = -1
    best_actor_weights = None
    eval_rows: list[dict[str, float | int]] = []
    total_env_steps = 0
    latest_checkpoint_path: pathlib.Path | None = None
    for episode in range(cfg.total_episodes):
        global_episode = episode_offset + episode + 1
        total_episode_target = episode_offset + cfg.total_episodes
        print(f"Episode {global_episode:03d}/{total_episode_target}", flush=True)
        episode_noise = linear_decay(global_episode - 1, cfg.std_dev_start, cfg.std_dev_end, cfg.std_dev_decay_episodes)
        noise.std_dev = episode_noise * np.ones(noise_shape, dtype=np.float32)
        if args.num_envs == 1:
            prev_state, _ = env.reset(seed=args.seed + global_episode - 1)
        else:
            seeds = [args.seed + (global_episode - 1) * args.num_envs + i for i in range(args.num_envs)]
            prev_state, _ = env.reset(seed=seeds)

        noise.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_rewards = np.zeros(args.num_envs, dtype=np.float32) if args.num_envs > 1 else None
        progress = keras.utils.Progbar(cfg.max_steps_per_episode, stateful_metrics=["partial_reward"]) if args.progress_bar == 1 else None

        for step_idx in range(cfg.max_steps_per_episode):
            episode_steps = step_idx + 1
            if args.num_envs == 1:
                action = choose_action(
                    state=prev_state,
                    noise=noise,
                    actor_model=actor_model,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )

                state, reward, terminated, truncated, _ = env.step(action)
                replay_buffer.record(prev_state, action, float(reward), state)
                episode_reward += float(reward)
                total_env_steps += 1
            else:
                actions = choose_actions_batch(
                    states=prev_state,
                    noise=noise,
                    actor_model=actor_model,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )

                state, reward, terminated, truncated, _ = env.step(actions)
                replay_buffer.record_batch(prev_state, actions, reward, state)
                episode_rewards += reward.astype(np.float32)
                total_env_steps += args.num_envs

            if args.num_envs == 1:
                partial_reward = episode_reward
            else:
                partial_reward = float(np.mean(episode_rewards)) if episode_rewards is not None else 0.0

            if progress is not None:
                progress.update(
                    step_idx + 1,
                    values=[
                        ("partial_reward", partial_reward),
                    ],
                )

            if replay_buffer.can_sample() and replay_buffer.size() >= cfg.warmup_steps:
                for _ in range(cfg.updates_per_step):
                    state_batch, action_batch, reward_batch, next_state_batch = replay_buffer.sample()
                    train_step(
                        state_batch=state_batch,
                        action_batch=action_batch,
                        reward_batch=reward_batch,
                        next_state_batch=next_state_batch,
                        actor_model=actor_model,
                        critic_model=critic_model,
                        target_actor=target_actor,
                        target_critic=target_critic,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        gamma=cfg.gamma,
                    )

                    update_targets(target_actor, actor_model, cfg.tau)
                    update_targets(target_critic, critic_model, cfg.tau)

            if args.num_envs == 1:
                if terminated or truncated:
                    break
                prev_state = state
            else:
                prev_state = state

            if args.progress_bar == 0 and (step_idx + 1) % args.log_interval_steps == 0:
                print(
                    f"Episode {global_episode:03d}/{total_episode_target} | Step {step_idx + 1}/{cfg.max_steps_per_episode} | Partial reward: {partial_reward:.2f}",
                    flush=True,
                )

        if args.num_envs > 1 and episode_rewards is not None:
            episode_reward = float(np.mean(episode_rewards))
        episodic_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        avg_reward = float(np.mean(episodic_rewards[-40:]))
        rolling_avg_rewards.append(avg_reward)
        if avg_reward > best_train_avg_reward:
            best_train_avg_reward = avg_reward
            best_train_episode = global_episode
            if not eval_enabled:
                best_actor_weights = actor_model.get_weights()
        print(
            f"Episode {global_episode:03d}/{total_episode_target} | Steps: {episode_steps:04d} | Reward: {episode_reward:.2f} | Avg(40): {avg_reward:.2f} | Noise: {episode_noise:.3f}",
            flush=True,
        )

        should_eval = eval_enabled and (
            global_episode % args.eval_every_episodes == 0 or episode == cfg.total_episodes - 1
        )
        if should_eval:
            eval_seed = args.seed + global_episode * 1000
            eval_metrics = run_deterministic_evaluation(
                actor_model=actor_model,
                env_id=resolved_env_id,
                episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
                seed=eval_seed,
            )

            eval_rows.append(
                {
                    "episode": global_episode,
                    "mean_return": eval_metrics["mean_return"],
                    "median_return": eval_metrics["median_return"],
                    "return_std": eval_metrics["return_std"],
                    "return_stderr": eval_metrics["return_stderr"],
                    "return_ci95_low": eval_metrics["return_ci95_low"],
                    "return_ci95_high": eval_metrics["return_ci95_high"],
                    "success_rate": eval_metrics["success_rate"],
                    "success_ci95_low": eval_metrics["success_ci95_low"],
                    "success_ci95_high": eval_metrics["success_ci95_high"],
                    "avg_time_to_failure_steps": eval_metrics["avg_time_to_failure_steps"],
                    "avg_resets_per_episode": eval_metrics["avg_resets_per_episode"],
                    "eval_episodes": int(eval_metrics["eval_episodes"]),
                    "eval_max_steps": args.eval_max_steps,
                }
            )

            print(
                " | ".join(
                    [
                        f"Eval @ episode {global_episode:03d}",
                        f"mean={eval_metrics['mean_return']:.2f}",
                        f"median={eval_metrics['median_return']:.2f}",
                        f"95% CI=[{eval_metrics['return_ci95_low']:.2f}, {eval_metrics['return_ci95_high']:.2f}]",
                        f"success={100.0 * eval_metrics['success_rate']:.1f}%",
                        (
                            f"success 95% CI=[{100.0 * eval_metrics['success_ci95_low']:.1f}%, "
                            f"{100.0 * eval_metrics['success_ci95_high']:.1f}%]"
                        ),
                        f"avg time-to-failure={eval_metrics['avg_time_to_failure_steps']:.1f}",
                        f"avg resets={eval_metrics['avg_resets_per_episode']:.2f}",
                    ]
                ),
                flush=True,
            )

            if eval_metrics["mean_return"] > best_eval_mean_return:
                best_eval_mean_return = eval_metrics["mean_return"]
                best_eval_episode = global_episode
                best_actor_weights = actor_model.get_weights()
                print(
                    f"New best checkpoint by eval mean return: episode {global_episode} ({best_eval_mean_return:.2f})",
                    flush=True,
                )

        if cfg.checkpoint_interval_episodes > 0 and global_episode % cfg.checkpoint_interval_episodes == 0:
            checkpoint_path = save_actor_checkpoint(global_episode)
            if latest_checkpoint_path is not None and latest_checkpoint_path != checkpoint_path and latest_checkpoint_path.exists():
                latest_checkpoint_path.unlink()
            latest_checkpoint_path = checkpoint_path
            print(f"Checkpoint saved: {checkpoint_path}", flush=True)

    env.close()
    episodes_completed = len(episodic_rewards)
    total_episodes_completed = episode_offset + episodes_completed
    artifact_prefix = f"model_pendulum_j{args.joints}_ep{total_episodes_completed}_{run_stamp}_{env_slug}"
    eval_metrics_csv_name = f"{artifact_prefix}_eval_metrics.csv"
    eval_metrics_csv_path = output_dir / eval_metrics_csv_name

    if eval_rows:
        eval_fieldnames = [
            "episode",
            "mean_return",
            "median_return",
            "return_std",
            "return_stderr",
            "return_ci95_low",
            "return_ci95_high",
            "success_rate",
            "success_ci95_low",
            "success_ci95_high",
            "avg_time_to_failure_steps",
            "avg_resets_per_episode",
            "eval_episodes",
            "eval_max_steps",
        ]
        with eval_metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=eval_fieldnames)
            writer.writeheader()
            writer.writerows(eval_rows)
        print(f"Saved eval metrics CSV: {eval_metrics_csv_path}", flush=True)

    actor_path = output_dir / f"{artifact_prefix}_actor.weights.h5"
    if best_actor_weights is not None:
        best_actor_model = get_actor(num_states=num_states, num_actions=num_actions, upper_bound=upper_bound)
        best_actor_model.set_weights(best_actor_weights)
        best_actor_model.save_weights(actor_path)
    else:
        actor_model.save_weights(actor_path)

    if args.save_full_artifacts == 1:
        critic_path = output_dir / f"{artifact_prefix}_critic.weights.h5"
        target_actor_path = output_dir / f"{artifact_prefix}_target_actor.weights.h5"
        target_critic_path = output_dir / f"{artifact_prefix}_target_critic.weights.h5"
        rewards_path = output_dir / f"{artifact_prefix}_rewards.npy"
        episode_lengths_path = output_dir / f"{artifact_prefix}_episode_lengths.npy"

        critic_model.save_weights(critic_path)
        target_actor.save_weights(target_actor_path)
        target_critic.save_weights(target_critic_path)
        np.save(rewards_path, np.array(rolling_avg_rewards, dtype=np.float32))
        np.save(episode_lengths_path, np.array(episode_lengths, dtype=np.int32))

        metadata = {
            "created_at": run_stamp,
            "env": resolved_env_id,
            "joints": args.joints,
            "seed": args.seed,
            "episodes_requested": cfg.total_episodes,
            "episodes_completed": episodes_completed,
            "episodes": total_episodes_completed,
            "resume_actor_weights": resume_actor_weights if resume_actor_weights else None,
            "resume_episode_offset": episode_offset,
            "total_episodes_completed": total_episodes_completed,
            "max_steps_per_episode": cfg.max_steps_per_episode,
            "num_envs": args.num_envs,
            "buffer_capacity": cfg.buffer_capacity,
            "batch_size": cfg.batch_size,
            "noise_start": cfg.std_dev_start,
            "noise_end": cfg.std_dev_end,
            "noise_decay_episodes": cfg.std_dev_decay_episodes,
            "updates_per_step": cfg.updates_per_step,
            "warmup_steps": cfg.warmup_steps,
            "checkpoint_interval_episodes": cfg.checkpoint_interval_episodes,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "actor_lr": cfg.actor_lr,
            "critic_lr": cfg.critic_lr,
            "best_train_avg_reward_40": best_train_avg_reward if best_train_avg_reward > float("-inf") else None,
            "best_train_episode": best_train_episode if best_train_episode > 0 else None,
            "eval_enabled": eval_enabled,
            "eval_every_episodes": args.eval_every_episodes,
            "eval_episodes": args.eval_episodes,
            "eval_max_steps": args.eval_max_steps,
            "best_eval_mean_return": best_eval_mean_return if best_eval_mean_return > float("-inf") else None,
            "best_eval_episode": best_eval_episode if best_eval_episode > 0 else None,
            "visible_gpus": len(gpus),
            "final_avg_reward_40": rolling_avg_rewards[-1] if rolling_avg_rewards else None,
            "final_eval_mean_return": eval_rows[-1]["mean_return"] if eval_rows else None,
            "eval_metrics_csv": eval_metrics_csv_name if eval_rows else None,
            "actor_weights": actor_path.name,
            "critic_weights": critic_path.name,
            "target_actor_weights": target_actor_path.name,
            "target_critic_weights": target_critic_path.name,
            "rewards_file": rewards_path.name,
            "episode_lengths_file": episode_lengths_path.name,
        }

        metadata_path = output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    print(f"Training complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
