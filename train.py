import argparse
import datetime as dt
import json
import os
import pathlib
import random
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf
from keras import layers


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
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(x)
    outputs = outputs * upper_bound

    return keras.Model(inputs, outputs)


def get_critic(num_states: int, num_actions: int) -> keras.Model:
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(64, activation="relu")(state_input)
    state_out = layers.Dense(64, activation="relu")(state_out)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(64, activation="relu")(action_input)
    action_out = layers.Dense(32, activation="relu")(action_out)

    concat = layers.Concatenate()([state_out, action_out])

    x = layers.Dense(512, activation="relu")(concat)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
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
    parser.add_argument("--env-id", type=str, default="InvertedDoublePendulum-v5", help="Gymnasium environment id (for example Pendulum-v1 or InvertedDoublePendulum-v5).")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory for model files and metadata.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--render", action="store_true", help="Render environment while training (slower; not for HPC).")
    parser.add_argument("--require-gpu", type=int, default=0, choices=[0, 1], help="Exit with error when no GPU is detected.")
    parser.add_argument("--max-steps-per-episode", type=int, default=2000, help="Maximum environment steps per episode.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environment instances for faster simulation.")
    parser.add_argument("--log-interval-steps", type=int, default=100, help="Print in-episode progress every N steps.")
    parser.add_argument("--actor-lr", type=float, default=0.0003, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=0.001, help="Critic learning rate.")
    parser.add_argument("--tau", type=float, default=0.002, help="Target network update factor.")
    parser.add_argument("--buffer-capacity", type=int, default=200000, help="Replay buffer capacity.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for replay sampling.")
    parser.add_argument("--noise-start", type=float, default=0.3, help="Initial exploration noise stddev.")
    parser.add_argument("--noise-end", type=float, default=0.05, help="Final exploration noise stddev.")
    parser.add_argument("--noise-decay-episodes", type=int, default=300, help="Episodes over which exploration noise decays.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.log_interval_steps < 1:
        raise ValueError("--log-interval-steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.buffer_capacity < args.batch_size:
        raise ValueError("--buffer-capacity must be >= --batch-size")
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
    )

    set_seed(args.seed)

    render_mode = "human" if args.render else None
    if args.num_envs == 1:
        env = gym.make(args.env_id, render_mode=render_mode)
        obs_space = env.observation_space
        action_space = env.action_space
    else:
        env_fns = [lambda env_id=args.env_id: gym.make(env_id) for _ in range(args.num_envs)]
        env = gym.vector.AsyncVectorEnv(env_fns)
        obs_space = env.single_observation_space
        action_space = env.single_action_space

    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError(f"Environment {args.env_id} must use a continuous Box action space.")
    if not isinstance(obs_space, gym.spaces.Box):
        raise ValueError(f"Environment {args.env_id} must use a Box observation space.")

    num_states = obs_space.shape[0]
    num_actions = action_space.shape[0]
    upper_bound = action_space.high.astype(np.float32)
    lower_bound = action_space.low.astype(np.float32)

    print(f"State space: {num_states}")
    print(f"Action space: {num_actions}")
    print(f"Environment: {args.env_id}")
    print(f"Action bounds: low={lower_bound} high={upper_bound}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Log interval steps: {args.log_interval_steps}")
    print(f"Actor lr: {cfg.actor_lr}")
    print(f"Critic lr: {cfg.critic_lr}")
    print(f"Replay capacity: {cfg.buffer_capacity}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Exploration noise: start={cfg.std_dev_start:.3f} end={cfg.std_dev_end:.3f} decay_episodes={cfg.std_dev_decay_episodes}")
    if args.env_id.startswith("InvertedDoublePendulum") and args.episodes < 200:
        print("Warning: InvertedDoublePendulum usually needs many more than 200 episodes for clear learning progress.", flush=True)

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

    episodic_rewards: list[float] = []
    rolling_avg_rewards: list[float] = []
    episode_lengths: list[int] = []
    best_avg_reward = float("-inf")
    best_episode = -1
    best_actor_weights = None
    import csv
    def evaluate_policy(actor_model, env, num_episodes=5, max_steps=2000, seed=42):
        rewards = []
        lengths = []
        for ep in range(num_episodes):
            state, _ = env.reset(seed=seed + ep)
            total_reward = 0.0
            for t in range(max_steps):
                action = actor_model(np.expand_dims(state, axis=0), training=False).numpy().reshape(-1)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            lengths.append(t + 1)
        return np.mean(rewards), np.mean(lengths)

    csv_path = pathlib.Path(args.output_dir) / "progress.csv"
    csv_header = ["episode", "avg_reward_40", "eval_avg_reward", "eval_avg_length"]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    for episode in range(cfg.total_episodes):
        print(f"Episode {episode + 1:03d}/{cfg.total_episodes} started", flush=True)
        episode_noise = linear_decay(episode, cfg.std_dev_start, cfg.std_dev_end, cfg.std_dev_decay_episodes)
        noise.std_dev = episode_noise * np.ones(noise_shape, dtype=np.float32)
        if args.num_envs == 1:
            prev_state, _ = env.reset(seed=args.seed + episode)
        else:
            seeds = [args.seed + episode * args.num_envs + i for i in range(args.num_envs)]
            prev_state, _ = env.reset(seed=seeds)

        noise.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_rewards = np.zeros(args.num_envs, dtype=np.float32) if args.num_envs > 1 else None

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

            if replay_buffer.can_sample():
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

            if (step_idx + 1) % args.log_interval_steps == 0:
                if args.num_envs == 1:
                    partial_reward = episode_reward
                else:
                    partial_reward = float(np.mean(episode_rewards)) if episode_rewards is not None else 0.0
                print(
                    f"Episode {episode + 1:03d}/{cfg.total_episodes} | Step {step_idx + 1}/{cfg.max_steps_per_episode} | Partial reward: {partial_reward:.2f}",
                    flush=True,
                )

        if args.num_envs > 1 and episode_rewards is not None:
            episode_reward = float(np.mean(episode_rewards))
        episodic_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        avg_reward = float(np.mean(episodic_rewards[-40:]))
        rolling_avg_rewards.append(avg_reward)
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_episode = episode + 1
            best_actor_weights = actor_model.get_weights()
        print(
            f"Episode {episode + 1:03d}/{cfg.total_episodes} | Steps: {episode_steps:04d} | Reward: {episode_reward:.2f} | Avg(40): {avg_reward:.2f} | Noise: {episode_noise:.3f}",
            flush=True,
        )

        # Every 10 episodes: evaluate, log to CSV, save checkpoint, auto-push
        if (episode + 1) % 10 == 0 or (episode + 1) == cfg.total_episodes:
            eval_avg_reward, eval_avg_length = evaluate_policy(actor_model, env, num_episodes=5, max_steps=cfg.max_steps_per_episode, seed=args.seed + 10000)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode + 1, avg_reward, eval_avg_reward, eval_avg_length])

            # Save checkpoint
            checkpoint_prefix = f"model_pendulum_{run_stamp}_ep{episode + 1}"
            actor_model.save_weights(output_dir / f"{checkpoint_prefix}_actor.weights.h5")
            critic_model.save_weights(output_dir / f"{checkpoint_prefix}_critic.weights.h5")
            target_actor.save_weights(output_dir / f"{checkpoint_prefix}_target_actor.weights.h5")
            target_critic.save_weights(output_dir / f"{checkpoint_prefix}_target_critic.weights.h5")

            # Auto-push artifacts if requested
            if os.environ.get("AUTO_PUSH", "0") == "1":
                import subprocess
                try:
                    subprocess.run(["git", "add", "-f", str(output_dir)], check=True)
                    subprocess.run(["git", "commit", "-m", f"hpc: checkpoint and progress at episode {episode + 1}"], check=True)
                    subprocess.run(["git", "push"], check=True)
                except Exception as e:
                    print(f"Auto-push failed: {e}", flush=True)

    env.close()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_prefix = f"model_pendulum_{run_stamp}_ep{cfg.total_episodes}"

    actor_path = output_dir / f"{artifact_prefix}_actor.weights.h5"
    critic_path = output_dir / f"{artifact_prefix}_critic.weights.h5"
    target_actor_path = output_dir / f"{artifact_prefix}_target_actor.weights.h5"
    target_critic_path = output_dir / f"{artifact_prefix}_target_critic.weights.h5"
    rewards_path = output_dir / f"{artifact_prefix}_rewards.npy"
    episode_lengths_path = output_dir / f"{artifact_prefix}_episode_lengths.npy"
    best_actor_path = output_dir / f"{artifact_prefix}_best_actor.weights.h5"

    actor_model.save_weights(actor_path)
    critic_model.save_weights(critic_path)
    target_actor.save_weights(target_actor_path)
    target_critic.save_weights(target_critic_path)
    np.save(rewards_path, np.array(rolling_avg_rewards, dtype=np.float32))
    np.save(episode_lengths_path, np.array(episode_lengths, dtype=np.int32))

    if best_actor_weights is not None:
        best_actor_model = get_actor(num_states=num_states, num_actions=num_actions, upper_bound=upper_bound)
        best_actor_model.set_weights(best_actor_weights)
        best_actor_model.save_weights(best_actor_path)

    metadata = {
        "created_at": run_stamp,
        "env": args.env_id,
        "seed": args.seed,
        "episodes": cfg.total_episodes,
        "max_steps_per_episode": cfg.max_steps_per_episode,
        "num_envs": args.num_envs,
        "buffer_capacity": cfg.buffer_capacity,
        "batch_size": cfg.batch_size,
        "noise_start": cfg.std_dev_start,
        "noise_end": cfg.std_dev_end,
        "noise_decay_episodes": cfg.std_dev_decay_episodes,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "best_avg_reward_40": best_avg_reward if best_avg_reward > float("-inf") else None,
        "best_episode": best_episode if best_episode > 0 else None,
        "visible_gpus": len(gpus),
        "final_avg_reward_40": rolling_avg_rewards[-1] if rolling_avg_rewards else None,
        "actor_weights": actor_path.name,
        "critic_weights": critic_path.name,
        "target_actor_weights": target_actor_path.name,
        "target_critic_weights": target_critic_path.name,
        "rewards_file": rewards_path.name,
        "episode_lengths_file": episode_lengths_path.name,
        "best_actor_weights": best_actor_path.name if best_actor_weights is not None else None,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Training complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
