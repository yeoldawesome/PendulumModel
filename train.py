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
    total_episodes: int = 100
    std_dev: float = 0.2
    critic_lr: float = 0.002
    actor_lr: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005
    buffer_capacity: int = 50000
    batch_size: int = 64
    max_steps_per_episode: int = 200


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG on a continuous-control Gymnasium environment with Keras.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1", help="Gymnasium environment id (for example Pendulum-v1 or InvertedDoublePendulum-v4).")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory for model files and metadata.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--render", action="store_true", help="Render environment while training (slower; not for HPC).")
    parser.add_argument("--require-gpu", type=int, default=0, choices=[0, 1], help="Exit with error when no GPU is detected.")
    parser.add_argument("--max-steps-per-episode", type=int, default=200, help="Maximum environment steps per episode.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environment instances for faster simulation.")
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
    if args.render and args.num_envs > 1:
        raise ValueError("--render is only supported with --num-envs=1")

    cfg = DDPGConfig(
        total_episodes=args.episodes,
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

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Visible GPUs: {len(gpus)}")
    if args.require_gpu == 1 and len(gpus) == 0:
        raise RuntimeError("No GPU detected, and --require-gpu=1 was requested.")

    noise_shape = (1,) if args.num_envs == 1 else (args.num_envs, num_actions)
    noise = OUActionNoise(mean=np.zeros(noise_shape), std_deviation=float(cfg.std_dev) * np.ones(noise_shape))

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
    for episode in range(cfg.total_episodes):
        if args.num_envs == 1:
            prev_state, _ = env.reset(seed=args.seed + episode)
        else:
            seeds = [args.seed + episode * args.num_envs + i for i in range(args.num_envs)]
            prev_state, _ = env.reset(seed=seeds)

        noise.reset()
        episode_reward = 0.0
        episode_rewards = np.zeros(args.num_envs, dtype=np.float32) if args.num_envs > 1 else None

        for _ in range(cfg.max_steps_per_episode):
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

        if args.num_envs > 1 and episode_rewards is not None:
            episode_reward = float(np.mean(episode_rewards))
        episodic_rewards.append(episode_reward)
        avg_reward = float(np.mean(episodic_rewards[-40:]))
        rolling_avg_rewards.append(avg_reward)
        print(f"Episode {episode + 1:03d}/{cfg.total_episodes} | Reward: {episode_reward:.2f} | Avg(40): {avg_reward:.2f}")

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

    actor_model.save_weights(actor_path)
    critic_model.save_weights(critic_path)
    target_actor.save_weights(target_actor_path)
    target_critic.save_weights(target_critic_path)
    np.save(rewards_path, np.array(rolling_avg_rewards, dtype=np.float32))

    metadata = {
        "created_at": run_stamp,
        "env": args.env_id,
        "seed": args.seed,
        "episodes": cfg.total_episodes,
        "max_steps_per_episode": cfg.max_steps_per_episode,
        "num_envs": args.num_envs,
        "buffer_capacity": cfg.buffer_capacity,
        "batch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "std_dev": cfg.std_dev,
        "visible_gpus": len(gpus),
        "final_avg_reward_40": rolling_avg_rewards[-1] if rolling_avg_rewards else None,
        "actor_weights": actor_path.name,
        "critic_weights": critic_path.name,
        "target_actor_weights": target_actor_path.name,
        "target_critic_weights": target_critic_path.name,
        "rewards_file": rewards_path.name,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Training complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
