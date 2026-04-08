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

    def can_sample(self) -> bool:
        return min(self.buffer_counter, self.buffer_capacity) >= self.batch_size

    def sample(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        return state_batch, action_batch, reward_batch, next_state_batch


def get_actor(num_states: int, upper_bound: float) -> keras.Model:
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(x)
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


def choose_action(state: np.ndarray, noise: OUActionNoise, actor_model: keras.Model, lower_bound: float, upper_bound: float) -> np.ndarray:
    state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
    sampled_actions = tf.squeeze(actor_model(state_tensor)).numpy()
    sampled_actions = sampled_actions + noise()
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return np.array([np.squeeze(legal_action)], dtype=np.float32)


def update_targets(target: keras.Model, source: keras.Model, tau: float) -> None:
    target_weights = target.get_weights()
    source_weights = source.get_weights()

    for index in range(len(target_weights)):
        target_weights[index] = source_weights[index] * tau + target_weights[index] * (1.0 - tau)

    target.set_weights(target_weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG on Pendulum-v1 with Keras.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory for model files and metadata.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--render", action="store_true", help="Render environment while training (slower; not for HPC).")
    parser.add_argument("--require-gpu", type=int, default=0, choices=[0, 1], help="Exit with error when no GPU is detected.")
    parser.add_argument("--max-steps-per-episode", type=int, default=200, help="Maximum environment steps per episode.")
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
    cfg = DDPGConfig(total_episodes=args.episodes, max_steps_per_episode=args.max_steps_per_episode)

    set_seed(args.seed)

    render_mode = "human" if args.render else None
    env = gym.make("Pendulum-v1", render_mode=render_mode)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    upper_bound = float(env.action_space.high[0])
    lower_bound = float(env.action_space.low[0])

    print(f"State space: {num_states}")
    print(f"Action space: {num_actions}")
    print(f"Action bounds: [{lower_bound}, {upper_bound}]")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Visible GPUs: {len(gpus)}")
    if args.require_gpu == 1 and len(gpus) == 0:
        raise RuntimeError("No GPU detected, and --require-gpu=1 was requested.")

    noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(cfg.std_dev) * np.ones(1))

    actor_model = get_actor(num_states=num_states, upper_bound=upper_bound)
    critic_model = get_critic(num_states=num_states, num_actions=num_actions)

    target_actor = get_actor(num_states=num_states, upper_bound=upper_bound)
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
        prev_state, _ = env.reset(seed=args.seed + episode)
        noise.reset()
        episode_reward = 0.0

        for _ in range(cfg.max_steps_per_episode):
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

            if terminated or truncated:
                break

            prev_state = state

        episodic_rewards.append(episode_reward)
        avg_reward = float(np.mean(episodic_rewards[-40:]))
        rolling_avg_rewards.append(avg_reward)
        print(f"Episode {episode + 1:03d}/{cfg.total_episodes} | Reward: {episode_reward:.2f} | Avg(40): {avg_reward:.2f}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    actor_path = output_dir / f"pendulum_actor_{run_stamp}.weights.h5"
    critic_path = output_dir / f"pendulum_critic_{run_stamp}.weights.h5"
    target_actor_path = output_dir / f"pendulum_target_actor_{run_stamp}.weights.h5"
    target_critic_path = output_dir / f"pendulum_target_critic_{run_stamp}.weights.h5"
    rewards_path = output_dir / f"pendulum_rewards_{run_stamp}.npy"

    actor_model.save_weights(actor_path)
    critic_model.save_weights(critic_path)
    target_actor.save_weights(target_actor_path)
    target_critic.save_weights(target_critic_path)
    np.save(rewards_path, np.array(rolling_avg_rewards, dtype=np.float32))

    metadata = {
        "created_at": run_stamp,
        "env": "Pendulum-v1",
        "seed": args.seed,
        "episodes": cfg.total_episodes,
        "max_steps_per_episode": cfg.max_steps_per_episode,
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
