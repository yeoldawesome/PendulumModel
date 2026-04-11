import sys
import os
import csv
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from triple_pendulum_env import InvertedTriplePendulumEnv, TRIPLE_PENDULUM_ENV_ID

def run_custom_eval(model, env_id, eval_episodes=10, max_steps=1000, seed=42):
    env = gym.make(env_id)
    returns = []
    time_to_failure = []
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                time_to_failure.append(t + 1)
                break
        else:
            time_to_failure.append(max_steps)
        returns.append(total_reward)
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "return_std": float(np.std(returns)),
        "avg_time_to_failure_steps": float(np.mean(time_to_failure)),
        "median_time_to_failure_steps": float(np.median(time_to_failure)),
        "max_time_to_failure_steps": float(np.max(time_to_failure)),
    }

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train SAC with SB3 on triple pendulum.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--env-id", type=str, default=TRIPLE_PENDULUM_ENV_ID, help="Gymnasium environment id.")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N episodes.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes per eval.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode.")
    args = parser.parse_args()

    env_id = args.env_id
    total_episodes = args.episodes
    eval_every = args.eval_every
    eval_episodes = args.eval_episodes
    max_steps = args.max_steps
    output_csv = os.path.join("artifacts", f"sac_eval_metrics.csv")
    os.makedirs("artifacts", exist_ok=True)

    env = gym.make(env_id)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./sac_sb3_tensorboard/",
        batch_size=256,
        learning_rate=3e-4,
        buffer_size=1000000,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        policy_kwargs=dict(net_arch=[512, 512]),
    )

    # Prepare CSV
    csv_header = [
        "episode", "mean_return", "median_return", "return_std",
        "avg_time_to_failure_steps", "median_time_to_failure_steps", "max_time_to_failure_steps"
    ]
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    steps_per_episode = max_steps
    for ep in range(1, total_episodes + 1):
        model.learn(total_timesteps=steps_per_episode, reset_num_timesteps=False, progress_bar=False)
        if ep % eval_every == 0 or ep == 1:
            metrics = run_custom_eval(model, env_id, eval_episodes, max_steps)
            row = [ep] + [metrics[k] for k in csv_header[1:]]
            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"[Eval] Episode {ep}: {metrics}", flush=True)

    model.save(os.path.join("artifacts", "sac_triple_pendulum_sb3"))
    print("Training complete. Model and metrics saved.")

if __name__ == "__main__":
    main()
