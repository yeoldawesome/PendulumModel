
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# If your custom env is registered, you can use its ID directly
# Otherwise, import and register it here
from triple_pendulum_env import InvertedTriplePendulumEnv, TRIPLE_PENDULUM_ENV_ID

def main():
    # Register the custom environment if not already registered
    try:
        env = gym.make(TRIPLE_PENDULUM_ENV_ID)
    except Exception:
        # Fallback: direct instantiation
        env = InvertedTriplePendulumEnv()

    # Optionally wrap in a vectorized env for SB3
    # env = make_vec_env(lambda: env, n_envs=1)

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

    model.learn(total_timesteps=1_000_000)
    model.save("sac_triple_pendulum_sb3")

if __name__ == "__main__":
    main()
