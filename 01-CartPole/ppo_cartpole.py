"""
Chapter 1: Training CartPole using PPO with Stable-Baselines3

Training metrics (reward curves, losses, etc.) are logged with SwanLab.

Usage:
    # Default: training + SwanLab logging (no GUI, faster)
    python 1-ppo_cartpole.py

    # Enable GUI demo after training
    python 1-ppo_cartpole.py --gui

About the --gui flag:
    Training always runs headless (without rendering), so GUI does not affect training speed.
    The --gui flag only controls whether a CartPole animation window appears after training.
    With GUI enabled, rendering slows execution because frames wait for screen refresh.
    Without GUI, evaluation runs much faster.
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from swanlab.integration.sb3 import SwanLabCallback
import swanlab


class LogApproxKL(BaseCallback):
    """
    Logs train/approx_kl manually to SwanLab.

    SB3 PPO internally records "train/approx_kl" using logger.record(),
    but the value is stored as numpy.float32.

    SwanLab's SB3 callback checks types using:
        isinstance(value, (int, float))

    numpy.float32 does not pass this check, so approx_kl gets skipped silently.

    This callback retrieves approx_kl from the logger cache,
    converts it to a Python float, and logs it manually.
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logger = self.model.logger

        if hasattr(logger, "name_to_value") and "train/approx_kl" in logger.name_to_value:
            value = float(logger.name_to_value["train/approx_kl"])

            swanlab.log(
                {"train/approx_kl": value},
                step=self.num_timesteps
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SB3 PPO CartPole Training"
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show CartPole GUI demo after training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs("output", exist_ok=True)

    # ==========================================
    # Training Phase
    # ==========================================
    env = gym.make("CartPole-v1")

    # Print environment information
    print("=" * 50)
    print("CartPole-v1 Environment Information")
    print("=" * 50)

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space:      {env.action_space}")
    print(f"Observation High:  {env.observation_space.high}")
    print(f"Observation Low:   {env.observation_space.low}")

    print(
        f"Termination Condition: "
        f"Position > ±{env.unwrapped.x_threshold}, "
        f"Angle > ±{env.unwrapped.theta_threshold_radians:.4f} rad "
        f"(≈ ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°)"
    )

    print("=" * 50)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1
    )

    print("Starting training with SwanLab logging...")

    swanlab_cb = SwanLabCallback(
        project="cartpole-ppo",
        experiment_name="PPO-CartPole-v1",
        mode="local",
    )

    model.learn(
        total_timesteps=80000,
        callback=[swanlab_cb, LogApproxKL()],
    )

    # Evaluation
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10
    )

    print(f"Training complete!")
    print(f"Average Reward: {mean_reward} +/- {std_reward}")

    model.save("output/ppo_cartpole")

    env.close()

    # ==========================================
    # Demonstration Phase
    # ==========================================
    print("\nRunning trained agent demonstration...")

    render_mode = "human" if args.gui else None

    vis_env = gym.make(
        "CartPole-v1",
        render_mode=render_mode
    )

    model = PPO.load("output/ppo_cartpole")

    for episode in range(5):
        obs, info = vis_env.reset()

        done = False
        truncated = False
        score = 0

        while not (done or truncated):
            action, _states = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, done, truncated, info = vis_env.step(action)

            score += reward

        print(f"Episode {episode + 1} Score: {score}")

    vis_env.close()

    if args.gui:
        print("\nGUI demonstration finished.")
    else:
        print("\nTip: Use --gui to visualize the CartPole agent.")

    print("SwanLab dashboard: swanlab watch swanlog")


if __name__ == "__main__":
    main()