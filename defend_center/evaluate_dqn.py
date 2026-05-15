import argparse
import importlib

from vizdoom import gymnasium_wrapper

from defend_center.actions import (
    DEFEND_CENTER_ACTION_NAMES,
    DEFEND_CENTER_DISCRETE_ACTIONS,
)
from defend_center.doom_env_defend_center import DefendCenterDoomEnv


MODEL_PATH = "checkpoints/dqn_VizdoomDefendCenter-v1_raw_5000000/best/best_model.zip"
ENV_ID = "VizdoomDefendCenter-v1"


def make_doom_env(render=True, reward_mode="raw"):
    return DefendCenterDoomEnv(
        env_id=ENV_ID,
        render=render,
        discrete_actions=DEFEND_CENTER_DISCRETE_ACTIONS,
        action_names=DEFEND_CENTER_ACTION_NAMES,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
        reward_shaping=reward_mode == "shaped",
    )


def evaluate(
    model_path=MODEL_PATH,
    episodes=5,
    render=True,
    deterministic=True,
    reward_mode="raw",
):
    sb3 = importlib.import_module("stable_baselines3")
    model = sb3.DQN.load(model_path)

    env = make_doom_env(render=render, reward_mode=reward_mode)

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            total_base_reward = 0.0
            step = 0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                action_idx = int(action)

                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated
                total_reward += reward
                total_base_reward += info.get("base_reward", reward)

                action_name = DEFEND_CENTER_ACTION_NAMES.get(
                    action_idx,
                    f"action_{action_idx}",
                )
                print(
                    f"ep={ep + 1} "
                    f"step={step:4d} "
                    f"action={action_name:<16s} "
                    f"health={info.get('health', 0.0):>5.1f} "
                    f"ammo={info.get('ammo', 0.0):>5.1f} "
                    f"reward={reward:+.3f} "
                    f"base={info.get('base_reward', reward):+.3f}"
                )

                step += 1

            print(
                f"\nEpisode {ep + 1}: {step} steps | "
                f"returned reward={total_reward:.2f} | "
                f"base reward={total_base_reward:.2f}\n"
            )

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=["raw", "shaped"],
        default="raw",
        help="Reward mode to use during evaluation.",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
        reward_mode=args.reward_mode,
    )
