import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vizdoom import gymnasium_wrapper

from actions import DEFEND_LINE_DISCRETE_ACTIONS, DEFEND_LINE_ACTION_NAMES
from doom_env_defend_the_line import DefendLineDoomEnv


MODEL_PATH = "checkpoints/recurrent_ppo_VizdoomDefendLine-v1/best/best_model.zip"
ENV_ID = "VizdoomDefendLine-v1"


def make_doom_env(render=True):
    return DefendLineDoomEnv(
        env_id=ENV_ID,
        render=render,
        discrete_actions=DEFEND_LINE_DISCRETE_ACTIONS,
        action_names=DEFEND_LINE_ACTION_NAMES,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    )


def evaluate(model_path=MODEL_PATH, episodes=3, render=True, deterministic=True):
    sb3_contrib = importlib.import_module("sb3_contrib")
    model = sb3_contrib.RecurrentPPO.load(model_path)

    env = make_doom_env(render=render)

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            lstm_states = None
            episode_starts = [True]
            total_reward = 0.0
            total_base_reward = 0.0
            step = 0

            while not done:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
                action_idx = int(action)

                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated
                episode_starts = [done]
                total_reward += reward
                total_base_reward += info.get("base_reward", 0.0)

                action_name = DEFEND_LINE_ACTION_NAMES.get(action_idx, f"action_{action_idx}")
                health = info.get("health")
                health_text = f"{health:5.1f}" if health is not None else "    ?"
                print(
                    f"ep={ep + 1} "
                    f"step={step:4d} "
                    f"action={action_name:<20s} "
                    f"health={health_text} "
                    f"reward={reward:+.3f} "
                    f"(base={info.get('base_reward', 0.0):+.3f})"
                )

                step += 1

            print(
                f"\nEpisode {ep + 1}: {step} steps | "
                f"shaped reward={total_reward:.2f} | "
                f"base reward={total_base_reward:.2f}\n"
            )

    finally:
        env.close()


if __name__ == "__main__":
    evaluate(
        model_path=MODEL_PATH,
        episodes=5,
        render=True,
        deterministic=True,
    )
