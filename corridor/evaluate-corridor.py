import importlib
from vizdoom import gymnasium_wrapper

from actions import BASIC_DISCRETE_ACTIONS, ACTION_NAMES
from doom_env_cooridor import DoomEnv


MODEL_PATH = "corridor/ppo_vizdoom_model_5.zip"
ENV_ID = "VizdoomDeadlyCorridor-v1"


def load_ppo():
    sb3 = importlib.import_module("stable_baselines3")
    return sb3.PPO


def make_doom_env(render=True):
    return DoomEnv(
        env_id=ENV_ID,
        render=render,
        discrete_actions=BASIC_DISCRETE_ACTIONS,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    )


def evaluate(model_path=MODEL_PATH, episodes=3, render=True, deterministic=True):
    PPO = load_ppo()
    model = PPO.load(model_path)

    env = make_doom_env(render=render)

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            step = 0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                action_idx = int(action)

                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated
                total_reward += reward

                action_name = ACTION_NAMES.get(action_idx, f"action_{action_idx}")
                mapped = BASIC_DISCRETE_ACTIONS[action_idx]

                print(
                    f"episode={ep + 1} "
                    f"step={step} "
                    f"action={action_idx} "
                    f"name={action_name} "
                    f"mapped={mapped} "
                    f"reward={reward:.3f}"
                )

                step += 1

            print(f"Episode {ep + 1} total reward: {total_reward:.3f}\n")

    finally:
        env.close()


if __name__ == "__main__":
    evaluate(
        model_path=MODEL_PATH,
        episodes=5,
        render=True,
        deterministic=True,
    )