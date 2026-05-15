import importlib
from vizdoom import gymnasium_wrapper

from actions import PREDICT_POSITION_DISCRETE_ACTIONS, ACTION_NAMES
from doom_env_position import PositionDoomEnv

MODEL_PATH = "checkpoints/VizdoomPredictPosition-MultiBinary-v1/best/recurrent_ppo_VizdoomPredictPosition-MultiBinary-v1_best_model.zip"
ENV_ID = "VizdoomPredictPosition-MultiBinary-v1"


def load_ppo():
    sb3 = importlib.import_module("stable_baselines3")
    return sb3.PPO


def make_doom_env(render=True):
    return PositionDoomEnv(
        env_id=ENV_ID,
        render=render,
        discrete_actions=PREDICT_POSITION_DISCRETE_ACTIONS,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    )


def evaluate(model_path=MODEL_PATH, episodes=5, render=True, deterministic=True):
    PPO = load_ppo()
    model = PPO.load(model_path)

    env = make_doom_env(render=render)

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            total_shaped_reward = 0.0
            step = 0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                action_idx = int(action)

                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated
                total_reward += info["base_reward"]
                total_shaped_reward += reward

                action_name = ACTION_NAMES.get(action_idx, f"action_{action_idx}")
                mapped = PREDICT_POSITION_DISCRETE_ACTIONS[action_idx]

                print(
                    f"episode={ep + 1} "
                    f"step={step} "
                    f"action={action_idx} "
                    f"name={action_name:<18} "
                    f"mapped={mapped} "
                    f"base_reward={info['base_reward']:+.3f} "
                    f"shaped_reward={reward:+.3f}"
                )

                step += 1

            print(f"Episode {ep + 1} base reward: {total_reward:.3f}, shaped reward: {total_shaped_reward:.3f}\n")

    finally:
        env.close()


if __name__ == "__main__":
    evaluate(
        model_path=MODEL_PATH,
        episodes=5,
        render=True,
        deterministic=True,
    )
