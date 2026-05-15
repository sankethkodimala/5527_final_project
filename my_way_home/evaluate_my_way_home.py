import importlib
import os
import sys
import vizdoom.gymnasium_wrapper

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from actions import MY_WAY_HOME_DISCRETE_ACTIONS, MY_WAY_HOME_ACTION_NAMES
from doom_env_my_way_home import MyWayHomeDoomEnv

MODEL_PATH = "my_way_home/ppo_lstm_vizdoom_model.zip"
ENV_ID = "VizdoomMyWayHome-v1"

def load_recurrent_ppo():
    sb3_contrib = importlib.import_module("sb3_contrib")
    return sb3_contrib.RecurrentPPO

def make_doom_env(render=True):
    return MyWayHomeDoomEnv(
        env_id=ENV_ID,
        render=render,
        discrete_actions=MY_WAY_HOME_DISCRETE_ACTIONS,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    )

def evaluate(model_path=MODEL_PATH, episodes=3, render=True, deterministic=True):
    RecurrentPPO = load_recurrent_ppo()
    try:
        model = RecurrentPPO.load(model_path)
    except Exception as e:
        print(f"Model could not be loaded: {e}\nEnsure sb3_contrib is installed and the model file exists.")
        return

    env = make_doom_env(render=render)

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            step = 0
            
            # RecurrentPPO requires hidden state initialization
            lstm_states = None
            episode_starts = [True]

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
                total_reward += reward
                episode_starts = [done]

                action_name = MY_WAY_HOME_ACTION_NAMES.get(action_idx, f"action_{action_idx}")
                mapped = MY_WAY_HOME_DISCRETE_ACTIONS[action_idx]

                print(
                    f"episode={ep + 1} "
                    f"step={step} "
                    f"action={action_idx} "
                    f"name={action_name} "
                    f"mapped={mapped} "
                    f"reward={reward:.4f}"
                )

                step += 1

            print(f"Episode {ep + 1} total reward: {total_reward:.4f}\n")

    finally:
        env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to the trained .zip model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        render=not args.no_render,
        deterministic=True,
    )
