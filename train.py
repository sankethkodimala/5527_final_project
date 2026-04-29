import argparse
import os

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from basic.actions import BASIC_DISCRETE_ACTIONS
from basic.doom_env import DoomEnv
from corridor.actions import CORRIDOR_DISCRETE_ACTIONS
from corridor.doom_env_corridor import CorridorDoomEnv

import pipeline

def train(total_timesteps=1_000_000, checkpoint_freq=100_000, seed=5527, scenario_id="VizdoomBasic-v1", environment_class=DoomEnv):
    set_random_seed(seed)

    if environment_class == CorridorDoomEnv:
        scenario_id = "VizdoomDeadlyCorridor-v1"
        discrete_actions = CORRIDOR_DISCRETE_ACTIONS

    if environment_class == DoomEnv:
        scenario_id = "VizdoomBasic-v1"
        discrete_actions = BASIC_DISCRETE_ACTIONS

    ml_pipeline = pipeline.DoomMLPipeline(scenario_id, discrete_actions, environment_class)    
    



    torch_pkg, nn, PPO_cls, BaseFeaturesExtractor, _, _ = ml_pipeline.load_ml_dependencies()
    DoomCNN = ml_pipeline.make_doom_cnn_class(BaseFeaturesExtractor, nn, torch_pkg)

    model, env = ml_pipeline.build_model()
    eval_env = ml_pipeline.make_vec_env()


    os.makedirs("checkpoints/best", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="./checkpoints/",
            name_prefix="ppo_vizdoom",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path="./checkpoints/best/",
            log_path="./checkpoints/",
            eval_freq=checkpoint_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1,
        ),
    ]

    print(f"Starting training for {total_timesteps:,} timesteps")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_vizdoom",
        reset_num_timesteps=True,
    )

    print("Training finished. Saving final model.")
    model.save("ppo_vizdoom_baseline")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total environment steps to train for (default: 1M)")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="Save a checkpoint every N steps (default: 100k)")
    parser.add_argument("--seed", type=int, default=5527)
    parser.add_argument("--scenario-id", type=str, default="VizdoomBasic-v1",
                        help="ViZDoom scenario ID (default: VizdoomBasic-v1)")
    parser.add_argument("--environment-class", type=str, default="DoomEnv",
                        help="Environment class to use (default: DoomEnv)")

    args = parser.parse_args()

    train(args.timesteps, args.checkpoint_freq, args.seed)
