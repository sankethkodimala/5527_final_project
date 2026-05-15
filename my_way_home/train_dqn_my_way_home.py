import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vizdoom.gymnasium_wrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

# from my_way_home.actions import MY_WAY_HOME_DISCRETE_ACTIONS
# from my_way_home.doom_env_my_way_home import MyWayHomeDoomEnv
# from my_way_home.pipeline_dqn_my_way_home import MyWayHomeDQNPipeline

from actions import MY_WAY_HOME_DISCRETE_ACTIONS
from doom_env_my_way_home import MyWayHomeDoomEnv
from pipeline_dqn_my_way_home import MyWayHomeDQNPipeline

SCENARIO_ID = "VizdoomMyWayHome-v1"

def train(
    total_timesteps=5_000_000,
    checkpoint_freq=200_000,
    seed=5527,
    resume_path=None,
):
    set_random_seed(seed)

    ml_pipeline = MyWayHomeDQNPipeline(
        SCENARIO_ID, MY_WAY_HOME_DISCRETE_ACTIONS, MyWayHomeDoomEnv
    )
    run_name = f"dqn_{SCENARIO_ID}"

    model, env = ml_pipeline.build_model(seed=seed)
    eval_env = ml_pipeline.make_vec_env(num_envs=1)

    if resume_path is not None:
        if os.path.exists(resume_path):
            print(f"Loading weights from checkpoint: {resume_path}")
            model.set_parameters(resume_path)
        else:
            print(f"Warning: checkpoint {resume_path} not found. Starting from scratch.")

    checkpoint_dir = os.path.join("checkpoints", run_name)
    best_dir = os.path.join(checkpoint_dir, "best")

    os.makedirs(best_dir, exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=run_name,
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=checkpoint_dir,
            eval_freq=checkpoint_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1,
        ),
    ]

    print(f"Starting DQN training: {SCENARIO_ID} for {total_timesteps:,} timesteps")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        reset_num_timesteps=(resume_path is None),
    )

    print("Training finished. Saving final model.")
    model.save(os.path.join(checkpoint_dir, f"{run_name}_final"))

    env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Train DQN on {SCENARIO_ID}")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--checkpoint-freq", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=5527)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    train(
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
        resume_path=args.resume,
    )
