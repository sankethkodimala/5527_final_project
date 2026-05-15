import argparse
import importlib
import os

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from defend_the_line.actions import DEFEND_LINE_DISCRETE_ACTIONS
from defend_the_line.doom_env_defend_the_line import DefendLineDoomEnv
from defend_the_line.pipeline_recurrent_ppo_defend_the_line import (
    DefendLineRecurrentPPOPipeline,
)

SCENARIO_ID = "VizdoomDefendLine-v1"


def train(
    total_timesteps=10_000_000,
    checkpoint_freq=100_000,
    seed=5527,
    resume_path=None,
):
    set_random_seed(seed)

    ml_pipeline = DefendLineRecurrentPPOPipeline(
        SCENARIO_ID, DEFEND_LINE_DISCRETE_ACTIONS, DefendLineDoomEnv
    )
    run_name = f"recurrent_ppo_{SCENARIO_ID}"

    if resume_path is not None and os.path.exists(resume_path):
        env = ml_pipeline.make_vec_env(num_envs=8)
        sb3_contrib = importlib.import_module("sb3_contrib")
        print(f"Loading full model from checkpoint: {resume_path}")
        model = sb3_contrib.RecurrentPPO.load(
            resume_path,
            env=env,
            tensorboard_log="./tensorboard_logs/",
            seed=seed,
        )
    else:
        model, env = ml_pipeline.build_model(seed=seed)

    eval_env = ml_pipeline.make_vec_env(num_envs=1)

    if resume_path is not None and not os.path.exists(resume_path):
        print(f"Warning: checkpoint {resume_path} not found. Starting from scratch.")

    checkpoint_dir = os.path.join("checkpoints", f"recurrent_ppo_{SCENARIO_ID}")
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
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            verbose=1,
        ),
    ]

    print(f"Starting recurrent PPO training: {SCENARIO_ID} for {total_timesteps:,} timesteps")
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
    parser = argparse.ArgumentParser(
        description=f"Train recurrent PPO/LSTM on {SCENARIO_ID}"
    )
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total environment steps (default: 10M)")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="Checkpoint interval in steps (default: 100k)")
    parser.add_argument("--seed", type=int, default=5527)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .zip to resume from")

    args = parser.parse_args()
    train(args.timesteps, args.checkpoint_freq, args.seed, args.resume)
