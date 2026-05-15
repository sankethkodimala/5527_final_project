import argparse
import os

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from basic.actions import BASIC_DISCRETE_ACTIONS
from basic.doom_env import DoomEnv
from corridor.actions import CORRIDOR_DISCRETE_ACTIONS
from corridor.doom_env_corridor import CorridorDoomEnv
from defend_center.actions import DEFEND_CENTER_DISCRETE_ACTIONS
from defend_center.doom_env_defend_center import DefendCenterDoomEnv
from my_way_home.actions import MY_WAY_HOME_DISCRETE_ACTIONS
from my_way_home.doom_env_my_way_home import MyWayHomeDoomEnv

# Import the new standard PPO pipeline
import pipeline_mark_no_lstm as pipeline

def train(
    total_timesteps=1_000_000,
    checkpoint_freq=100_000,
    seed=5527,
    scenario_id="VizdoomBasic-v1",
    environment_class="DoomEnv",
    resume_path=None,
):
    set_random_seed(seed)

    if environment_class == "CorridorDoomEnv":
        scenario_id = "VizdoomDeadlyCorridor-v1"
        discrete_actions = CORRIDOR_DISCRETE_ACTIONS
        environment_class = CorridorDoomEnv
    elif environment_class == "DefendCenterDoomEnv":
        scenario_id = "VizdoomDefendCenter-v1"
        discrete_actions = DEFEND_CENTER_DISCRETE_ACTIONS
        environment_class = DefendCenterDoomEnv
    elif environment_class == "MyWayHomeDoomEnv":
        scenario_id = "VizdoomMyWayHome-v1"
        discrete_actions = MY_WAY_HOME_DISCRETE_ACTIONS
        environment_class = MyWayHomeDoomEnv
    elif environment_class == "DoomEnv":
        scenario_id = "VizdoomBasic-v1"
        discrete_actions = BASIC_DISCRETE_ACTIONS
        environment_class = DoomEnv
    else:
        raise ValueError(
            "Unknown environment_class. Use DoomEnv, CorridorDoomEnv, "
            "DefendCenterDoomEnv, or MyWayHomeDoomEnv."
        )

    ml_pipeline = pipeline.DoomMLPipeline(scenario_id, discrete_actions, environment_class)    
    
    # Change the run name so standard PPO logs don't overwrite RecurrentPPO logs
    run_name = f"standard_ppo_{scenario_id}_v2"

    model, env = ml_pipeline.build_model(seed=seed)
    eval_env = ml_pipeline.make_vec_env(num_envs=1)

    if resume_path is not None:
        if os.path.exists(resume_path):
            print(f"Loading weights from checkpoint: {resume_path}")
            model.set_parameters(resume_path)
        else:
            print(f"Warning: Checkpoint {resume_path} not found. Starting from scratch.")

    checkpoint_dir = os.path.join("checkpoints", scenario_id)
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

    print(f"Starting Standard PPO training for {total_timesteps:,} timesteps")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total environment steps to train for (default: 1M)")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="Save a checkpoint every N steps (default: 100k)")
    parser.add_argument("--seed", type=int, default=5527)
    parser.add_argument("--scenario-id", type=str, default="VizdoomBasic-v1",
                        help="ViZDoom scenario ID (default: VizdoomBasic-v1)")
    parser.add_argument("--environment-class", type=str, default="DoomEnv",
                        help="Environment class to use: DoomEnv, CorridorDoomEnv, DefendCenterDoomEnv, or MyWayHomeDoomEnv")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .zip file to resume training from")

    args = parser.parse_args()

    train(args.timesteps, args.checkpoint_freq, args.seed, args.scenario_id, args.environment_class, args.resume)
