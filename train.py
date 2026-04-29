import argparse
import os

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from basic.actions import BASIC_DISCRETE_ACTIONS
from basic.doom_env import DoomEnv

import pipeline

def train(total_timesteps=1_000_000, checkpoint_freq=100_000, seed=5527):
    set_random_seed(seed)

    ml_pipeline = pipeline.DoomMLPipeline("VizdoomBasic-v1", BASIC_DISCRETE_ACTIONS, DoomEnv)    



    torch_pkg, nn, PPO_cls, BaseFeaturesExtractor, _, _ = ml_pipeline.load_ml_dependencies()
    DoomCNN = ml_pipeline.make_doom_cnn_class(BaseFeaturesExtractor, nn, torch_pkg)

    env = ml_pipeline.make_doom_env(render=False)
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

    model = PPO_cls(
        "CnnPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": DoomCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "normalize_images": False,
        },
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        seed=seed,
    )

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
    args = parser.parse_args()

    train(args.timesteps, args.checkpoint_freq, args.seed)
