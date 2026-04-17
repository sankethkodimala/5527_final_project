import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from cnn import make_doom_cnn_class, make_vec_env, load_ml_dependencies

def train():
    # Set random seed
    seed = 5527
    set_random_seed(seed)
    
    # Define hyperparameters
    learning_rate = 3e-4
    n_steps = 1024
    batch_size = 64
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    total_timesteps = 2048 # A short run to test training loop works
    
    # Load dependencies and build the custom CNN class
    torch_pkg, nn, PPO_cls, BaseFeaturesExtractor, _, _ = load_ml_dependencies()
    DoomCNN = make_doom_cnn_class(BaseFeaturesExtractor, nn, torch_pkg)
    
    # env initialization
    env = make_vec_env()
    env.seed(seed)
    
    # Setup PPO model with Tensorboard logging
    model = PPO_cls(
        "CnnPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": DoomCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "normalize_images": False,
        },
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        tensorboard_log="./tensorboard_logs/", # TO VIEW: tensorboard --logdir ./tensorboard_logs/
        verbose=1,
        seed=seed
    )
    
    print("Starting training")
    # Train model
    model.learn(total_timesteps=total_timesteps, tb_log_name="ppo_vizdoom", reset_num_timesteps=False)
    
    print("Training finished.")
    model.save("ppo_vizdoom_baseline")
    
    env.close()

if __name__ == "__main__":
    train()
