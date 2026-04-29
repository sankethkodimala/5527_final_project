from __future__ import annotations

import importlib
import gymnasium as gym

import matplotlib.pyplot as plt

from basic.actions import BASIC_DISCRETE_ACTIONS
from corridor.doom_env_cooridor import DoomEnv


def load_ml_dependencies():
    torch = importlib.import_module("torch")
    sb3 = importlib.import_module("stable_baselines3")
    torch_layers = importlib.import_module("stable_baselines3.common.torch_layers")
    vec_env = importlib.import_module("stable_baselines3.common.vec_env")

    return (
        torch,
        torch.nn,
        sb3.PPO,
        torch_layers.BaseFeaturesExtractor,
        vec_env.DummyVecEnv,
        vec_env.VecMonitor,
    )


def make_doom_cnn_class(BaseFeaturesExtractor, nn, torch):
    class DoomCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=256):
            super().__init__(observation_space, features_dim)

            shape = observation_space.shape
            if shape is None or len(shape) != 3:
                raise ValueError(f"Expected 3D image observations, got {shape!r}")

            self.channels_first = shape[0] <= 16
            n_input_channels = shape[0] if self.channels_first else shape[-1]

            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                sample = observation_space.sample()
                sample_tensor = torch.as_tensor(sample[None]).float()
                if not self.channels_first:
                    sample_tensor = sample_tensor.permute(0, 3, 1, 2)
                n_flatten = self.cnn(sample_tensor).shape[1]

            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations):
            if not self.channels_first:
                observations = observations.permute(0, 3, 1, 2)
            return self.linear(self.cnn(observations))

    return DoomCNN


def make_doom_env(render=False):
    return DoomEnv(
        env_id="VizdoomDeadlyCorridor-MultiBinary-v1",    
        render=render,
        discrete_actions=BASIC_DISCRETE_ACTIONS,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    )


def make_vec_env():
    _, _, _, _, DummyVecEnv, VecMonitor = load_ml_dependencies()

    return VecMonitor(DummyVecEnv([make_doom_env]))


def build_model():
    torch, nn, PPO, BaseFeaturesExtractor, _, _ = load_ml_dependencies()
    DoomCNN = make_doom_cnn_class(BaseFeaturesExtractor, nn, torch)
    env = make_vec_env()

    model = PPO(
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
        verbose=1,
    )

    return model, env

def evaluate(model, episodes=3):
    env = make_doom_env(render=True)
    rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep} reward: {total_reward}")
        rewards.append(total_reward)

    env.close()
    return rewards

def train_model(model):
    eval_rewards = []
    checkpoints = []

    for i in range(10):
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
        print(f"Completed training iteration {i+1}/10")

        rewards = evaluate(model, episodes=1)
        eval_rewards.append(rewards[0])
        checkpoints.append((i + 1) * 100000)

    model.save("ppo_vizdoom_model")

    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, eval_rewards, marker="o")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Evaluation Reward")
    plt.title("PPO ViZDoom Evaluation Reward")
    plt.grid(True)
    plt.savefig("vizdoom_eval_reward.png")
    plt.show()


if __name__ == "__main__":
    model, env = build_model()
    print("CNN + PPO pipeline created successfully.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(model.policy)

    train_model(model)
    evaluate(model)