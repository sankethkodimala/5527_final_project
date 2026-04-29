from __future__ import annotations

import importlib
import gymnasium as gym

import matplotlib.pyplot as plt

from basic.actions import BASIC_DISCRETE_ACTIONS
from corridor.doom_env_cooridor import DoomEnv



def load_ml_dependencies(input_doom_env):
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


def make_doom_env(doom_env_id, render=False):
    return DoomEnv(
        env_id=doom_env_id,
        render=render,
        discrete_actions=BASIC_DISCRETE_ACTIONS,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    )


def make_vec_env(doom_env_id):
    _, _, _, _, DummyVecEnv, VecMonitor = load_ml_dependencies()

    return VecMonitor(DummyVecEnv([lambda: make_doom_env(doom_env_id)]))


def build_model(doom_env_id):
    torch, nn, PPO, BaseFeaturesExtractor, _, _ = load_ml_dependencies()
    DoomCNN = make_doom_cnn_class(BaseFeaturesExtractor, nn, torch)
    env = make_vec_env(doom_env_id)

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
