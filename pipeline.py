from __future__ import annotations

import importlib
import gymnasium as gym

import matplotlib.pyplot as plt
import numpy as np

from basic.actions import BASIC_DISCRETE_ACTIONS

class DoomMLPipeline:
    def __init__(self, doom_env_id, action_space, environment_class):
        self.doom_env_id = doom_env_id
        self.action_space = action_space
        self.environment_class = environment_class

    def load_ml_dependencies(self):
        torch = importlib.import_module("torch")
        sb3 = importlib.import_module("sb3_contrib")
        torch_layers = importlib.import_module("stable_baselines3.common.torch_layers")
        vec_env = importlib.import_module("stable_baselines3.common.vec_env")

        return (
            torch,
            torch.nn,
            sb3.RecurrentPPO,
            torch_layers.BaseFeaturesExtractor,
            vec_env.DummyVecEnv,
            vec_env.VecMonitor,
        )


    def make_doom_cnn_class(self, BaseFeaturesExtractor, nn, torch):
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


    def make_doom_env(self, render=False):
        return self.environment_class(
            env_id=self.doom_env_id,
            render=render,
            discrete_actions=self.action_space,
            preprocess=True,
            resize_shape=(84, 84),
            grayscale=True,
            frame_stack=4,
        )


    def make_vec_env(self):
        _, _, _, _, DummyVecEnv, VecMonitor = self.load_ml_dependencies()

        return VecMonitor(DummyVecEnv([lambda: self.make_doom_env()]))


    def build_model(self, seed=None, tensorboard_log="./tensorboard_logs/"):
        torch, nn, RecurrentPPO, BaseFeaturesExtractor, _, _ = self.load_ml_dependencies()
        DoomCNN = self.make_doom_cnn_class(BaseFeaturesExtractor, nn, torch)
        env = self.make_vec_env()

        model = RecurrentPPO(
            "CnnLstmPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": DoomCNN,
                "features_extractor_kwargs": {"features_dim": 256},
                "normalize_images": False,
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm": True,
            },
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=seed,
        )

        return model, env
