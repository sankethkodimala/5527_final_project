import importlib

from defend_center.actions import DEFEND_CENTER_DISCRETE_ACTIONS
from defend_center.doom_env_defend_center import DefendCenterDoomEnv


class DefendCenterDQNPipeline:
    """
    DQN pipeline for VizdoomDefendCenter-v1.

    DQN is off-policy and uses a replay buffer, so this keeps a single
    environment in DummyVecEnv. reward_mode controls whether the environment
    returns the original ViZDoom reward or the shaped reward used in later PPO
    experiments.
    """

    def __init__(
        self,
        doom_env_id="VizdoomDefendCenter-v1",
        action_space=None,
        reward_mode="raw",
    ):
        if reward_mode not in {"raw", "shaped"}:
            raise ValueError("reward_mode must be 'raw' or 'shaped'.")

        self.doom_env_id = doom_env_id
        self.action_space = action_space or DEFEND_CENTER_DISCRETE_ACTIONS
        self.reward_mode = reward_mode

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
        return DefendCenterDoomEnv(
            env_id=self.doom_env_id,
            render=render,
            discrete_actions=self.action_space,
            preprocess=True,
            resize_shape=(84, 84),
            grayscale=True,
            frame_stack=4,
            reward_shaping=self.reward_mode == "shaped",
        )

    def make_vec_env(self):
        vec_env = importlib.import_module("stable_baselines3.common.vec_env")
        return vec_env.VecMonitor(vec_env.DummyVecEnv([lambda: self.make_doom_env()]))

    def build_model(self, seed=None, tensorboard_log="./tensorboard_logs/"):
        torch = importlib.import_module("torch")
        nn = torch.nn
        sb3 = importlib.import_module("stable_baselines3")
        torch_layers = importlib.import_module("stable_baselines3.common.torch_layers")

        DoomCNN = self.make_doom_cnn_class(
            torch_layers.BaseFeaturesExtractor,
            nn,
            torch,
        )
        env = self.make_vec_env()

        model = sb3.DQN(
            "CnnPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": DoomCNN,
                "features_extractor_kwargs": {"features_dim": 256},
                "normalize_images": False,
            },
            learning_rate=1e-4,
            buffer_size=5_000,
            learning_starts=1_000,
            batch_size=16,
            gamma=0.99,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            optimize_memory_usage=False,
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=seed,
        )

        return model, env
