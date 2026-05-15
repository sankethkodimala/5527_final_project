import importlib
import sys
sys.path.append(".")

from pipeline_mark_no_lstm import DoomMLPipeline


class DefendLineDQNPipeline(DoomMLPipeline):
    """
    DQN pipeline specific to defend_the_line.

    DQN is off-policy and uses a replay buffer, so it only supports a single
    training environment (num_envs=1). DummyVecEnv is used instead of
    SubprocVecEnv to avoid multiprocessing overhead for a single env.

    Key differences from PPO:
    - buffer_size: stores past transitions for off-policy updates
    - learning_starts: fills the buffer with random transitions before training
    - target_update_interval: how often to sync the target network
    - exploration_fraction: fraction of training spent on epsilon-greedy decay
    - optimize_memory_usage: halves buffer RAM by not storing next_obs separately
    """

    def make_vec_env(self, num_envs=1):
        vec_env = importlib.import_module("stable_baselines3.common.vec_env")
        return vec_env.VecMonitor(
            vec_env.DummyVecEnv([lambda: self.make_doom_env()])
        )

    def build_model(self, seed=None, tensorboard_log="./tensorboard_logs/"):
        torch, nn, _, BaseFeaturesExtractor, _, _ = self.load_ml_dependencies()
        sb3 = importlib.import_module("stable_baselines3")
        DoomCNN = self.make_doom_cnn_class(BaseFeaturesExtractor, nn, torch)
        env = self.make_vec_env(num_envs=1)

        model = sb3.DQN(
            "CnnPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": DoomCNN,
                "features_extractor_kwargs": {"features_dim": 256},
                "normalize_images": False,
            },
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=10_000,
            batch_size=32,
            gamma=0.99,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            optimize_memory_usage=True,
            replay_buffer_kwargs={"handle_timeout_termination": False},
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=seed,
        )

        return model, env
