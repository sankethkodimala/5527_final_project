import importlib
import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_mark_no_lstm import DoomMLPipeline

class MyWayHomeDQNPipeline(DoomMLPipeline):
    """
    DQN pipeline specific to my_way_home.

    DQN is off-policy and uses a replay buffer, so it only supports a single
    training environment (num_envs=1). DummyVecEnv is used instead of
    SubprocVecEnv to avoid multiprocessing overhead for a single env.
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
            optimize_memory_usage=False,
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=seed,
        )

        return model, env
