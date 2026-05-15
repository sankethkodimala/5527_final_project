import sys
sys.path.append(".")

from pipeline_mark_no_lstm import DoomMLPipeline


class DefendLinePPOPipeline(DoomMLPipeline):
    """
    PPO pipeline specific to defend_the_line.

    Overrides build_model to use a linear learning-rate schedule instead of
    the shared pipeline's constant lr=1e-4.  The linear decay (1e-4 → 0)
    prevents the late-training policy regression observed in the first run,
    where constant updates kept overshooting a near-optimal policy.
    """

    def build_model(self, seed=None, tensorboard_log="./tensorboard_logs/"):
        torch, nn, PPO, BaseFeaturesExtractor, _, _ = self.load_ml_dependencies()
        DoomCNN = self.make_doom_cnn_class(BaseFeaturesExtractor, nn, torch)
        env = self.make_vec_env(num_envs=8)

        # progress_remaining goes 1.0 → 0.0 over training, so lr decays linearly
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": DoomCNN,
                "features_extractor_kwargs": {"features_dim": 256},
                "normalize_images": False,
            },
            learning_rate=lambda progress: 1e-4 * progress,
            n_steps=1024,
            batch_size=512,
            ent_coef=0.01,
            target_kl=0.05,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=seed,
        )

        return model, env
