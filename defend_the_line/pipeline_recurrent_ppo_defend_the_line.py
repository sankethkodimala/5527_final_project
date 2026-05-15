import sys
sys.path.append(".")

from pipeline_mark import DoomMLPipeline


class DefendLineRecurrentPPOPipeline(DoomMLPipeline):
    """
    Recurrent PPO pipeline specific to defend_the_line.

    This mirrors the standard PPO setup for this scenario but swaps in
    CnnLstmPolicy from sb3-contrib so the final comparison has an explicit
    LSTM baseline.
    """

    def build_model(self, seed=None, tensorboard_log="./tensorboard_logs/"):
        torch, nn, RecurrentPPO, BaseFeaturesExtractor, _, _ = self.load_ml_dependencies()
        DoomCNN = self.make_doom_cnn_class(BaseFeaturesExtractor, nn, torch)
        env = self.make_vec_env(num_envs=8)

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
