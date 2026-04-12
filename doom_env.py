import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vizdoom import gymnasium_wrapper
from wrappers import apply_preprocessing


class DoomEnv:
    """
    A wrapper for ViZDoom environments with preprocessing and action mapping.

    env_id: The Gymnasium ViZDoom env ID.
    render: Whether to render in 'human' mode.
    discrete_actions: List of action button combinations (for MultiBinary scenarios).
    preprocess: Whether to apply the preprocessing pipeline.
    resize_shape: Target resolution (height, width).
    grayscale: Whether to convert frames to grayscale.
    frame_stack: Number of frames to stack.
    """
    def __init__(
        self,
        env_id="VizdoomBasic-v1",
        render=False,
        discrete_actions=None,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    ):
        render_mode = "human" if render else None
        self.env = gym.make(env_id, render_mode=render_mode)

        # Action Space Handling (from Aaron's logic)
        self.discrete_actions = discrete_actions
        if self.discrete_actions is not None:
            if len(self.discrete_actions) == 0:
                raise ValueError("discrete_actions must contain at least one action.")
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        else:
            self.action_space = self.env.action_space

        # Preprocessing
        if preprocess:
            self.env = apply_preprocessing(
                self.env,
                resize_shape=resize_shape,
                grayscale=grayscale,
                frame_stack=frame_stack,
            )

        self.observation_space = self.env.observation_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        # Map discrete idx to button vector if discrete_actions is provided
        if self.discrete_actions is not None:
            action = np.array(
                self.discrete_actions[int(action)], dtype=self.env.unwrapped.action_space.dtype
            )
        return self.env.step(action)

    def sample_action(self):
        return self.action_space.sample()

    def close(self):
        self.env.close()
