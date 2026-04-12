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
    discrete_actions: List of button vectors, one per discrete action index.
        When provided, the wrapper exposes a Discrete(N) action space and
        converts integer actions to the corresponding button vector before
        passing them to the underlying MultiBinary environment.
        Import BASIC_DISCRETE_ACTIONS from actions.py for the canonical set.
    action_names: Optional dict mapping action index → human-readable name.
        Stored as self.action_names for external inspection; not used internally.
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
        action_names=None,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    ):
        render_mode = "human" if render else None
        self.env = gym.make(env_id, render_mode=render_mode)

        # Action space: either a custom Discrete mapping or the env's native space.
        self.discrete_actions = discrete_actions
        self.action_names = action_names  # optional index → name dict for inspection
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
