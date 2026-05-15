import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.append(".")
from wrappers import apply_preprocessing

class MyWayHomeDoomEnv(gym.Env):
    """
    A wrapper for ViZDoom My Way Home environment with preprocessing and action mapping.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_id="VizdoomMyWayHome-v1",
        render=False,
        discrete_actions=None,
        action_names=None,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4,
    ):
        super().__init__()

        render_mode = "human" if render else None
        
        self.env = gym.make(
            env_id,
            render_mode=render_mode,
            max_buttons_pressed=0,
            use_multi_binary_action_space=True,
        )

        self.discrete_actions = discrete_actions
        self.action_names = action_names

        if self.discrete_actions is not None:
            if len(self.discrete_actions) == 0:
                raise ValueError("discrete_actions must contain at least one action.")
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        else:
            self.action_space = self.env.action_space

        if preprocess:
            self.env = apply_preprocessing(
                self.env,
                resize_shape=resize_shape,
                grayscale=grayscale,
                frame_stack=frame_stack,
            )

        self.observation_space = self.env.observation_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        if self.discrete_actions is not None:
            action_idx = int(action)
            action_vec = np.array(self.discrete_actions[action_idx], dtype=np.int8)
        else:
            action_idx = None
            action_vec = action

        obs, reward, terminated, truncated, info = self.env.step(action_vec)

        # "My Way Home" has +1 for finding the vest, and -0.0001 per tick.
        # So we use the environment's base reward without manual shaping
        # as LSTM should be capable of handling this sparse reward given enough time.
        shaped_reward = float(reward)

        if info is None:
            info = {}

        info["base_reward"] = float(reward)
        info["shaped_reward"] = float(shaped_reward)
        info["action_idx"] = action_idx

        return obs, shaped_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def sample_action(self):
        return self.action_space.sample()

    def close(self):
        self.env.close()
