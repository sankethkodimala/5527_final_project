import gymnasium as gym
from gymnasium import spaces
from vizdoom import gymnasium_wrapper  # Need this to register the envs
import numpy as np


class DoomEnv:
    def __init__(self, env_id="VizdoomBasic-v1", render=False, discrete_actions=None):
        if render:
            render_mode = "human"
        else:
            render_mode = None

        self.env = gym.make(env_id, render_mode=render_mode)
        self.observation_space = self.env.observation_space

        # Optional discrete action mapping: index -> button vector
        self.discrete_actions = discrete_actions
        if self.discrete_actions is not None:
            if len(self.discrete_actions) == 0:
                raise ValueError("discrete_actions must contain at least one action.")
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        else:
            self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()


    def step(self, action):
        if self.discrete_actions is not None:
            action = np.array(self.discrete_actions[int(action)], dtype=self.env.action_space.dtype)
        return self.env.step(action)

    def sample_action(self):
        return self.action_space.sample()

    def close(self):
        self.env.close()