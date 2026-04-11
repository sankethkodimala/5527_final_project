import gymnasium as gym
from vizdoom import gymnasium_wrapper


class DoomEnv:
    def __init__(self, env_id="VizdoomBasic-v1", render=False, discrete_actions=None):
        render_mode = "human" if render else None
        self.env = gym.make(env_id, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def sample_action(self):
        return self.action_space.sample()

    def close(self):
        self.env.close()