import gymnasium as gym
from vizdoom import gymnasium_wrapper # Need this to register the envs


class DoomEnv:
    def __init__(self, env_id="VizdoomBasic-v1", render=False):
        if render:
            render_mode = "human"
        else:
            render_mode = None
            
        self.env = gym.make(env_id, render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def sample_action(self):
        return self.env.action_space.sample()

    def close(self):
        self.env.close()