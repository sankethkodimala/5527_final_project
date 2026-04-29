import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation


class ScreenOnlyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if "screen" not in env.observation_space.spaces:
            raise KeyError("Observation space does not contain 'screen'")
        self.observation_space = env.observation_space.spaces["screen"]
    
    def observation(self, observation):
        return observation["screen"]


class RescaleObservation(gym.ObservationWrapper):
    def __init__(self, env, min_obs=0.0, max_obs=1.0):
        super().__init__(env)
        self.min_obs = min_obs
        self.max_obs = max_obs
        self.observation_space = spaces.Box(
            low=self.min_obs,
            high=self.max_obs,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        # Use bounds of the input space for normalization
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        
        # Scale to [0, 1] based on the true source range, then scale to [min_obs, max_obs]
        obs_float = observation.astype(np.float32)
        return ((obs_float - low) / (high - low)) * (self.max_obs - self.min_obs) + self.min_obs


def apply_preprocessing(env, resize_shape=(84, 84), grayscale=True, frame_stack=4):
    """
    Applies common RL preprocessing wrappers to a ViZDoom environment.
    """
    # Extract the screen from the dictionary
    env = ScreenOnlyObservation(env)

    # Resize the observation
    if resize_shape:
        env = ResizeObservation(env, resize_shape)

    # Convert to Grayscale
    if grayscale:
        env = GrayscaleObservation(env, keep_dim=False)

    # Scale pixel values to [0.0, 1.0]
    env = RescaleObservation(env, min_obs=0.0, max_obs=1.0)

    # Stack frames for temporal context
    if frame_stack > 1:
        env = FrameStackObservation(env, stack_size=frame_stack)

    return env
