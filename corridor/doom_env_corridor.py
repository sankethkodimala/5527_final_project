import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
sys.path.append(".")
from wrappers import apply_preprocessing


class CorridorDoomEnv(gym.Env):
    """
    A wrapper for ViZDoom environments with preprocessing, discrete action
    mapping, and light reward shaping.

    env_id: The Gymnasium ViZDoom env ID.
    render: Whether to render in 'human' mode.
    discrete_actions: List of button vectors, one per discrete action index.
        When provided, the wrapper exposes a Discrete(N) action space and
        converts integer actions to the corresponding button vector before
        passing them to the underlying MultiBinary environment.
    action_names: Optional dict mapping action index -> human-readable name.
    preprocess: Whether to apply the preprocessing pipeline.
    resize_shape: Target resolution (height, width).
    grayscale: Whether to convert frames to grayscale.
    frame_stack: Number of frames to stack.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_id="VizdoomDeadlyCorridor-MultiBinary-v1",
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
        print("Underlying action space:", self.env.unwrapped.action_space)
        print("Available buttons:", self.env.unwrapped.game.get_available_buttons())

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
        self.prev_health = None
        self.prev_action_idx = None

    def _get_health(self):
        try:
            variables = self.env.unwrapped.game.get_available_game_variables()
            if len(variables) == 0:
                return None
            return self.env.unwrapped.get_game_variable(variables[0])
        except Exception:
            return None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_health = self._get_health()
        self.prev_action_idx = None
        return obs, info

    def step(self, action):
        if self.discrete_actions is not None:
            action_idx = int(action)
            action_vec = np.array(self.discrete_actions[action_idx], dtype=np.int8)
        else:
            action_idx = None
            action_vec = action

        obs, reward, terminated, truncated, info = self.env.step(action_vec)

        shaped_reward = float(reward)
        current_health = self._get_health()

        # Stronger penalty for taking damage
        if self.prev_health is not None and current_health is not None:
            damage_taken = self.prev_health - current_health
            if damage_taken > 0:
                shaped_reward -= 2.0 * damage_taken

        # Small reward for using attack
        # Actual ATTACK index in the env order:
        # [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]
        if action_vec[2] == 1:
            shaped_reward += 0.05

        # Small penalty for moving forward without attacking
        if action_vec[3] == 1 and action_vec[2] == 0:
            shaped_reward -= 0.05

        # Optional: discourage doing exactly the same action repeatedly
        if hasattr(self, "prev_action_idx") and self.prev_action_idx == action_idx:
            shaped_reward -= 0.01

        # Big death penalty
        if terminated:
            shaped_reward -= 50.0

        self.prev_health = current_health
        self.prev_action_idx = action_idx

        if info is None:
            info = {}

        info["base_reward"] = float(reward)
        info["shaped_reward"] = float(shaped_reward)
        info["health"] = current_health
        info["action_idx"] = action_idx

        return obs, shaped_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def sample_action(self):
        return self.action_space.sample()

    def close(self):
        self.env.close()