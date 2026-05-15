import sys

import gymnasium as gym
from vizdoom import gymnasium_wrapper  # registers all ViZDoom envs with gymnasium
from gymnasium import spaces
import numpy as np

sys.path.append(".")
from wrappers import apply_preprocessing


class DefendLineDoomEnv(gym.Env):
    """
    Wrapper for VizdoomDefendLine-v1 with reward shaping.

    The native reward is +1 per kill and -1 on death. Shaping adds a small
    bonus for firing and a penalty for taking damage and action repetition so
    the agent learns to actively engage rather than spin idly.

    Button order: [TURN_LEFT, TURN_RIGHT, ATTACK]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_id="VizdoomDefendLine-v1",
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
        self.prev_health = None
        self.prev_action_idx = None

    def _get_health(self):
        try:
            return self.env.unwrapped.game.get_game_variable(
                __import__("vizdoom").GameVariable.HEALTH
            )
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

        # penalize taking damage — capped at 10 HP per step to prevent a single
        # burst-damage frame from dominating the signal over many kill rewards
        if self.prev_health is not None and current_health is not None:
            damage_taken = self.prev_health - current_health
            if damage_taken > 0:
                shaped_reward -= 0.5 * min(damage_taken, 10)

        # bonus for pressing ATTACK (index 2 in [TURN_LEFT, TURN_RIGHT, ATTACK])
        if action_vec[2] == 1:
            shaped_reward += 0.1

        # discourage spinning in place by penalizing repeating the same action
        if self.prev_action_idx is not None and self.prev_action_idx == action_idx:
            shaped_reward -= 0.01

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
