import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vizdoom import gymnasium_wrapper

sys.path.append(".")
from wrappers import apply_preprocessing


class PositionDoomEnv(gym.Env):
	"""
	A wrapper for ViZDoom Predict Position environments with preprocessing,
	discrete action mapping, and reward shaping.

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

	# Reward shaping penalties and bonuses
	STEP_PENALTY = 0.001
	NO_OP_PENALTY = 0.01
	WASTED_SHOT_PENALTY = 0.02
	EMPTY_SHOT_PENALTY = 0.02
	DEATH_PENALTY = 2.0
	HIT_BONUS = 1.0  # Extra bonus for successful hits

	def __init__(
		self,
		env_id="VizdoomPredictPosition-MultiBinary-v1",
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
		self.prev_ammo = None

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

	def _get_game_variable(self, variable_name):
		"""Get a game variable value from the ViZDoom game state."""
		try:
			game = self.env.unwrapped.game
			for variable in game.get_available_game_variables():
				if variable.name == variable_name:
					return game.get_game_variable(variable)
		except Exception:
			return None
		return None

	def _get_ammo(self):
		"""Get current ammo count."""
		return self._get_game_variable("AMMO2")

	def reset(self, *, seed=None, options=None):
		obs, info = self.env.reset(seed=seed, options=options)
		self.prev_ammo = self._get_ammo()
		return obs, info

	def step(self, action):
		if self.discrete_actions is not None:
			action_idx = int(action)
			action_vec = np.array(
				self.discrete_actions[action_idx],
				dtype=self.env.unwrapped.action_space.dtype,
			)
		else:
			action_idx = None
			action_vec = action

		obs, reward, terminated, truncated, info = self.env.step(action_vec)

		if info is None:
			info = {}

		info["base_reward"] = float(reward)
		info["shaped_reward"] = float(reward)
		info["action_idx"] = action_idx

		return obs, float(reward), terminated, truncated, info

	def render(self):
		return self.env.render()

	def sample_action(self):
		return self.action_space.sample()

	def close(self):
		self.env.close()
