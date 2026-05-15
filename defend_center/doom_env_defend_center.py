import numpy as np

from basic.doom_env import DoomEnv


class DefendCenterDoomEnv(DoomEnv):
    """
    A wrapper for the Vizdoom Defend Center environment with light reward shaping.
    """

    STEP_PENALTY = 0.001
    NO_OP_PENALTY = 0.02
    DAMAGE_PENALTY_SCALE = 0.05
    WASTED_SHOT_PENALTY = 0.03
    EMPTY_SHOT_PENALTY = 0.05
    DEATH_PENALTY = 5.0

    def __init__(self, env_id="VizdoomDefendCenter-v1", reward_shaping=True, **kwargs):
        super().__init__(env_id=env_id, **kwargs)
        self.reward_shaping = reward_shaping
        self.prev_health = None
        self.prev_ammo = None
        self.prev_action_idx = None

    def _get_game_variable(self, variable_name):
        try:
            game = self.env.unwrapped.game
            for variable in game.get_available_game_variables():
                if variable.name == variable_name:
                    return game.get_game_variable(variable)
        except Exception:
            return None
        return None

    def _get_health(self):
        return self._get_game_variable("HEALTH")

    def _get_ammo(self):
        return self._get_game_variable("AMMO2")

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_health = self._get_health()
        self.prev_ammo = self._get_ammo()
        self.prev_action_idx = None
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

        shaped_reward = float(reward)
        current_health = self._get_health()
        current_ammo = self._get_ammo()

        damage_taken = 0.0
        ammo_spent = 0.0
        if self.prev_ammo is not None and current_ammo is not None:
            ammo_spent = max(0.0, float(self.prev_ammo - current_ammo))

        if self.reward_shaping:
            shaped_reward -= self.STEP_PENALTY

            if self.prev_health is not None and current_health is not None:
                damage_taken = max(0.0, float(self.prev_health - current_health))
                shaped_reward -= self.DAMAGE_PENALTY_SCALE * damage_taken

            is_no_op = bool(np.sum(action_vec) == 0)
            is_attack = bool(len(action_vec) > 2 and action_vec[2] == 1)

            if is_no_op:
                shaped_reward -= self.NO_OP_PENALTY

            if self.prev_ammo is not None and current_ammo is not None:
                if is_attack and float(reward) <= 0.0:
                    if ammo_spent > 0.0:
                        shaped_reward -= self.WASTED_SHOT_PENALTY
                    elif current_ammo <= 0:
                        shaped_reward -= self.EMPTY_SHOT_PENALTY

        if self.reward_shaping and terminated:
            shaped_reward -= self.DEATH_PENALTY

        self.prev_health = current_health
        self.prev_ammo = current_ammo
        self.prev_action_idx = action_idx

        if info is None:
            info = {}
        info["base_reward"] = float(reward)
        info["shaped_reward"] = float(shaped_reward)
        info["health"] = current_health
        info["ammo"] = current_ammo
        info["damage_taken"] = damage_taken
        info["ammo_spent"] = ammo_spent
        info["action_idx"] = action_idx
        info["reward_shaping"] = self.reward_shaping

        return obs, shaped_reward, terminated, truncated, info

