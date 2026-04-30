from basic.doom_env import DoomEnv


class DefendCenterDoomEnv(DoomEnv):
    """
    A wrapper for the Vizdoom Defend Center environment
    """

    def __init__(self, env_id="VizdoomDefendCenter-v1", **kwargs):
        super().__init__(env_id=env_id, **kwargs)
