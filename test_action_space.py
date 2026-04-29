"""
Validates that:
  1. DoomEnv exposes the correct action space when given a custom discrete_actions list
  2. Every discrete action index can be executed without error
  3. Each resulting observation has the expected shape, dtype, and value range
  4. action_names covers every valid index
"""

import sys
import numpy as np

from corridor.doom_env_cooridor import DoomEnv
from basic.actions import BASIC_DISCRETE_ACTIONS, ACTION_NAMES, N_ACTIONS, get_action_name

# ── Constants ────────────────────────────────────────────────────────────────

ENV_ID = "VizdoomBasic-MultiBinary-v1"

# defaults based on preprocessing
EXPECTED_OBS_SHAPE = (4, 84, 84)   # (frame_stack, height, width)
EXPECTED_OBS_DTYPE = np.float32
OBS_MIN = 0.0
OBS_MAX = 1.0

# ── Helpers ──────────────────────────────────────────────────────────────────

def _assert_observation(obs: np.ndarray, label: str) -> None:
    """Raise AssertionError with a clear message if obs doesn't look right."""
    assert obs.shape == EXPECTED_OBS_SHAPE, (
        f"{label}: expected shape {EXPECTED_OBS_SHAPE}, got {obs.shape}"
    )
    assert obs.dtype == EXPECTED_OBS_DTYPE, (
        f"{label}: expected dtype {EXPECTED_OBS_DTYPE}, got {obs.dtype}"
    )
    assert obs.min() >= OBS_MIN - 1e-5, (
        f"{label}: obs.min()={obs.min():.4f} is below {OBS_MIN}"
    )
    assert obs.max() <= OBS_MAX + 1e-5, (
        f"{label}: obs.max()={obs.max():.4f} is above {OBS_MAX}"
    )


# ── Test ─────────────────────────────────────────────────────────────────────

def test_action_space() -> None:
    print(f"Environment : {ENV_ID}")
    print(f"Action count: {N_ACTIONS}")
    print()

    env = DoomEnv(
        env_id=ENV_ID,
        render=False,
        discrete_actions=BASIC_DISCRETE_ACTIONS,
        action_names=ACTION_NAMES,
        preprocess=True,
    )

    # 1. Verify action_space is Discrete(N_ACTIONS)
    assert hasattr(env.action_space, "n"), (
        f"action_space should be Discrete, got {type(env.action_space)}"
    )
    assert env.action_space.n == N_ACTIONS, (
        f"Expected Discrete({N_ACTIONS}), got Discrete({env.action_space.n})"
    )
    print(f"[PASS] action_space  = {env.action_space}")

    # 2. Verify action_names covers every index
    assert env.action_names is not None, "action_names should not be None"
    assert set(env.action_names.keys()) == set(range(N_ACTIONS)), (
        f"action_names keys {set(env.action_names.keys())} "
        f"don't match expected {set(range(N_ACTIONS))}"
    )
    print(f"[PASS] action_names covers all {N_ACTIONS} indices")

    # 3. Reset and verify initial observation
    obs, _info = env.reset(seed=0)
    _assert_observation(obs, label="reset")
    print(f"[PASS] reset obs     shape={obs.shape}  dtype={obs.dtype}  "
          f"range=[{obs.min():.3f}, {obs.max():.3f}]")
    print()

    # 4. Step through every discrete action at least once
    for action_idx in range(N_ACTIONS):
        name = get_action_name(action_idx)
        button_vec = BASIC_DISCRETE_ACTIONS[action_idx]

        obs, reward, terminated, truncated, _info = env.step(action_idx)
        _assert_observation(obs, label=f"step action {action_idx}")

        print(
            f"[PASS] action {action_idx} ({name:<20})  "
            f"buttons={button_vec}  reward={reward:+.3f}"
        )

        if terminated or truncated:
            obs, _info = env.reset(seed=0)

    env.close()
    print()
    print("All action-space checks passed.")


if __name__ == "__main__":
    try:
        test_action_space()
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
