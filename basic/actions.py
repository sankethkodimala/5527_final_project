"""
VizdoomBasic-MultiBinary-v1 exposes a 3-button MultiBinary action space with
button order: [MOVE_LEFT, MOVE_RIGHT, ATTACK].
"""

from typing import Dict, List

# Button order in VizdoomBasic MultiBinary: [MOVE_LEFT, MOVE_RIGHT, ATTACK]
# each inner list is a button vector - 1 = pressed, 0 = not pressed
# idx 0 kept as no-op so the agent can "wait"
BASIC_DISCRETE_ACTIONS: List[List[int]] = [
    [0, 0, 0],  # 0: no-op          — do nothing
    [1, 0, 0],  # 1: move_left      — strafe left
    [0, 1, 0],  # 2: move_right     — strafe right
    [0, 0, 1],  # 3: shoot          — attack without moving
    [1, 0, 1],  # 4: move_left_shoot  — strafe left while shooting
    [0, 1, 1],  # 5: move_right_shoot — strafe right while shooting
]

ACTION_NAMES: Dict[int, str] = {
    0: "no_op",
    1: "move_left",
    2: "move_right",
    3: "shoot",
    4: "move_left_shoot",
    5: "move_right_shoot",
}

N_ACTIONS: int = len(BASIC_DISCRETE_ACTIONS)


def get_action_name(action_idx: int) -> str:
    if action_idx not in ACTION_NAMES:
        raise ValueError(
            f"Invalid action index {action_idx}. "
            f"Valid range: 0–{N_ACTIONS - 1}."
        )
    return ACTION_NAMES[action_idx]