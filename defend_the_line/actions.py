from typing import Dict, List

# Button order reported by VizdoomDefendLine-v1:
# [TURN_LEFT, TURN_RIGHT, ATTACK]
DEFEND_LINE_DISCRETE_ACTIONS: List[List[int]] = [
    [0, 0, 0],  # 0: no_op
    [1, 0, 0],  # 1: turn_left
    [0, 1, 0],  # 2: turn_right
    [0, 0, 1],  # 3: shoot
    [1, 0, 1],  # 4: turn_left_shoot
    [0, 1, 1],  # 5: turn_right_shoot
]

DEFEND_LINE_ACTION_NAMES: Dict[int, str] = {
    0: "no_op",
    1: "turn_left",
    2: "turn_right",
    3: "shoot",
    4: "turn_left_shoot",
    5: "turn_right_shoot",
}

N_ACTIONS: int = len(DEFEND_LINE_DISCRETE_ACTIONS)


def get_action_name(action_idx: int) -> str:
    if action_idx not in DEFEND_LINE_ACTION_NAMES:
        raise ValueError(
            f"Invalid action index {action_idx}. "
            f"Valid range: 0-{N_ACTIONS - 1}."
        )
    return DEFEND_LINE_ACTION_NAMES[action_idx]
