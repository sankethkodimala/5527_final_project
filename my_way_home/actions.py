from typing import Dict, List

# Actual button order reported by the environment:
# [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT]

MY_WAY_HOME_DISCRETE_ACTIONS: List[List[int]] = [
    [0, 0, 0, 0, 0],  # 0: no_op
    [1, 0, 0, 0, 0],  # 1: turn_left
    [0, 1, 0, 0, 0],  # 2: turn_right
    [0, 0, 1, 0, 0],  # 3: move_forward
    [0, 0, 0, 1, 0],  # 4: move_left
    [0, 0, 0, 0, 1],  # 5: move_right
    [1, 0, 1, 0, 0],  # 6: turn_left_forward
    [0, 1, 1, 0, 0],  # 7: turn_right_forward
]

MY_WAY_HOME_ACTION_NAMES: Dict[int, str] = {
    0: "no_op",
    1: "turn_left",
    2: "turn_right",
    3: "move_forward",
    4: "move_left",
    5: "move_right",
    6: "turn_left_forward",
    7: "turn_right_forward",
}

N_ACTIONS: int = len(MY_WAY_HOME_DISCRETE_ACTIONS)

def get_action_name(action_idx: int) -> str:
    if action_idx not in MY_WAY_HOME_ACTION_NAMES:
        raise ValueError(
            f"Invalid action index {action_idx}. "
            f"Valid range: 0–{N_ACTIONS - 1}."
        )
    return MY_WAY_HOME_ACTION_NAMES[action_idx]
