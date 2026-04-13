from typing import Dict, List

# Actual button order reported by the environment:
# [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]

BASIC_DISCRETE_ACTIONS: List[List[int]] = [
    [0, 0, 0, 0, 0, 0, 0],  # 0: no_op

    [0, 0, 0, 1, 0, 0, 0],  # 1: forward
    [0, 0, 0, 0, 1, 0, 0],  # 2: backward

    [1, 0, 0, 0, 0, 0, 0],  # 3: strafe_left
    [0, 1, 0, 0, 0, 0, 0],  # 4: strafe_right

    [0, 0, 0, 0, 0, 1, 0],  # 5: turn_left
    [0, 0, 0, 0, 0, 0, 1],  # 6: turn_right

    [0, 0, 1, 0, 0, 0, 0],  # 7: shoot

    [0, 0, 1, 1, 0, 0, 0],  # 8: forward_shoot
    [0, 0, 1, 0, 0, 1, 0],  # 9: turn_left_shoot
    [0, 0, 1, 0, 0, 0, 1],  # 10: turn_right_shoot

    [0, 0, 0, 1, 0, 1, 0],  # 11: forward_turn_left
    [0, 0, 0, 1, 0, 0, 1],  # 12: forward_turn_right

    [0, 0, 1, 1, 0, 1, 0],  # 13: forward_turn_left_shoot
    [0, 0, 1, 1, 0, 0, 1],  # 14: forward_turn_right_shoot
]

ACTION_NAMES: Dict[int, str] = {
    0: "no_op",
    1: "forward",
    2: "backward",
    3: "strafe_left",
    4: "strafe_right",
    5: "turn_left",
    6: "turn_right",
    7: "shoot",
    8: "forward_shoot",
    9: "turn_left_shoot",
    10: "turn_right_shoot",
    11: "forward_turn_left",
    12: "forward_turn_right",
    13: "forward_turn_left_shoot",
    14: "forward_turn_right_shoot",
}

N_ACTIONS: int = len(BASIC_DISCRETE_ACTIONS)


def get_action_name(action_idx: int) -> str:
    if action_idx not in ACTION_NAMES:
        raise ValueError(
            f"Invalid action index {action_idx}. "
            f"Valid range: 0–{N_ACTIONS - 1}."
        )
    return ACTION_NAMES[action_idx]