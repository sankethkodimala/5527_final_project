"""
VizdoomPredictPosition-MultiBinary-v1 exposes a 3-button MultiBinary action space
with button order: [TURN_LEFT, TURN_RIGHT, ATTACK].
"""

from typing import Dict, List

# Button order in VizdoomPredictPosition MultiBinary: [TURN_LEFT, TURN_RIGHT, ATTACK]
# Each inner list is a button vector - 1 = pressed, 0 = not pressed.
PREDICT_POSITION_DISCRETE_ACTIONS: List[List[int]] = [
	[1, 0, 0],  # 1: turn_left
	[0, 1, 0],  # 2: turn_right
	[0, 0, 1],  # 3: shoot
	[1, 0, 1],  # 4: turn_left_shoot
	[0, 1, 1],  # 5: turn_right_shoot
]

ACTION_NAMES: Dict[int, str] = {
	0: "turn_left",
	1: "turn_right",
	2: "shoot",
	3: "turn_left_shoot",
	4: "turn_right_shoot",
}

N_ACTIONS: int = len(PREDICT_POSITION_DISCRETE_ACTIONS)


def get_action_name(action_idx: int) -> str:
	if action_idx not in ACTION_NAMES:
		raise ValueError(
			f"Invalid action index {action_idx}. "
			f"Valid range: 0–{N_ACTIONS - 1}."
		)
	return ACTION_NAMES[action_idx]
