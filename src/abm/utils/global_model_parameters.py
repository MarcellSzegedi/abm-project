"""Global constants, parameters, etc."""

from typing import TypedDict


class MovementArgs(TypedDict):
    """Arguments for the function that decides whether a bystander agent can move to a cell."""

    row_to_check: str
    only_opp_rioter: bool


MAX_AVAILABLE_AGENT_IN_CELL = 5

INITIAL_PROB_OF_RIOT = 0.1
INITIAL_PROB_OF_BASE = 1 - INITIAL_PROB_OF_RIOT


INJURY_MINIMUM_AGENT_THD = 3
MAX_INJURY_PROB = 0.1

RIOT_MINIMUM_AGENT_THD = 2

STEP_THD = 7

ROW_MASKS = {"bot": [6, 7, 8], "mid": [3, 4, 5], "top": [0, 1, 2]}

MOVEMENT_ARGUMENTS: list[MovementArgs] = [
    {"row_to_check": "bot", "only_opp_rioter": False},
    {"row_to_check": "mid", "only_opp_rioter": False},
    {"row_to_check": "top", "only_opp_rioter": False},
    {"row_to_check": "bot", "only_opp_rioter": True},
    {"row_to_check": "mid", "only_opp_rioter": True},
    {"row_to_check": "top", "only_opp_rioter": True},
]
ROW_ADJUSTMENT_MAP = {"bot": 1, "mid": 0, "top": -1}
ROW_FILTERING_CONDITIONS = {
    "bot": lambda row_cord, curr_pos: row_cord > curr_pos,
    "mid": lambda row_cord, curr_pos: row_cord == curr_pos,
    "top": lambda row_cord, curr_pos: row_cord < curr_pos,
}

AGENT_ANIM_PLACEMENT_COL = {
    1: [0.5],
    2: [0.2, 0.8],
    3: [0.2, 0.5, 0.8],
    4: [0.2, 0.2, 0.8, 0.8],
    5: [0.2, 0.2, 0.5, 0.8, 0.8],
}
AGENT_ANIM_PLACEMENT_ROW = {
    1: [0.5],
    2: [0.5, 0.5],
    3: [0.2, 0.5, 0.2],
    4: [0.2, 0.8, 0.2, 0.8],
    5: [0.2, 0.8, 0.5, 0.2, 0.8],
}