"""Global constants, parameters, etc."""

MAX_AVAILABLE_AGENT_IN_CELL = 5

MIN_PROB_OF_TRANS_TO_BASE = 0.01
MIN_PROB_OF_TRANS_TO_RIOT = 0.01
MIN_PROB_OF_TRANS_TO_INJURED = 0.0

INITIAL_PROB_OF_RIOT = 0.1
INITIAL_PROB_OF_BASE = 1 - INITIAL_PROB_OF_RIOT
INITIAL_ROUND_OF_ENTRY_HOME = 1000
INITIAL_ROUND_OF_ENTRY_AWAY = 500

DOWNWARD_EXTRA_WEIGHT = 5

INJURY_MINIMUM_AGENT_THD = 3
MAX_INJURY_PROB = 0.3

ROW_MASKS = {"bot": [6, 7, 8], "mid": [3, 4, 5], "top": [0, 1, 2]}

MOVEMENT_ARGUMENT = [
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
