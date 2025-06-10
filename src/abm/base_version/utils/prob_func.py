"""Utility functions to calculate the probabilities of the agents."""

import math

import numpy as np
import numpy.typing as npt

from abm.base_version.utils.global_model_parameters import MAX_AVAILABLE_AGENT_IN_CELL


def state_change_probability(
    n_of_in_state_agents: int, n_of_available_cell: int, k: float = 5
) -> float:
    """Calculates the relative probability of a state change."""
    agent_state_ratio = n_of_in_state_agents / (n_of_available_cell * MAX_AVAILABLE_AGENT_IN_CELL)
    agent_state_ratio = min(1.0, max(0.0, agent_state_ratio))
    return 1 / (1 + math.exp(-k * (agent_state_ratio - 0.5)))


def cell_movement_probability(
    n_of_in_team_agents: npt.NDArray[int], n_of_available_cell: int, k: float = 5
) -> npt.NDArray[float]:
    """Calculates the relative probability of an agent moving to a cell."""
    agent_team_ratios = n_of_in_team_agents / (n_of_available_cell * MAX_AVAILABLE_AGENT_IN_CELL)
    agent_team_ratios = np.clip(agent_team_ratios, 0.0, 1.0)
    return 1 / (1 + np.exp(-k * (agent_team_ratios - 0.5)))
