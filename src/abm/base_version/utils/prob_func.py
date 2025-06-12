"""Utility functions to calculate the probabilities of the agents."""

import numpy as np
import numpy.typing as npt

from abm.base_version.utils.global_model_parameters import (
    MAX_AVAILABLE_AGENT_IN_CELL,
    DOWNWARD_EXTRA_WEIGHT
)


def state_change_probability(
    n_of_in_state_agents: int, n_of_available_cell: int, k: float = 5
) -> float:
    """Calculates the relative probability of a state change."""
    return n_of_in_state_agents / (n_of_available_cell * MAX_AVAILABLE_AGENT_IN_CELL)


def cell_movement_probability(
        pos: tuple[int, int],
        available_cells_to_move: list[tuple[int, int]],
        n_of_in_team_agents: npt.NDArray[int],
        n_of_available_cell: int
) -> npt.NDArray[float]:
    """Calculates the relative probability of an agent moving to a cell."""
    rel_probs = n_of_in_team_agents / (n_of_available_cell * MAX_AVAILABLE_AGENT_IN_CELL)
    downward_cell_mask = np.array([cell_cord[1] > pos[1] for cell_cord in available_cells_to_move])
    rel_probs[downward_cell_mask] *= DOWNWARD_EXTRA_WEIGHT
    return rel_probs

