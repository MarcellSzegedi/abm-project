"""Contains the agent class for the ABM's base version."""

import random
from typing import TYPE_CHECKING

import numpy as np
from mesa import Agent

from abm.base_version.utils.global_model_parameters import (
    MAX_AVAILABLE_AGENT_IN_CELL,
    MIN_PROB_OF_TRANS_TO_BASE,
    MIN_PROB_OF_TRANS_TO_INJURED,
    MIN_PROB_OF_TRANS_TO_RIOT,
)
from abm.base_version.utils.prob_func import cell_movement_probability, state_change_probability

if TYPE_CHECKING:
    from abm.base_version.model import RiotModel


class FanAgent(Agent):
    """Class describing a fan in the abm model."""

    def __init__(
        self,
        pos: tuple[int, int],
        unique_id: int,
        model: "RiotModel",
        team: bool,
        state: str = "base",
    ) -> None:
        """Initializes a FanAgent."""
        super().__init__(model=model, unique_id=unique_id)

        self.pos = pos
        self.unique_id = unique_id
        self.state = state
        self.team = team

    def spread_agent(self) -> None:
        """Moves a given agent to a cell in the moore neighborhood, towards the exit of the map."""
        available_cells = self._find_available_downward_cells()
        if available_cells:
            self._remove_agent_state_from_map()
            self._remove_agent_team_from_map()

            self.model.grid.move_agent(agent=self, pos=random.choice(available_cells))

            self._add_agent_state_to_map()
            self._add_agent_team_to_map()

    def step_agent(self) -> None:
        """Executes events during an agent's step."""
        if not self.state == "injured":
            available_cells_to_move = self._find_possible_n_available_cells()
            self._set_agent_state(available_cells_to_move)
            self._move_agent(available_cells_to_move)

    def _move_agent(self, potential_cell_to_move: list[tuple[int, int]]) -> None:
        """Moves the agent to a neighboring cell based on the agents in the vicinity."""
        if self.pos[0] != self.model.height:
            available_cells_to_move = self._find_cell_to_move(potential_cell_to_move)
            if available_cells_to_move:
                match self.state:
                    case "base":
                        rows, cols = zip(*available_cells_to_move)
                        team_map = getattr(
                            self.model, f"{'home' if self.team else 'away'}_team_map"
                        )
                        n_same_team_agents = team_map[rows, cols]
                        rel_probs_of_available_cells = cell_movement_probability(
                            n_same_team_agents, len(available_cells_to_move)
                        )

                        self._remove_agent_state_from_map()
                        self._remove_agent_team_from_map()

                        self.model.grid.move_agent(
                            agent=self,
                            pos=np.random.choice(
                                available_cells_to_move,
                                p=rel_probs_of_available_cells
                                / np.sum(rel_probs_of_available_cells),
                            ),
                        )

                        self._add_agent_state_to_map()
                        self._add_agent_team_to_map()
                    case "riot":
                        rows, cols = zip(*available_cells_to_move)
                        riot_map = self.model.riot_state_map
                        n_riot_agents = riot_map[rows, cols]
                        rel_probs_of_available_cells = cell_movement_probability(
                            n_riot_agents, len(available_cells_to_move)
                        )

                        self._remove_agent_state_from_map()
                        self._remove_agent_team_from_map()

                        self.pos = np.random.choice(
                            available_cells_to_move,
                            p=rel_probs_of_available_cells / np.sum(rel_probs_of_available_cells),
                        )

                        self._add_agent_state_to_map()
                        self._add_agent_team_to_map()
                    case _:
                        raise ValueError(f"Unknown state: {self.state}")
        else:
            self.model.remove_agent(self)
            self._remove_agent_state_from_map()
            self._remove_agent_team_from_map()

    def _set_agent_state(self, available_cells: list[tuple[int, int]]) -> None:
        """Chooses and sets the new state of the agent."""
        n_base_agent = self._find_agent_states_in_vicinity("base", available_cells)
        n_riot_agent = self._find_agent_states_in_vicinity("riot", available_cells)
        n_riot_injured_agent_center_pos = self._find_agent_states_in_vicinity(
            "riot", [self.pos]
        ) + self._find_agent_states_in_vicinity("injured", [self.pos])

        rel_prob_base = max(
            MIN_PROB_OF_TRANS_TO_BASE, state_change_probability(n_base_agent, len(available_cells))
        )
        rel_prob_riot = max(
            MIN_PROB_OF_TRANS_TO_RIOT, state_change_probability(n_riot_agent, len(available_cells))
        )
        rel_prob_inured = max(
            MIN_PROB_OF_TRANS_TO_INJURED,
            state_change_probability(n_riot_injured_agent_center_pos, 1) / 10,
        )

        relative_probs = np.array([rel_prob_base, rel_prob_riot, rel_prob_inured])

        self.state = np.random.choice(
            ["base", "riot", "injured"], p=relative_probs / relative_probs.sum()
        )

    def _remove_agent_state_from_map(self) -> None:
        """Removes the agent from the corresponding state map."""
        state_map = getattr(self.model, f"{self.state}_state_map")
        state_map[self.pos] -= 1

    def _add_agent_state_to_map(self) -> None:
        """Adds the agent to the corresponding state map."""
        state_map = getattr(self.model, f"{self.state}_state_map")
        state_map[self.pos] += 1

    def _remove_agent_team_from_map(self) -> None:
        """Removes the agent from the corresponding team map."""
        team_map = getattr(self.model, f"{'home' if self.team else 'away'}_team_map")
        team_map[self.pos] -= 1

    def _add_agent_team_to_map(self) -> None:
        """Adds the agent to the corresponding team map."""
        team_map = getattr(self.model, f"{'home' if self.team else 'away'}_team_map")
        team_map[self.pos] += 1

    def _find_cell_to_move(
        self, potential_cells_to_move: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Finds possible cells an agent can move to."""
        available_cells_to_move = [
            cell
            for cell in potential_cells_to_move
            if len(self.model.grid.get_cell_list_contents(cell)) < MAX_AVAILABLE_AGENT_IN_CELL
        ]
        return available_cells_to_move

    def _find_possible_n_available_cells(self):
        all_neighbouring_cell = self.model.grid.get_neighborhood(
            pos=self.pos, moore=True, include_center=True
        )
        return [cell for cell in all_neighbouring_cell if self.model.map[cell]]

    def _find_agent_states_in_vicinity(
        self, state_to_find: str, available_cells: list[tuple[int, int]]
    ) -> int:
        """Aggregate the number of agents in a given state in the vicinity."""
        state_map = getattr(self.model, f"{state_to_find}_state_map")
        rows, cols = zip(*available_cells)
        return np.sum(state_map[rows, cols])

    def _find_available_downward_cells(self) -> list[tuple[int, int]]:
        """Finds all the moore neighbor cells, with higher row coordinate."""
        return [
            cell
            for cell in self._find_cell_to_move(self._find_possible_n_available_cells())
            if cell[0] > self.pos[0]
        ]
