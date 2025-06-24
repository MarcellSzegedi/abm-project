"""Contains the agent class for the ABM's base version."""

import random
from typing import TYPE_CHECKING

import numpy as np
from mesa import Agent

from abm.utils.global_model_parameters import (
    INJURY_MINIMUM_AGENT_THD,
    MAX_AVAILABLE_AGENT_IN_CELL,
    MAX_INJURY_PROB,
    MOVEMENT_ARGUMENTS,
    RIOT_MINIMUM_AGENT_THD,
    ROW_FILTERING_CONDITIONS,
)

if TYPE_CHECKING:
    from abm.model import RiotModel


class FanAgent(Agent):
    """Class describing a fan in the abm model."""

    def __init__(
        self,
        pos: tuple[int, int],
        unique_id: int,
        model: "RiotModel",
        team: bool,
        state: str,
    ) -> None:
        """Initializes a FanAgent."""
        super().__init__(model=model, unique_id=unique_id)

        self.pos = pos
        self.unique_id = unique_id
        self.state = state
        self.team = team

    def step(self) -> None:
        """Executes events during an agent's step."""
        if not self.state == "injured":
            accessible_nbhood_cells = self._get_accessible_nbhood()
            self._set_agent_state(accessible_nbhood_cells)
            self._move_agent(accessible_nbhood_cells)

    def spread_agent(self) -> None:
        """Moves a given agent to a cell in the moore neighborhood, towards the exit of the map."""
        available_cells = self._find_available_downward_cells()
        if available_cells:
            self._execute_agent_movement(new_pos=random.choice(available_cells))

    def _set_agent_state(self, nbhood_cells: list[tuple[int, int]]) -> None:
        """Chooses and sets the new state of the agent."""
        cols, rows = zip(*nbhood_cells)
        match self.state:
            case "bystander":
                self._set_bystander_state(rows, cols)
            case "rioter":
                self._set_rioter_state(rows, cols)
            case _:
                raise ValueError(f"Unknown state: {self.state}")

    def _move_agent(self, accessible_nbhood_cells: list[tuple[int, int]]) -> None:
        """Moves the agent to a neighboring cell based on the agents in the vicinity."""
        available_cells_to_move = self._find_cell_to_move(accessible_nbhood_cells)
        if available_cells_to_move:
            all_cols, all_rows = zip(*accessible_nbhood_cells)
            available_cols, available_rows = zip(*available_cells_to_move)
            match self.state:
                case "bystander":
                    self._move_bystander(available_rows, available_cols)
                case "rioter":
                    self._move_rioter(all_rows, all_cols, available_rows, available_cols)
                case "injured":
                    pass
                case _:
                    raise ValueError(f"Unknown state: {self.state}")

    def _set_bystander_state(self, rows: tuple[int], cols: tuple[int]) -> None:
        """Sets the new state, given the current state of the agent is bystander.

        First check whether agent got injured and if not, change its state.
        The agent becomes a rioter and joins the riot if there are more rioters of the same team
            in the Moore nbhood than there are rioters of the opposite team.
        """
        if not self._check_injury():
            own_team_rioters = getattr(self.model, f"{'home' if self.team else 'away'}_riot_map")[
                rows, cols
            ]
            opp_team_rioters = getattr(
                self.model, f"{'home' if not self.team else 'away'}_riot_map"
            )[rows, cols]
            if (np.sum(own_team_rioters) > np.sum(opp_team_rioters)) and np.sum(
                own_team_rioters
            ) > RIOT_MINIMUM_AGENT_THD:
                self.state = "rioter"
                self.model.add_agent_to_utility_maps(agent=self)

    def _set_rioter_state(self, rows: tuple[int], cols: tuple[int]) -> None:
        """Sets the new state, given the current state of the agent is rioter.

        The agent becomes a bystander if there are no other rioters around, i.e. the current agent
            is the only rioter in the Moore nbhood <=> there is only one rioting agent in the Moore
            nbhood.
        """
        if not self._check_injury():
            if (
                np.sum(self.model.home_riot_map[rows, cols])
                + np.sum(self.model.away_riot_map[rows, cols])
                == 1
            ):
                self.model.remove_agent_from_utility_maps(agent=self)
                self.state = "bystander"

    def _check_injury(self) -> bool:
        """Checks if the agent is going ot be injured."""
        if self._check_injury_potential():
            prob_to_be_injured = self._injury_prob_calc()
            if random.random() < prob_to_be_injured:
                self.model.remove_agent_from_utility_maps(agent=self)
                self.state = "injured"
                return True
        return False

    def _check_injury_potential(self) -> bool:
        """Checks whether the agent's current position allows it to be injured.

        The conditions need to be held are the following:
        - The number of agents (including the agent in consideration) be larger or equal than the
            THD.
        """
        return len(self.model.grid.get_cell_list_contents([self.pos])) > INJURY_MINIMUM_AGENT_THD

    def _injury_prob_calc(self) -> float:
        """Calculated the probability of an agent being injured.

        The probability increases with the number of rioters in the cell, and is capped at a
            maximum value.
        """
        return (
            (
                np.sum(self.model.home_riot_map[self.pos[1], self.pos[0]])
                + np.sum(self.model.away_riot_map[self.pos[1], self.pos[0]])
            )
            / MAX_AVAILABLE_AGENT_IN_CELL
            * MAX_INJURY_PROB
        )

    def _move_bystander(self, rows: tuple[int], cols: tuple[int]) -> None:
        """Moves the agent to a Moore neighbourhood, given its state is bystander."""
        if self.pos[1] != self.model.city_map.height - 1:
            for mov_arg in MOVEMENT_ARGUMENTS:
                row_coords, col_coords = self._check_row_for_rioters(
                    rows,
                    cols,
                    row_to_check=mov_arg["row_to_check"],
                    only_opp_rioter=mov_arg["only_opp_rioter"],
                )
                if len(row_coords) > 0:
                    chosen_idx = np.random.choice(range(len(row_coords)))
                    self._execute_agent_movement(
                        new_pos=(col_coords[chosen_idx], row_coords[chosen_idx])
                    )
                    break
        else:
            self._agent_leaves()

    def _check_row_for_rioters(
        self, rows: tuple[int], cols: tuple[int], row_to_check: str, only_opp_rioter: bool
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Checks if the selected row has any of the given rioters."""
        condition = ROW_FILTERING_CONDITIONS[row_to_check]
        riot_map = (
            self.model.home_riot_map + self.model.away_riot_map
            if not only_opp_rioter
            else getattr(self.model, f"{'home' if self.team else 'away'}_riot_map")
        )

        matches = [
            (row, cols[i])
            for i, row in enumerate(rows)
            if condition(row, self.pos[1]) and riot_map[row, cols[i]] == 0
        ]
        row_coords, col_coords = zip(*matches) if matches else ((), ())
        return row_coords, col_coords

    def _move_rioter(
        self,
        all_rows: tuple[int],
        all_cols: tuple[int],
        available_rows: tuple[int],
        available_cols: tuple[int],
    ) -> None:
        """Moves the agent to a Moore neighbourhood, given its state is rioter.

        First check if there are more rioters of the same team than the opposite team in the
            neighbourhood. If not, move towards your own team to be safe, if yes, move towards
            the opposite team to start a fight.
        """
        num_own_team_rioters = np.sum(
            getattr(self.model, f"{'home' if self.team else 'away'}_riot_map")[all_rows, all_cols]
        )
        num_opposite_team_rioters = np.sum(
            getattr(self.model, f"{'home' if not self.team else 'away'}_riot_map")[
                all_rows, all_cols
            ]
        )

        if num_own_team_rioters < num_opposite_team_rioters:
            own_team_rioters = getattr(self.model, f"{'home' if self.team else 'away'}_riot_map")[
                available_rows, available_cols
            ]
            chosen_coord_idx = random.choice(
                np.where(own_team_rioters == np.max(own_team_rioters))[0]
            )
        else:
            # If there is at least one opposite team rioter, move towards the one with the least
            # number of rioters in the cell. If there are no opposite team rioters, move randomly
            opp_team_rioters = getattr(
                self.model, f"{'home' if not self.team else 'away'}_riot_map"
            )[available_rows, available_cols]

            nonzero_opposite_rioters = opp_team_rioters[np.nonzero(opp_team_rioters)]
            if len(nonzero_opposite_rioters) > 0:
                chosen_coord_idx = random.choice(
                    np.where(opp_team_rioters == np.min(nonzero_opposite_rioters))[0]
                )
            else:
                chosen_coord_idx = random.choice(np.where(opp_team_rioters == 0)[0])

        chosen_row, chosen_col = available_rows[chosen_coord_idx], available_cols[chosen_coord_idx]
        self._execute_agent_movement(new_pos=(chosen_col, chosen_row))

    def _execute_agent_movement(self, new_pos: tuple[int, int]) -> None:
        """Executes the movement of an agent."""
        self.model.remove_agent_from_utility_maps(agent=self)
        self.model.grid.move_agent(agent=self, pos=new_pos)
        self.model.add_agent_to_utility_maps(agent=self)

    def _agent_leaves(self) -> None:
        """Removes the agent from the model."""
        if self.team:
            self.model.left_home_fan_counter += 1
        else:
            self.model.left_away_fan_counter += 1

        self.model.remove_agent(self)

    def _find_cell_to_move(
        self, potential_cells_to_move: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Finds possible cells an agent can move to based on other agent's positions."""
        available_cells_to_move = [
            cell
            for cell in potential_cells_to_move
            if len(self.model.grid.get_cell_list_contents(cell)) < MAX_AVAILABLE_AGENT_IN_CELL
        ]
        return available_cells_to_move

    def _get_accessible_nbhood(self):
        """Returns the list of cells in the moore neighbourhood of the agent.

        This includes the cell the agent is currently in, and only considers cells that are
            accessible in the city map, i.e. cells that are not blocked by a building.
        """
        all_neighbouring_cell = self.model.grid.get_neighborhood(
            pos=self.pos, moore=True, include_center=True
        )
        return [cell for cell in all_neighbouring_cell if self.model.city_map.grid[cell[::-1]]]

    def _find_available_downward_cells(self) -> list[tuple[int, int]]:
        """Finds all the moore neighbor cells, with higher row coordinate."""
        return [
            cell
            for cell in self._find_cell_to_move(self._get_accessible_nbhood())
            if cell[1] > self.pos[1]
        ]
