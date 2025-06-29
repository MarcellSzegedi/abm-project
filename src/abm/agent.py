"""Contains the agent class for the ABM's base version."""

from typing import TYPE_CHECKING

import numpy as np
from mesa import Agent

from abm.utils.global_model_parameters import (
    INJURY_MINIMUM_AGENT_THD,
    MAX_AVAILABLE_AGENT_IN_CELL,
    MOVEMENT_ARGUMENTS,
    RIOT_MINIMUM_AGENT_THD,
    ROW_FILTERING_CONDITIONS,
    STEP_THD,
)

if TYPE_CHECKING:
    from abm.model import RiotModel


class FanAgent(Agent):
    """Class representing a fan in the ABM."""

    def __init__(
        self,
        pos: tuple[int, int],
        unique_id: int,
        model: "RiotModel",
        team: bool,
        state: str,
    ) -> None:
        """Initializes a FanAgent.

        Args:
            pos: Position of the agent in (col, row) format.
            unique_id: Unique ID of the agent.
            model: ABM model.
            team: Team of the agent.
            state: State of the agent.
        """
        super().__init__(model=model, unique_id=unique_id)

        self.pos = pos
        self.unique_id = unique_id
        self.state = state
        self.team = team
        self.willingness_to_riot = self.model.rng.random()
        self.step_counter = 0

    def step(self) -> None:
        """Executes events during an agent's step.

        First, it checks whether the agent's state is 'injured'.
        If not, it performs a state reconsideration and then potentially moves the agent
        based on its new state.
        """
        if not self.state == "injured":
            accessible_nbhood_cells = self._get_accessible_nbhood()
            self._set_agent_state(accessible_nbhood_cells)
            self._move_agent(accessible_nbhood_cells)
            self.step_counter += 1

    def _set_agent_state(self, nbhood_cells: list[tuple[int, int]]) -> None:
        """Sets the new state of the agent, based on the current one.

        Args:
            nbhood_cells: Coordinates of the cells in the Moore neighbourhood of the agent
                            that are accessible based on the city_map.
        """
        cols, rows = zip(*nbhood_cells)
        match self.state:
            case "bystander":
                self._set_bystander_state(rows, cols)
            case "rioter":
                self._set_rioter_state(rows, cols)
            case _:
                raise ValueError(f"Unknown state: {self.state}")

    def _move_agent(self, nbhood_cells: list[tuple[int, int]]) -> None:
        """Moves the agent to a neighboring cell based on its current state.

        Args:
            nbhood_cells: Coordinates of the cells in the Moore neighbourhood of the agent
                            that are accessible based on the city_map.
        """
        available_cells_to_move = self._find_cell_to_move(nbhood_cells)
        if available_cells_to_move:
            available_cols, available_rows = zip(*available_cells_to_move)
            match self.state:
                case "bystander":
                    self._move_bystander(available_rows, available_cols)
                case "rioter":
                    all_cols, all_rows = zip(*nbhood_cells)
                    self._move_rioter(all_rows, all_cols, available_rows, available_cols)
                case "injured":
                    pass
                case _:
                    raise ValueError(f"Unknown state: {self.state}")

    def _set_bystander_state(self, rows: tuple[int], cols: tuple[int]) -> None:
        """Sets a new state, given the current state of the agent is 'bystander'.

        First, checks whether the agent is injured. If not, the agent may change state.
        The agent becomes a rioter if all the following conditions are met:

        - There are more rioters of the same team in the Moore neighbourhood than there are
            rioters of the opposite team.
        - The number of rioters in the Moore neighbourhood is higher or equal than the
            corresponding threshold.
        - The number of opposite team's rioters in the Moore neighbourhood is positive.

        Args:
            rows: Tuple of row coordinates of the cells accessible in the Moore neighbourhood
                    of the agent based on the city_map.
            cols: Tuple of column coordinates of the cells accessible in the Moore neighbourhood
                    of the agent based on the city_map.
        """
        if not self._check_injury():
            n_own_rioters = np.sum(
                self.model.home_riot_map[rows, cols]
                if self.team
                else self.model.away_riot_map[rows, cols]
            )

            n_opp_rioters = np.sum(
                self.model.away_riot_map[rows, cols]
                if self.team
                else self.model.home_riot_map[rows, cols]
            )

            if (
                True
                and n_own_rioters >= n_opp_rioters
                and n_own_rioters + n_opp_rioters >= RIOT_MINIMUM_AGENT_THD
                and n_opp_rioters > 0
                and self.willingness_to_riot > self.model.riot_willingness_thd
            ):
                self.state = "rioter"
                self.model.add_agent_to_utility_maps(agent=self)

    def _set_rioter_state(self, rows: tuple[int], cols: tuple[int]) -> None:
        """Sets the new state, given the current state of the agent is 'rioter'.

        The agent becomes a bystander if there are no other rioters around, i.e. the current agent
        is the only rioter in the Moore neighbourhood <=> there is only one rioting agent in the
        Moore neighbourhood.

        Args:
            rows: Tuple of row coordinates of the cells accessible in the Moore neighbourhood
                    of the agent based on the city_map.
            cols: Tuple of column coordinates of the cells accessible in the Moore neighbourhood
                    of the agent based on the city_map.
        """
        if not self._check_injury():
            total_rioters = np.sum(self.model.home_riot_map[rows, cols]) + np.sum(
                self.model.away_riot_map[rows, cols]
            )

            if self.step_counter >= STEP_THD and total_rioters == 1:
                self.model.remove_agent_from_utility_maps(agent=self)
                self.state = "bystander"

    def _check_injury(self) -> bool:
        """Checks whether the agent gets injured and updates its state accordingly.

        If the agent can be injured based on the other agents on the same location, a probability
        injury is calculated and used to determine the outcome.

        Returns:
            True if the agent got injured, False otherwise.
        """
        if self._check_injury_potential():
            prob_to_be_injured = self._injury_prob_calc()
            if self.model.rng.random() < prob_to_be_injured:
                self.model.remove_agent_from_utility_maps(agent=self)
                self.state = "injured"
                return True
        return False

    def _check_injury_potential(self) -> bool:
        """Checks whether the agent's current position allows it to be injured.

        An agent can be injured if the number of agents at the same location
        (including the agent itself) exceeds the corresponding threshold.

        Returns:
            True if the agent's current position allows it to be injured, False otherwise.
        """
        return len(self.model.grid.get_cell_list_contents([self.pos])) > INJURY_MINIMUM_AGENT_THD

    def _injury_prob_calc(self) -> float:
        """Calculates the probability of the agent being injured.

        The injury probability:
        - Linearly increases with the number of rioters in the cell (starting from 0
            if there are none).
        - Doubles if there are rioters from the opposite team.
        - Is capped at a maximum value.

        Returns:
            Probability of being injured, given that the agent current position allows it.
        """
        col, row = self.pos
        n_home_rioters = np.sum(self.model.home_riot_map[row, col])
        n_away_rioters = np.sum(self.model.away_riot_map[row, col])

        injury_prob = (
            (n_home_rioters + n_away_rioters)
            / MAX_AVAILABLE_AGENT_IN_CELL
            * self.model.p_injury_ub
        )

        if not (n_home_rioters and n_away_rioters):
            injury_prob *= 0.5

        return injury_prob

    def _move_bystander(self, rows: tuple[int], cols: tuple[int]) -> None:
        """Moves the agent to a Moore neighbourhood cell, given its state is 'bystander'.

        Checks whether the agent is in the last row of the grid; if so, it deterministically
        leaves the grid. Otherwise, it selects a new position randomly from the accessible cells
        in the Moore neighbourhood that contain no rioters. The selection prioritizes cells in
        the following order:
        1. Bottom row (higher row index),
        2. Middle row (including the agent's current position),
        3. Top row (lower row index).

        If all candidate cells contain at least one rioter, the same logic is repeated with the
        change that avoidance is restricted to the opposite team's rioters only.

        Args:
            rows: Tuple of the rows coordinates of the accessible (based on city_map and the agent
                    count) cells in the Moore neighbourhood based on the city_map.
            cols: Tuple of the column coordinates of the accessible (based on city_map and the
                    agent count) cells in the Moore neighbourhood based on the city_map.
        """
        if self.pos[1] == self.model.city_map.height - 1:
            self._agent_leaves()
            return

        for mov_arg in MOVEMENT_ARGUMENTS:
            row_coords, col_coords = self._check_row_for_rioters(
                rows,
                cols,
                row_to_check=mov_arg["row_to_check"],
                only_opp_rioter=mov_arg["only_opp_rioter"],
            )

            if len(row_coords) > 0:
                chosen_idx = self.model.rng.choice(range(len(row_coords)))
                self._execute_agent_movement(
                    new_pos=(col_coords[chosen_idx], row_coords[chosen_idx])
                )
                return

    def _check_row_for_rioters(
        self, rows: tuple[int], cols: tuple[int], row_to_check: str, only_opp_rioter: bool
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Checks if the selected row has any of the given rioters.

        Args:
            rows: Tuple of row coordinates of cells considered for the agent’s movement.
            cols: Tuple of column coordinates of cells considered for the agent’s movement.
            row_to_check: One of 'bot', 'mid', or 'top', specifies which row to check for rioters.
            only_opp_rioter: If True, only the opposite team's rioters will be checked.

        Returns:
            row_coords: Tuple of row coordinates of the cells which have zero rioters.
            col_coords: Tuple of column coordinates of the cells which have zero rioters.
        """
        condition = ROW_FILTERING_CONDITIONS[row_to_check]
        riot_map = (
            self.model.home_riot_map + self.model.away_riot_map
            if not only_opp_rioter
            else self.model.home_riot_map
            if self.team
            else self.model.away_riot_map
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
        """Moves the agent to a Moore neighbourhood, given its state is 'rioter'.

        First check if the number of steps done since the spawn of the agent is above or equal to
        the corresponding threshold. If so, then move downward (toward higher row coordinates)
        if possible, of randomly if not. If the step count is high enough, it checks if the number
        of rioters from the agent's own team is more than of the opposite team's.

        If yes:
        Move towards the opposite team. (fight) Randomly choose a cell form the available cells in
        the neighbourhood with the least number of opposite team's rioter.

        Otherwise:
        Move towards your own team (herding). Randomly choose a cell from the available cells in
        the neighbourhood with the highest number of own team's rioter. If there are no rioters
        from the opposite team, the agent moves randomly.

        Args:
            all_rows:Tuple of row coordinates of the cells accessible in the Moore neighbourhood
                                of the agent based on the city_map.
            all_cols: Tuple of column coordinates of the cells accessible in the Moore
                                neighbourhood of the agent based on the city_map.
            available_rows: Tuple of row coordinates of cells considered for the agent’s movement.
            available_cols: Tuple of column coordinates of cells considered for the agent’s
                                movement.
        """
        if self.step_counter < STEP_THD:
            if self._find_available_downward_cells():
                return self._execute_agent_movement(
                    new_pos=self.model.rng.choice(self._find_available_downward_cells())
                )

            choice = self.model.rng.choice(list(zip(available_cols, available_rows)))

            return self._execute_agent_movement(new_pos=choice)

        n_own_rioters = np.sum(
            self.model.home_riot_map[all_rows, all_cols]
            if self.team
            else self.model.away_riot_map[all_rows, all_cols]
        )
        n_opp_rioters = np.sum(
            self.model.away_riot_map[all_rows, all_cols]
            if self.team
            else self.model.home_riot_map[all_rows, all_cols]
        )

        if n_own_rioters < n_opp_rioters:
            own_rioters = (
                self.model.home_riot_map[available_rows, available_cols]
                if self.team
                else self.model.away_riot_map[available_rows, available_cols]
            )
            chosen_coord_idx = self.model.rng.choice(
                np.where(own_rioters == np.max(own_rioters))[0]
            )
        else:
            opp_rioters = (
                self.model.away_riot_map[available_rows, available_cols]
                if self.team
                else self.model.home_riot_map[available_rows, available_cols]
            )

            if len(nonzero_opp_rioters := opp_rioters[np.nonzero(opp_rioters)]) > 0:
                chosen_coord_idx = self.model.rng.choice(
                    np.where(opp_rioters == np.min(nonzero_opp_rioters))[0]
                )
            else:
                chosen_coord_idx = self.model.rng.choice(np.where(opp_rioters == 0)[0])

        chosen_pos = (available_cols[chosen_coord_idx], available_rows[chosen_coord_idx])
        return self._execute_agent_movement(new_pos=chosen_pos)

    def _execute_agent_movement(self, new_pos: tuple[int, int]) -> None:
        """Executes the movement of an agent.

        Args:
            new_pos: New position of the agent in (col, row) format.
        """
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

    def _get_accessible_nbhood(self) -> list[tuple[int, int]]:
        """Gets accessible Moore neighbourhood cells based on the city map.

        Collects the coordinates of the cells that are in the Moore neighbourhood and are
        accessible based on the city_map, i.e. cells that are not blocked by a building.

        Returns:
            List of coordinates in (col, row) format of the accessible cells.
        """
        all_neighbouring_cell = self.model.grid.get_neighborhood(
            pos=self.pos, moore=True, include_center=True
        )
        return [cell for cell in all_neighbouring_cell if self.model.city_map.grid[cell[::-1]]]

    def _find_cell_to_move(
        self, potential_cells_to_move: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Collects possible cells an agent can move to based on their crowdedness.

        The function collects the cells in the Moore neighbourhood that are:
        - Accessible based on the city_map
        - Has lower number of agents than the maximum crowdedness threshold.

        Args:
            potential_cells_to_move: List of coordinates in (col, row) format of the accessible
                                        cells based on the city map.

        Returns:
            List of tuples containing the coordinates (col, row) of the cells the agent can
            move to.
        """
        return [
            cell
            for cell in potential_cells_to_move
            if len(self.model.grid.get_cell_list_contents(cell)) < MAX_AVAILABLE_AGENT_IN_CELL
        ]

    def _find_available_downward_cells(self) -> list[tuple[int, int]]:
        """Collects all the bottom cells in the available Moore neighbourhood.

        The function collects the coordinates in (col, row) format of the cells that:
        - Are accessible in the Moore neighbourhood (based on the city map).
        - Have a row coordinate higher than the agent's current position.

        Returns:
            List of tuples containing the coordinates (col, row) of cells meeting both criteria.
        """
        return [
            cell
            for cell in self._find_cell_to_move(self._get_accessible_nbhood())
            if cell[1] > self.pos[1]
        ]
