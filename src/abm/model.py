"""ABM model."""

import logging

import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from tqdm import trange

from abm.agent import FanAgent
from abm.city_map import CityMap
from abm.utils.global_model_parameters import (
    INITIAL_PROB_OF_BASE,
    INITIAL_PROB_OF_RIOT,
    MAX_AVAILABLE_AGENT_IN_CELL,
)
from abm.utils.logging_config import setup_logging
from abm.utils.util_func import count_agents_in_state, count_agents_in_team
from abm.visualisation.animation import CellInfoContainer, animate_model, get_grid_data

setup_logging()
logger = logging.getLogger(__name__)


class RiotModel(Model):
    """Contains the implementation of the Riot model."""

    def __init__(
        self,
        width: int,
        height: int,
        n_home_fans: int,
        n_away_fans: int,
        entry_points_home: list[tuple[int, int]],
        entry_points_away: list[tuple[int, int]],
        city_map: CityMap,
        animate: bool = False,
    ) -> None:
        """Initializes the Riot model."""
        super().__init__()

        self.scheduler = RandomActivation(self)

        self.grid = MultiGrid(width=width, height=height, torus=False) 
        self.city_map = city_map
        self.home_riot_map = np.zeros(shape=(height, width))
        self.away_riot_map = np.zeros(shape=(height, width))

        self.n_home_fans = n_home_fans
        self.n_away_fans = n_away_fans
        self.entry_points_home = entry_points_home  # (col, row) format
        self.entry_points_away = entry_points_away  # (col, row) format

        self.agent_state_datacollector = DataCollector(
            {
                "Bystander": lambda m: count_agents_in_state(m, "bystander"),
                "Rioter": lambda m: count_agents_in_state(m, "rioter"),
                "Injured": lambda m: count_agents_in_state(m, "injured"),
            }
        )
        self.control_team_fan_counter = DataCollector(
            {
                "Home": lambda m: count_agents_in_team(m, True),
                "Away": lambda m: count_agents_in_team(m, False),
            }
        )

        self.entered_home_fan_counter = 0
        self.left_home_fan_counter = 0
        self.entered_away_fan_counter = 0
        self.left_away_fan_counter = 0

        self.animation_frames: list[list[CellInfoContainer]] | None = [] if animate else None

    @classmethod
    def run_riot_model(
        cls,
        width: int,
        height: int,
        n_home_fans: int,
        n_away_fans: int,
        entry_points_home: list[tuple[int, int]],
        entry_points_away: list[tuple[int, int]],
        n_step: int,
        city_map: CityMap,
        animate: bool = False,
        detailed_logging: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[list[CellInfoContainer]] | None]:
        """Runs the abm model."""
        riot_model = cls(
            width,
            height,
            n_home_fans,
            n_away_fans,
            entry_points_home,
            entry_points_away,
            city_map,
            animate,
        )

        # Collect initial data
        riot_model.agent_state_datacollector.collect(riot_model)
        riot_model.control_team_fan_counter.collect(riot_model)

        for _ in trange(n_step):
            riot_model._spawn_agents(detailed_logging)
            riot_model.agent_state_datacollector.collect(riot_model)
            riot_model.control_team_fan_counter.collect(riot_model)
            if riot_model.animation_frames is not None:
                riot_model.animation_frames.append(get_grid_data(riot_model))

            riot_model.step()

        agent_state_data: pd.DataFrame = (
            riot_model.agent_state_datacollector.get_model_vars_dataframe()
        )
        control_team_data: pd.DataFrame = (
            riot_model.control_team_fan_counter.get_model_vars_dataframe()
        )

        return agent_state_data, control_team_data, riot_model.animation_frames

    def step(self) -> None:
        """Executes events in one step of the model."""
        self.scheduler.step()
        self.agent_state_datacollector.collect(self)
        self.control_team_fan_counter.collect(self)
        if self.animation_frames is not None:
            self.animation_frames.append(get_grid_data(self))

    def add_agent(self, pos: tuple[int, int], team: bool, state: str) -> None:
        """Adds a new agent to the model."""
        if len(self.grid.get_cell_list_contents([pos])) >= MAX_AVAILABLE_AGENT_IN_CELL:
            raise ValueError(
                f"Agent cannot be placed at {pos}, as there are too many other agents already."
            )

        agent = FanAgent(pos=pos, unique_id=self.next_id(), team=team, state=state, model=self)
        self.grid.place_agent(agent, pos)
        self.scheduler.add(agent)
        self.add_agent_to_utility_maps(agent)

    def remove_agent(self, agent: FanAgent) -> None:
        """Removes the agent from the model."""
        self.remove_agent_from_utility_maps(agent)
        self.grid.remove_agent(agent)
        self.scheduler.remove(agent)

    def remove_agent_from_utility_maps(self, agent: FanAgent) -> None:
        """Removes the agent from the corresponding utility maps."""
        if agent.state == "rioter":
            riot_map = getattr(self, f"{'home' if agent.team else 'away'}_riot_map")
            riot_map[agent.pos[::-1]] -= 1

    def add_agent_to_utility_maps(self, agent: FanAgent) -> None:
        """Adds the agent to the corresponding utility maps."""
        if agent.state == "rioter":
            riot_map = getattr(self, f"{'home' if agent.team else 'away'}_riot_map")
            riot_map[agent.pos[::-1]] += 1

    def _spawn_agents(self, log_switch: bool) -> None:
        """Spawns Agents into the Grid."""
        if self.entered_home_fan_counter < self.n_home_fans:
            self._add_fans_batch(team=True, log_switch=log_switch)
        if self.entered_away_fan_counter < self.n_away_fans:
            self._add_fans_batch(team=False, log_switch=log_switch)

        if log_switch:
            logger.info(f"{self.entered_home_fan_counter} Home Fans are present in the Grid")
            logger.info(f"{self.entered_away_fan_counter} Away Fans are present in the Grid")

            num_injured = sum(agent.state == "injured" for agent in self.scheduler.agents)
            logger.info(f"Number of injured agents: {num_injured}")

            injured_home = self._count_injured_at_entry_points(True)
            injured_away = self._count_injured_at_entry_points(False)
            logger.info(f"{injured_home} injured home agents at entry points")
            logger.info(f"{injured_away} injured away agents at entry points")

    def _add_fans_batch(self, team: bool, log_switch: bool) -> None:
        """Adds a batch of fans (5 for us) for a given team.

        Ensures only one agent is added in an entry point of the grid per batch.
        """
        fans_added = 0
        entry_points = self.entry_points_home if team else self.entry_points_away
        for entry_point in entry_points:
            if len(self.grid.get_cell_list_contents([entry_point])) < MAX_AVAILABLE_AGENT_IN_CELL:
                self.add_agent(
                    pos=entry_point,
                    team=team,
                    state=np.random.choice(
                        np.array(["bystander", "rioter"]),
                        p=np.array([INITIAL_PROB_OF_BASE, INITIAL_PROB_OF_RIOT]),
                    ),
                )
                fans_added += 1
                if team:
                    self.entered_home_fan_counter += 1
                else:
                    self.entered_away_fan_counter += 1

        if log_switch:
            logger.info(f"{fans_added} {'home' if team else 'away'} fans were added in this Batch")

    def _count_injured_at_entry_points(self, team: bool) -> int:
        """Counts the number of injured agents at the entry points for a given team."""
        total_injured = 0
        entry_points = self.entry_points_home if team else self.entry_points_away
        for entry_point in entry_points:
            agents_in_cell = self.grid.get_cell_list_contents([entry_point])
            injured_agents = [agent for agent in agents_in_cell if agent.state == "injured"]
            total_injured += len(injured_agents)
        return total_injured


if __name__ == "__main__":
    width = 50
    height = 100
    n_home_fans = 9000
    n_away_fans = 1000
    n_streets = 4
    street_width = 7
    exit_space_height = 10
    HOME_EXIT_1_RANGE = range(14, 19)
    HOME_EXIT_2_RANGE = range(22, 27)
    AWAY_EXIT_RANGE = range(30, 35)
    entry_points_home = [(i, 0) for i in HOME_EXIT_1_RANGE] + [
        (i, 0) for i in HOME_EXIT_2_RANGE
    ]  # 2 exits for home fans
    entry_points_away = [(i, 0) for i in AWAY_EXIT_RANGE]  # 1 exit for away fans
    n_step = 20  # TODO: Needs to be increased accordingly

    logger.info("Starting Riot Simulation")
    logger.info(f"Map size: {width}x{height}, with {n_streets} streets")
    logger.info(f"Street width: {street_width}, exit height: {exit_space_height}")
    logger.info(f"Entry points (home): {entry_points_home}")
    logger.info(f"Entry points (away): {entry_points_away}")
    logger.info(f"Simulation steps: {n_step}")

    city_map = CityMap(width, height, n_streets, street_width, exit_space_height)
    _, _, frames = RiotModel.run_riot_model(
        width,
        height,
        n_home_fans,
        n_away_fans,
        entry_points_home,
        entry_points_away,
        n_step,
        city_map,
        animate=True,
    )
    logger.info("Riot Simulation Completed. PLotting Results")
    animate_model(frames, city_map.grid, height, width)
