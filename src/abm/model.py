"""ABM model."""

import logging

import matplotlib.pyplot as plt
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
from abm.utils.utility_func import count_agents_in_state
from abm.visualisation.animation import CellInfoContainer, get_grid_data

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
        p_injury_ub: float,
        riot_willingness_thd: float,
        city_map: CityMap,
        animate: bool = False,
    ) -> None:
        """Initializes the Riot model."""
        super().__init__()
        self.rng = np.random.default_rng()

        self.scheduler = RandomActivation(self)

        self.grid = MultiGrid(width=width, height=height, torus=False)
        self.city_map = city_map
        self.home_riot_map = np.zeros(shape=(height, width))
        self.away_riot_map = np.zeros(shape=(height, width))

        self.n_home_fans = n_home_fans
        self.n_away_fans = n_away_fans
        self.entry_points_home = entry_points_home
        self.entry_points_away = entry_points_away
        self.p_injury_ub = p_injury_ub
        self.riot_willingness_thd = riot_willingness_thd

        self.agent_state_datacollector = DataCollector(
            {
                "Bystander": lambda m: count_agents_in_state(m, "bystander"),
                "Rioter": lambda m: count_agents_in_state(m, "rioter"),
                "Injured": lambda m: count_agents_in_state(m, "injured"),
            }
        )

        self.entered_home_fan_counter = 0
        self.entered_away_fan_counter = 0
        self.left_home_fan_counter = 0
        self.left_away_fan_counter = 0

        self.animation_frames: list[list[CellInfoContainer]] | None = [] if animate else None

        self.running = True

    @classmethod
    def run_riot_model(
        cls,
        width: int,
        height: int,
        n_home_fans: int,
        n_away_fans: int,
        entry_points_home: list[tuple[int, int]],
        entry_points_away: list[tuple[int, int]],
        p_injury_ub: float,
        riot_willingness_thd: float,
        n_step: int,
        city_map: CityMap,
        animate: bool = False,
        detailed_logging: bool = True,
    ) -> tuple[pd.DataFrame, list[list[CellInfoContainer]] | None]:
        """Runs the ABM model.

        Args:
            width: Width of the grid measured as the number of cells.
            height: Height of the grid measured as the number of cells.
            n_home_fans: Total number of home fans to spawn on the grid throughout the simulation.
            n_away_fans: Total number of away fans to spawn on the grid throughout the simulation.
            entry_points_home: Coordinates of the spawn points of the home fan agents
                                in (col, row) format.
            entry_points_away: Coordinates of the spawn points of the away fan agents
                                in (col, row) format.
            p_injury_ub: Upper bound of the probability of injury for the agents.
            riot_willingness_thd: Threshold of the willingness to riot for the agents.
            n_step: Number of simulation steps.
            city_map: Boolean numpy array representing the city map,
                                with True values in the places where agents can move
                                and False where they cannot.
            animate: True if the simulation should be animated, False otherwise.
            detailed_logging: True if the simulation should be logged, False otherwise.

        Returns:
            agent_state_data: Dataframe with the distribution of the various states per step.
            riot_model.animation_frames: Animation frames if animate is True, otherwise None.
        """
        riot_model = cls(
            width,
            height,
            n_home_fans,
            n_away_fans,
            entry_points_home,
            entry_points_away,
            p_injury_ub,
            riot_willingness_thd,
            city_map,
            animate,
        )

        riot_model.agent_state_datacollector.collect(riot_model)
        for _ in trange(n_step):
            riot_model._spawn_agents(detailed_logging)
            if riot_model.animation_frames is not None:
                riot_model.animation_frames.append(get_grid_data(riot_model))

            riot_model.step()
            if not riot_model.running:
                break

        agent_state_data: pd.DataFrame = (
            riot_model.agent_state_datacollector.get_model_vars_dataframe()
        )

        return agent_state_data, riot_model.animation_frames

    def step(self) -> None:
        """Executes events in one step of the model and collects the required data."""
        self.scheduler.step()

        if count_agents_in_state(self, target_state="rioter") + count_agents_in_state(
            self, target_state="bystander"
        ) <= 0.01 * (self.n_home_fans + self.n_away_fans) and (
            self.entered_away_fan_counter + self.entered_home_fan_counter
            >= self.n_home_fans + self.n_away_fans
        ):
            self.running = False

        self.agent_state_datacollector.collect(self)
        if self.animation_frames is not None:
            self.animation_frames.append(get_grid_data(self))

    def add_agent(self, pos: tuple[int, int], team: bool, state: str) -> None:
        """Adds a new agent to the model.

        The function first checks if the position is available, according to the city map and the
        number of agents already located on the cell.
        Then it adds to agent to the
        - Grid
        - Scheduler
        - Utility maps (based on its attributes)

        Args:
            pos: Spawn position of the agent in (col, row) format.
            team: True if the agent is in the home team, False otherwise.
            state: The initial state of the agent.
        """
        if not self.city_map.grid[pos[::-1]]:
            raise ValueError(
                f"Agent cannot be placed at {pos}, as it is not accessible for the agents"
                f"due to the city map structure."
            )
        if len(self.grid.get_cell_list_contents([pos])) >= MAX_AVAILABLE_AGENT_IN_CELL:
            raise ValueError(
                f"Agent cannot be placed at {pos}, as there are too many other agents already."
            )

        agent = FanAgent(pos=pos, unique_id=self.next_id(), team=team, state=state, model=self)
        self.grid.place_agent(agent, pos)
        self.scheduler.add(agent)
        self.add_agent_to_utility_maps(agent)

    def remove_agent(self, agent: FanAgent) -> None:
        """Removes the agent from the model.

        The function removes the agent from
        - Grid
        - Scheduler
        - Utility maps

        Args:
            agent: Agent to remove from the model.
        """
        self.grid.remove_agent(agent)
        self.scheduler.remove(agent)
        self.remove_agent_from_utility_maps(agent)

    def add_agent_to_utility_maps(self, agent: FanAgent) -> None:
        """Adds the agent to the corresponding utility maps.

        Checks whether the agent is a rioter, and if so, adds it to the corresponding utility map
        based on its team.

        Args:
            agent: Agent to add to the utility map.
        """
        if agent.state == "rioter":
            if agent.team:
                self.home_riot_map[agent.pos[::-1]] += 1
            else:
                self.away_riot_map[agent.pos[::-1]] += 1

    def remove_agent_from_utility_maps(self, agent: FanAgent) -> None:
        """Removes the agent from the corresponding utility maps.

        Checks whether the agent is a rioter, and if so, removes it to the corresponding utility
        map based on its team.

        Args:
            agent: Agent to remove from the utility map.
        """
        if agent.state == "rioter":
            if agent.team:
                self.home_riot_map[agent.pos[::-1]] -= 1
            else:
                self.away_riot_map[agent.pos[::-1]] -= 1

    def _spawn_agents(self, log_switch: bool) -> None:
        """Spawns agents into the grid.

        Checks (for both teams) if the total number of fans has reached the limit.
        If not, add one agent per all the available entry points for the corresponding team.

        Args:
            log_switch: If True, the spawning is logged; otherwise, it is not.
        """
        if self.entered_home_fan_counter < self.n_home_fans:
            self._add_fan_batch(team=True, log_switch=log_switch)
        if self.entered_away_fan_counter < self.n_away_fans:
            self._add_fan_batch(team=False, log_switch=log_switch)

        if log_switch:
            self._agent_info_logging()

    def _add_fan_batch(self, team: bool, log_switch: bool) -> None:
        """Adds a batch of fans for a given team.

        Ensures that only one agent is added per entry point in each batch.

        Args:
            team: The team whose entry points should be used.
            log_switch: If True, the spawning is logged; otherwise, it is not.
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

    def _agent_info_logging(self) -> None:
        """Logs various info about the agents in the model.

        Specifically:
        - Number of agents already entered to the model for both teams separately.
        - Number of injured agents in the model.
        - Number of injured agents at the entry points for both teams separately.
        """
        injured_home = self._count_injured_at_entry_points(True)
        injured_away = self._count_injured_at_entry_points(False)
        num_injured = sum(agent.state == "injured" for agent in self.scheduler.agents)

        info_lines = [
            f"{self.entered_home_fan_counter} Home Fans are present in the Grid",
            f"{self.entered_away_fan_counter} Away Fans are present in the Grid",
            f"Number of injured agents: {num_injured}",
            f"{injured_home} injured home agents at entry points",
            f"{injured_away} injured away agents at entry points",
        ]
        logger.info("\n".join(info_lines))

    def _count_injured_at_entry_points(self, team: bool) -> int:
        """Counts the number of injured agents at the entry points for a given team.

        Args:
            team: Team whose entry points should be used.

        Returns:
            Number of injured agents at the entry point.
        """
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
    n_home_fans = 4500
    n_away_fans = 500
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
    p_injury_upper_bound = 0.1
    willingness_to_riot_thd = 0.1
    n_step = 1000

    logger.info("Starting Riot Simulation")
    logger.info(f"Map size: {width}x{height}, with {n_streets} streets")
    logger.info(f"Street width: {street_width}, exit height: {exit_space_height}")
    logger.info(f"Entry points (home): {entry_points_home}")
    logger.info(f"Entry points (away): {entry_points_away}")
    logger.info(f"Simulation steps: {n_step}")

    city_map = CityMap(width, height, n_streets, street_width, exit_space_height)
    agent_data, _ = RiotModel.run_riot_model(
        width,
        height,
        n_home_fans,
        n_away_fans,
        entry_points_home,
        entry_points_away,
        p_injury_upper_bound,
        willingness_to_riot_thd,
        n_step,
        city_map,
        animate=True,
    )
    logger.info("Riot Simulation Completed. PLotting Results")
    agent_data.plot()
    plt.show()
    # animate_model(frames, city_map.grid, height, width)
