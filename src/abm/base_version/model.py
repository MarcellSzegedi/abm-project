"""ABM model."""

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from tqdm import trange

from abm.base_version.agent import FanAgent
from abm.base_version.utils.global_model_parameters import (
    INITIAL_PROB_OF_BASE,
    INITIAL_PROB_OF_RIOT,
    INITIAL_ROUND_OF_ENTRY_AWAY,
    INITIAL_ROUND_OF_ENTRY_HOME,
    MAX_AVAILABLE_AGENT_IN_CELL,
)


class RiotModel(Model):
    """Contains the implementation of the Riot model."""

    def __init__(
        self,
        width: int,
        height: int,
        entry_point_home: tuple[int, int],
        entry_point_away: tuple[int, int],
        city_map: npt.NDArray[np.bool] | None = None,
    ) -> None:
        super().__init__()

        self.scheduler = RandomActivation(self)

        self.grid = MultiGrid(width=width, height=height, torus=False)
        self.city_map = city_map if city_map is not None else np.ones((height, width), dtype=np.bool)
        self.base_state_map = np.zeros(shape=(height, width))
        self.riot_state_map = np.zeros(shape=(height, width))
        self.injured_state_map = np.zeros(shape=(height, width))
        self.home_team_map = np.zeros(shape=(height, width))
        self.away_team_map = np.zeros(shape=(height, width))
        self.entry_point_home = entry_point_home  # (col, row) format
        self.entry_point_away = entry_point_away  # (col, row) format

        self.agent_state_datacollector = DataCollector(
            {
                "Base": lambda m: np.sum(m.base_state_map),
                "Riot": lambda m: np.sum(m.riot_state_map),
                "Injured": lambda m: np.sum(m.injured_state_map),
            }
        )
        self.control_team_fan_counter = DataCollector(
            {"Home": lambda m: np.sum(m.home_team_map), "Away": lambda m: np.sum(m.away_team_map)}
        )

        self.entered_home_fan_counter = 0
        self.left_home_fan_counter = 0
        self.entered_away_fan_counter = 0
        self.left_away_fan_counter = 0

    @classmethod
    def run_riot_model(
        cls,
        width: int,
        height: int,
        entry_point_home: tuple[int, int],
        entry_point_away: tuple[int, int],
        n_step: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the abm model."""
        riot_model = cls(width, height, entry_point_home, entry_point_away)
        riot_model._init_population()
        for _ in trange(n_step):
            riot_model.step()

        agent_state_data = riot_model.agent_state_datacollector.get_model_vars_dataframe()
        control_team_data = riot_model.control_team_fan_counter.get_model_vars_dataframe()

        return agent_state_data, control_team_data

    def step(self) -> None:
        """Executes events in one step of the model."""
        self.scheduler.step()
        self.agent_state_datacollector.collect(self)
        self.control_team_fan_counter.collect(self)

    def add_agent(self, pos: tuple[int, int], team: bool, state: str) -> None:
        """Adds a new agent to the model."""
        if len(self.grid.get_cell_list_contents([pos])) >= MAX_AVAILABLE_AGENT_IN_CELL:
            raise ValueError(
                f"Agent cannot be placed at {pos}, as there are too many other agents already."
            )

        agent = FanAgent(pos=pos, unique_id=self.next_id(), team=team, state=state, model=self)
        self.grid.place_agent(agent, pos)
        self.scheduler.add(agent)
        self._add_agent_to_maps(pos=pos, team=team, state=state)

    def remove_agent(self, agent: FanAgent) -> None:
        """Removes the agent from the model."""
        self._remove_agent_from_maps(pos=agent.pos, team=agent.team, state=agent.state)
        self.grid.remove_agent(agent)
        self.scheduler.remove(agent)

    def _init_population(self) -> None:
        """Initializes the population."""
        while self.entered_home_fan_counter < INITIAL_ROUND_OF_ENTRY_HOME:
            while (
                len(self.grid.get_cell_list_contents([self.entry_point_home]))
                < MAX_AVAILABLE_AGENT_IN_CELL
            ):
                self.add_agent(
                    pos=self.entry_point_home,
                    team=True,
                    state=np.random.choice(
                        np.array(["base", "riot"]),
                        p=np.array([INITIAL_PROB_OF_BASE, INITIAL_PROB_OF_RIOT]),
                    ),
                )
                self.entered_home_fan_counter += 1
            self._spread_fans(team=True)
        while self.entered_away_fan_counter < INITIAL_ROUND_OF_ENTRY_AWAY:
            while (
                len(self.grid.get_cell_list_contents([self.entry_point_away]))
                < MAX_AVAILABLE_AGENT_IN_CELL
            ):
                self.add_agent(
                    pos=self.entry_point_away,
                    team=False,
                    state=np.random.choice(
                        np.array(["base", "riot"]),
                        p=np.array([INITIAL_PROB_OF_BASE, INITIAL_PROB_OF_RIOT]),
                    ),
                )
                self.entered_away_fan_counter += 1
            self._spread_fans(team=False)

    def _add_agent_to_maps(self, pos: tuple[int, int], team: bool, state: str) -> None:
        """Adds the agent to the team and state maps."""
        team_map = getattr(self, f"{'home' if team else 'away'}_team_map")
        state_map = getattr(self, f"{state}_state_map")
        team_map[pos[::-1]] += 1
        state_map[pos[::-1]] += 1

    def _remove_agent_from_maps(self, pos: tuple[int, int], team: bool, state: str) -> None:
        """Removes the agent from the team and state maps."""
        team_map = getattr(self, f"{'home' if team else 'away'}_team_map")
        state_map = getattr(self, f"{state}_state_map")
        team_map[pos[::-1]] -= 1
        state_map[pos[::-1]] -= 1

    def _spread_fans(self, team: bool) -> None:
        """Distributes the agents during the initialization of the model."""
        agents_to_move = [agent for agent in self.scheduler.agents if agent.team == team]
        random.shuffle(agents_to_move)

        agents_by_row = defaultdict(list)
        for agent in agents_to_move:
            agents_by_row[agent.pos[1]].append(agent)

        for row in sorted(list(agents_by_row.keys()), reverse=True):
            for agent in agents_by_row[row]:
                agent.spread_agent()


if __name__ == "__main__":
    agent_data, control_data = RiotModel.run_riot_model(100, 200, (10, 0), (90, 0), 1000)

    agent_data.plot()
    plt.show()
