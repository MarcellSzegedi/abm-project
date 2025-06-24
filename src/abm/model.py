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

from abm.agent import FanAgent
from abm.utils.global_model_parameters import (
    INITIAL_PROB_OF_BASE,
    INITIAL_PROB_OF_RIOT,
    INITIAL_ROUND_OF_ENTRY_AWAY,
    INITIAL_ROUND_OF_ENTRY_HOME,
    MAX_AVAILABLE_AGENT_IN_CELL,
)
from abm.utils.util_func import count_agents_in_state, count_agents_in_team


class RiotModel(Model):
    """Contains the implementation of the Riot model."""

    def __init__(
        self,
        width: int,
        height: int,
        entry_points_home: list[tuple[int, int]],
        entry_points_away: list[tuple[int, int]],
        city_map: npt.NDArray[np.bool] | None = None,
    ) -> None:
        """Initializes the Riot model."""
        super().__init__()

        self.scheduler = RandomActivation(self)

        self.grid = MultiGrid(width=width, height=height, torus=False)
        self.city_map = (
            city_map if city_map is not None else np.ones((height, width), dtype=np.bool)
        )
        self.home_riot_map = np.zeros(shape=(height, width))
        self.away_riot_map = np.zeros(shape=(height, width))

        self.entry_points_home = entry_points_home  # (col, row) format List of entry points
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
        # riot_model._init_population()
        riot_model._spawn_agents()
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
        self.add_agent_to_utility_maps(agent)

    def remove_agent(self, agent: FanAgent) -> None:
        """Removes the agent from the model."""
        self.remove_agent_from_utility_maps(agent)
        self.grid.remove_agent(agent)
        self.scheduler.remove(agent)

    # def _init_population(self) -> None:
    #     """Initializes the population."""
    #     c=0
    #     while self.entered_home_fan_counter < INITIAL_ROUND_OF_ENTRY_HOME:
    #         c+=1
    #         while (
    #             len(self.grid.get_cell_list_contents([self.entry_point_home]))
    #             < MAX_AVAILABLE_AGENT_IN_CELL
    #         ):
    #             print('Adding New agent')
    #             self.add_agent(
    #                 pos=self.entry_point_home,
    #                 team=True,
    #                 state=np.random.choice(
    #                     np.array(["bystander", "rioter"]),
    #                     p=np.array([INITIAL_PROB_OF_BASE, INITIAL_PROB_OF_RIOT]),
    #                 ),
    #             )
    #             self.entered_home_fan_counter += 1
    #         self._spread_fans(team=True)
    #         print(f"Total number of home fans that have entered the grid: {self.entered_home_fan_counter}")
    #         injured_agents = [agent
    #         for agent in self.grid.get_cell_list_contents([self.entry_point_home])
    #         if agent.state == "injured"]
    #         print(f"Number of injured agents at entry point home: {len(injured_agents)}")
    #     print('Finished adding')
    #     while self.entered_away_fan_counter < INITIAL_ROUND_OF_ENTRY_AWAY:
    #         while (
    #             len(self.grid.get_cell_list_contents([self.entry_point_away]))
    #             < MAX_AVAILABLE_AGENT_IN_CELL
    #         ):
    #             self.add_agent(
    #                 pos=self.entry_point_away,
    #                 team=False,
    #                 state=np.random.choice(
    #                     np.array(["bystander", "rioter"]),
    #                     p=np.array([INITIAL_PROB_OF_BASE, INITIAL_PROB_OF_RIOT]),
    #                 ),
    #             )
    #             self.entered_away_fan_counter += 1
    #         self._spread_fans(team=False)

    def _add_fans_batch(self, team: bool) -> None:
        """Adds a batch of fans (5 for us) for a given team. Ensures only one agent is added in an 
        entry point of the grid per batch."""
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
                    self.entered_home_fan_counter +=1
                else:
                    self.entered_away_fan_counter +=1
        print(f"{fans_added} {"home" if team else "away"} fans were added in this Batch")

    def _spawn_agents(self) -> None:
        """Spawns Agents into the Grid """
        while (self.entered_home_fan_counter < INITIAL_ROUND_OF_ENTRY_HOME or self.entered_away_fan_counter< INITIAL_ROUND_OF_ENTRY_AWAY): 
            if self.entered_home_fan_counter < INITIAL_ROUND_OF_ENTRY_HOME:
                self._add_fans_batch(team =True)
            if self.entered_away_fan_counter < INITIAL_ROUND_OF_ENTRY_AWAY:
                self._add_fans_batch(team=False)
            self._spread_fans()
            print(f"{self.entered_home_fan_counter} Home Fans are present in the Grid")
            print(f"{self.entered_away_fan_counter} Away Fans are present in the Grid")
            num_injured = sum(agent.state == "injured" for agent in self.scheduler.agents)
            print(f"Number of injured agents: {num_injured}")
            print(f"{self.count_injured_at_entry_points(True)} Number of injured home agents at the entry points")
            print(f"{self.count_injured_at_entry_points(False)} Number of injured away agents at the entry points")

    
    def count_injured_at_entry_points(self, team: bool) -> int:
        total_injured = 0
        entry_points = entry_points_home if team else entry_points_away
        for entry_point in entry_points:
            agents_in_cell = self.grid.get_cell_list_contents([entry_point])
            injured_agents = [agent for agent in agents_in_cell if agent.state == "injured"]
            total_injured += len(injured_agents)
        return total_injured

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

    def _spread_fans(self) -> None:
        """Distributes the agents during the initialization of the model."""
        agents_to_move = [agent for agent in self.scheduler.agents 
                          if not agent.state == "injured"]
        random.shuffle(agents_to_move)

        agents_by_row = defaultdict(list)
        for agent in agents_to_move:
            agents_by_row[agent.pos[1]].append(agent)

        for row in sorted(list(agents_by_row.keys()), reverse=True):
            for agent in agents_by_row[row]:
                agent.step()


if __name__ == "__main__":
    entry_points_home = [(10, 0), (15, 0), (20, 0),(25,0),(30, 0), (35, 0), (40,0)]
    entry_points_away = [(90, 0), (85, 0), (80, 0), (75, 0), (70,0)]
    agent_data, control_data = RiotModel.run_riot_model(100, 200, entry_points_home, entry_points_away, 1000)

    agent_data.plot()
    plt.show()
