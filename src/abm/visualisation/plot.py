"""Functions to plot the results of the ABM."""

import matplotlib.pyplot as plt
import numpy as np

from abm.city_map import CityMap
from abm.model import RiotModel


class PlotRiot: 
    """Generates plots to visualize the effect of agent movement parameters.

    Includes effects based on street width, number of streets, space between exits,
    and exit space height.
    """
    def __init__(
        self, 
        width: int = 50, 
        height: int = 100, 
        n_home: int = 4500, 
        n_away: int = 500, 
        n_streets: int = 3, 
        street_width: int = 7, 
        exit_space_height: int = 10, 
        p_injury_ub: float = 0.03, 
        riot_willingness_thd: float = 0.5, 
        n_steps: int = 250,
        n_runs: int = 30
    ):
        """Initilises the plotting class with initial values for parameters."""
        self.width = width
        self.height = height
        self.n_home = n_home
        self.n_away = n_away
        self.n_streets = n_streets
        self.street_width = street_width
        self.exit_space_height = exit_space_height
        self.p_injury_ub = p_injury_ub
        self.riot_willingness_thd = riot_willingness_thd
        self.n_steps = n_steps
        self.n_runs = n_runs
        self.steps = np.arange(n_steps + 1)

        # Set default entry points
        self.entry_home = [(i, 0) for i in range(14, 19)] + [(i, 0) for i in range(22, 27)]
        self.entry_away = [(i, 0) for i in range(30, 35)]


    def run_simulations(self, city_map):
        """Run n runs of the RiotModel on city map.
        
        :returns: mean and std of rioter counts.
        """
        data = []
        for _ in range(self.n_runs):
            agent_data, _ = RiotModel.run_riot_model(
                self.width,
                self.height,
                self.n_home,
                self.n_away,
                self.entry_home,
                self.entry_away,
                self.p_injury_ub, 
                self.riot_willingness_thd,
                self.n_steps,
                city_map,
                animate=False,
                detailed_logging=False
            )
            data.append(agent_data['Rioter'].to_numpy())
        arr = np.array(data)
        return arr.mean(axis=0), arr.std(axis=0)

    def plot_all(self):
        """Plots 4 subplots measuring number of rioters.

        1. Varying street width.
        2. Varying exit spacing.
        3. Varying number of streets.
        4. Varying exit door spacing.
        """
        _, axs = plt.subplots(2, 2, figsize=(16, 8))

        # Subplot 1: Varying street width
        for street_width in [5, 10, 15]:
            city_map = CityMap(
                self.width, 
                self.height, 
                n_streets=self.n_streets, 
                street_width=street_width, 
                exit_space_height=self.exit_space_height
                )
            mean, std = self.run_simulations(city_map)
            line, = axs[0,0].plot(self.steps, mean, label=f"Width {street_width}")
            axs[0,0].fill_between(
                self.steps, 
                mean-std, 
                mean+std, 
                alpha=0.3, 
                color=line.get_color()
                )
        axs[0,0].set_title('Street Width')
        axs[0,0].set_xlabel('Step')
        axs[0,0].set_ylabel('Rioters')
        axs[0,0].legend()

        # Subplot 2: Varying exit height
        for exit_height in [2, 6, 10]:
            city_map = CityMap(
                self.width, 
                self.height, 
                n_streets=self.n_streets, 
                street_width=self.street_width, 
                exit_space_height=exit_height
                )
            mean, std = self.run_simulations(city_map)
            line, = axs[0,1].plot(self.steps, mean, label=f"Exit {exit_height}")
            axs[0,1].fill_between(
                self.steps, 
                mean-std, 
                mean+std, 
                alpha=0.3, 
                color=line.get_color()
                )
        axs[0,1].set_title('Exit Height')
        axs[0,1].set_xlabel('Step')
        axs[0,1].set_ylabel('Rioters')
        axs[0,1].legend()

        # Subplot 3: Varying number of streets 
        for n_streets in [1, 3, 5]:
            city_map = CityMap(
                self.width, 
                self.height, 
                n_streets=n_streets,
                street_width=self.street_width, 
                exit_space_height=self.exit_space_height
                )
            mean, std = self.run_simulations(city_map)
            line, = axs[1,0].plot(self.steps, mean, label=f"Streets {n_streets}")
            axs[1,0].fill_between(
                self.steps, 
                mean-std, 
                mean+std,
                alpha=0.3, 
                color=line.get_color()
                )
        axs[1,0].set_title('Number of Streets')
        axs[1,0].set_xlabel('Step')
        axs[1,0].set_ylabel('Rioters')
        axs[1,0].legend()

        # Subplot 4: Varying exit door placement
        exit_scenarios = [
            (range(15,20), range(29,34), '2 spaces'),
            (range(12,17), range(32,37), '5 spaces'),
            (range(7,12), range(37,42), '10 spaces')
        ]
        for left, right, label in exit_scenarios:
            self.entry_home = [(i,0) for i in left] + [(i,0) for i in range(22,27)]
            self.entry_away = [(i,0) for i in right]
            city_map = CityMap(
                self.width, 
                self.height, 
                n_streets=self.n_streets, 
                street_width=self.street_width, 
                exit_space_height=self.exit_space_height
                )
            mean, std = self.run_simulations(city_map)
            line, = axs[1,1].plot(self.steps, mean, label=label)
            axs[1,1].fill_between(
                self.steps, 
                mean-std, 
                mean+std, 
                alpha=0.3, 
                color=line.get_color()
                )
        axs[1,1].set_title('Entry Spacing')
        axs[1,1].set_xlabel('Step')
        axs[1,1].set_ylabel('Rioters')
        axs[1,1].legend()

        plt.tight_layout()
        plt.show()

    def plot_single(self, category, values):
        """Plot one parameter at a time.
        
        :param category: Parameter to be plotted
        :param values: Chosen ranges or values for chosen parameters
        """
        _, ax = plt.subplots(figsize=(8,5))
        if category == 'entry_spacing':
            for left, right, label in values:
                entry_home = [(i,0) for i in left] + [(i,0) for i in range(22,27)]
                entry_away = [(i,0) for i in right]
                city_map = CityMap(
                    self.width, 
                    self.height, 
                    n_streets=self.n_streets, 
                    street_width=self.street_width, 
                    exit_space_height=self.exit_space_height
                    )
                mean, std = self.run_simulations(
                    city_map, 
                    entry_home=entry_home, 
                    entry_away=entry_away
                    )
                line, = ax.plot(self.steps, mean, label=label)
                ax.fill_between(self.steps, mean-std, mean+std, alpha=0.3, color=line.get_color())
        else:
            for val in values:
                if category == 'street_width':
                    city_map = CityMap(
                        self.width, 
                        self.height, 
                        n_streets=self.n_streets, 
                        street_width=val, 
                        exit_space_height=self.exit_space_height
                        )
                elif category == 'exit_space_height':
                    city_map = CityMap(
                        self.width, 
                        self.height, 
                        n_streets=self.n_streets, 
                        street_width=self.street_width, 
                        exit_space_height=val
                        )
                elif category == 'n_streets':
                    city_map = CityMap(
                        self.width, 
                        self.height, 
                        n_streets=val, 
                        street_width=self.street_width, 
                        exit_space_height=self.exit_space_height
                        )
                else:
                    raise ValueError('Invalid category')
                mean, std = self.run_simulations(city_map)
                line, = ax.plot(self.steps, mean, label=f"{category} {val}")
                ax.fill_between(self.steps, mean-std, mean+std, alpha=0.3, color=line.get_color())
        ax.set_title(category.replace('_',' ').title())
        ax.set_xlabel('Step')
        ax.set_ylabel('Rioters')
        ax.legend()
        plt.show()


if __name__ == "__main__": 
    plot_riot = PlotRiot()
    plot_riot.plot_all()



