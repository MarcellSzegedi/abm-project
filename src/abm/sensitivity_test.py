"""Module for performing sensitivity tests on the Riot model."""



from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli

from abm.city_map import CityMap
from abm.model import RiotModel


class SensitivityTests: 
    """Class to perform sensitivity tests on the Riot model using Sobol's method.

    :param problem: 
    :param width: 
    :param height: 
    :param n_home_fans: 
    :param n_away_fans:
    :param steps:
    :param num_samples:
    """

    def __init__(
            self, 
            problem: dict, 
            width: int = 50, 
            height: int = 100,
            n_home_fans: int = 4500,
            n_away_fans: int = 500,
            steps: int = 100, 
            num_samples: int = 256, 
            ): 
        """Initializes the sensitivity test class."""
        self.steps = steps
        self.num_samples = num_samples
        self.problem = problem
        self.width = width
        self.height = height
        self.n_home_fans = n_home_fans
        self.n_away_fans = n_away_fans
        self.sobol_matrix: Optional[np.ndarray] = None
    
    def _create_city_map(
        self,
        n_streets: int,
        street_width: int,
        exit_space_height: int,
        entry_separation_ratio: float,
        n_home_clusters: int = 2,
        n_away_clusters: int = 1,
        cluster_size: int = 5,
    ) -> tuple[CityMap, list[tuple[int, int]], list[tuple[int, int]]]:

        city_map = CityMap(
            width=self.width,
            height=self.height,
            n_streets=n_streets,
            street_width=street_width,
            exit_space_height=exit_space_height,
        )

        entry_points_home = [] 
        entry_points_away = [] 

        cluster_spacing = int(self.width * entry_separation_ratio) # Space between centers of clusters
        if n_home_clusters > 0:
            space_between_home = (n_home_clusters - 1) * cluster_spacing # Total space between centers of clusters
            home_start_x = (self.width - space_between_home) // 2 # X position of the first cluster

            for i in range(n_home_clusters):
                center = home_start_x + i * cluster_spacing 
                start = max(center - cluster_size // 2, 0)
                end = min(start + cluster_size, self.width)
                entry_points_home.extend((x, 0) for x in range(start, end))
        
        if n_away_clusters > 0:
            space_between_away = (n_away_clusters - 1) * cluster_spacing
            away_start_x = (self.width - space_between_away) // 2

            for i in range(n_away_clusters):
                center = away_start_x + i * cluster_spacing
                start = max(center - cluster_size // 2, 0)
                end = min(start + cluster_size, self.width)
                entry_points_away.extend((x, 0) for x in range(start, end))

        return city_map, entry_points_home, entry_points_away

    def evaluate_model(self, sample: np.ndarray) -> np.ndarray: 
        """Evaluates the model with a given sample of parameters.
        
        :param sample: A 2D numpy array where each row is a set of parameters to evaluate.
        """
        total_riot_activity = np.zeros(sample.shape[0])

        for i, params_vector in enumerate(sample):
            try: 
                n_streets = int(params_vector[0])
                street_width = int(params_vector[1])
                exit_space_height = int(params_vector[2])
                entry_separation = params_vector[3]

                city_map, entry_home, entry_away = self._create_city_map(
                    n_streets, street_width, exit_space_height, entry_separation
                )
                
                agent_data, _, _ = RiotModel.run_riot_model(
                    width=self.width,
                    height=self.height,
                    n_home_fans=self.n_home_fans,     
                    n_away_fans=self.n_away_fans,
                    entry_points_home=entry_home,  
                    entry_points_away=entry_away,
                    n_step=self.steps,
                    city_map=city_map,
                    animate=False,
                    detailed_logging=False,
                )

                rioters = agent_data["Rioter"]

                # Calculate the total number of rioters in the simulation
                if not rioters.empty:
                    total_riot_activity[i] = rioters.sum()
                else:
                    total_riot_activity[i] = 0

            except Exception: 
                # print(f"Invalid parameter combination: Streets: {n_streets}, Street Width: {street_width}, Exit Space: {exit_space_height}, Space Between Doors: {entry_separation}")
                total_riot_activity[i] = 0

        return total_riot_activity

    def sobol_sensitivity_test(self) -> pd.DataFrame:
        """Performs Sobol sensitivity analysis on the model.
        
        :returns dict: Sobol sensitivity indices (S1, ST, etc.) for each parameter.
        """
        sample_parameters = saltelli.sample(self.problem, self.num_samples)
        sample_riot_fractions = self.evaluate_model(sample_parameters)
        sobol_results = sobol.analyze(self.problem, sample_riot_fractions)
        self.sobol_matrix = sobol_results["S2"]

        return pd.DataFrame({
            "Parameter": self.problem["names"],
            "First-Order": sobol_results["S1"],
            "Total-Order": sobol_results["ST"],
            "First-Order Error": sobol_results["S1_conf"],
            "Total-Order Error": sobol_results["ST_conf"],
        })

    def plot_sobol_indices(self, sobol_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plots the Sobol sensitivity indices.
        
        :param sobol_df: DataFrame containing Sobol sensitivity indices.
        :param save_path: Path to save the plot (optional).
        """
        plt.figure(figsize=(10, 5))
        x = np.arange(len(sobol_df["Parameter"]))
        bar_width = 0.35

        first_order = sobol_df["First-Order"]
        total_order = sobol_df["Total-Order"]
        s1_err      = sobol_df["First-Order Error"]
        st_err      = sobol_df["Total-Order Error"]


        plt.bar(
            x - bar_width/2, 
            first_order, 
            width=bar_width, 
            yerr=s1_err,
            label='First-Order', 
            color='blue',
            edgecolor='black',
            linewidth=1,
            alpha=0.7
            )
        
        plt.bar(
            x + bar_width/2, 
            total_order, 
            width=bar_width, 
            yerr=st_err,
            label='Total-Order', 
            color='purple',
            edgecolor='black',
            linewidth=1,
            alpha=0.7
            ) 

        plt.xlabel('Parameters')
        plt.ylabel('Sensitivity Index')
        plt.title('Sobol Sensitivity Analysis')
        plt.xticks(x, sobol_df["Parameter"].tolist(), rotation=45, ha='right')
        plt.legend()

        if save_path:
            plt.savefig(f"results/{save_path}")
        
        plt.show()
    
    def plot_interactions(
            self, 
            sobol_matrix: np.ndarray, 
            parameter_names: list, 
            save_path: Optional[str] = None
            ):
        """Plots a heatmap of second-order (interaction) Sobol sensitivity indices."""
        plt.figure(figsize=(8, 6))
        
        sobol_matrix = np.nan_to_num(sobol_matrix)

        im = plt.imshow(sobol_matrix, cmap="viridis", interpolation='nearest')

        plt.colorbar(im, label="Second-Order Index (S2)")
        plt.xticks(
            ticks=np.arange(len(parameter_names)), 
            labels=parameter_names, 
            rotation=45, 
            ha='right'
            )
        plt.yticks(
            ticks=np.arange(len(parameter_names)), 
            labels=parameter_names
            )
        plt.title("Sobol Second-Order Interaction Heatmap")
        plt.tight_layout()

        if save_path:
            plt.savefig(f"results/{save_path}")
        plt.show()

if __name__ == "__main__":
    problem: Dict[str, Any] = {
            "num_vars": 4,
            "names": [
                "n_streets",                 # Urban: Number of streets
                "street_width",              # Urban: Street width
                "exit_space_height",         # Urban: Exit space size
                "entry_separation_ratio"     # Urban: How far apart entry points are
            ],
            "bounds": [
                [2, 6],         # Number of streets 
                [5, 15],        # Street width (in cells)
                [5, 15],        # Exit space height (in cells)
                [0.05, 0.3]     # 5% to 30% separation
            ] 
        }
    
    sensitivity_test = SensitivityTests(problem)
    sobol_df = sensitivity_test.sobol_sensitivity_test()

    sensitivity_test.plot_sobol_indices(
        sobol_df, 
        save_path="sobol_sensitivity_analysis.png"
        )

    if sensitivity_test.sobol_matrix is not None:
        sensitivity_test.plot_interactions(
            sensitivity_test.sobol_matrix,
            problem["names"],
            save_path="sobol_interaction_heatmap.png"
        )
    else:
        print("sobol_matrix is None, skipping plot_interactions")