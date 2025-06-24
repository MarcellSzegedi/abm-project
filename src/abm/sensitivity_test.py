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
    
    :param steps: Number of steps to run the model.
    :param model_run: Callable function to run the riot model.
    :param problem: Dictionary defining ranges and names of parameters for sensitivity analysis.
    :param num_samples: Number of samples to use for sensitivity analysis (default is 256).
    :param city_map: Instance of CityMap class to use in the model (default is None).
    """
    
    def __init__(
            self, 
            problem: dict, 
            width: int = 100, 
            height: int = 200,
            steps: int = 50, 
            num_samples: int = 4, 
            ): 
        """Initializes the sensitivity test class."""
        self.steps = steps
        self.num_samples = num_samples
        self.problem = problem
        self.width = width
        self.height = height
        self.sobol_matrix: Optional[np.ndarray] = None
    
    def _create_city_map(
            self, 
            n_streets: int, 
            street_width: int, 
            exit_space_height: int, 
            entry_separation_ratio: float):
        """Creates a city map for the Riot model."""
        city_map = CityMap(
            width=self.width,
            height=self.height,
            n_streets=n_streets,
            street_width=street_width,
            exit_space_height=exit_space_height
        )

        separation = int(self.width * entry_separation_ratio / 2)
        entry_point_home = (separation, 0)
        entry_point_away = (self.width - separation, 0)

        return city_map, entry_point_home, entry_point_away
        
    def evaluate_model(self, sample: np.ndarray) -> np.ndarray: 
        """Evaluates the model with a given sample of parameters.
        
        :param sample: A 2D numpy array where each row is a set of parameters to evaluate.
        """
        frac_max_rioters = np.zeros(sample.shape[0])

        for i, params_vector in enumerate(sample):
            n_streets = int(params_vector[0])
            street_width = int(params_vector[1])
            exit_space_height = int(params_vector[2])
            entry_separation = params_vector[3]

            city_map, entry_home, entry_away = self._create_city_map(
                n_streets, street_width, exit_space_height, entry_separation
            )
            
            agent_data, _ = RiotModel.run_riot_model(
                width=self.width,
                height=self.height,
                entry_point_home=entry_home,
                entry_point_away=entry_away,
                n_step=self.steps,
                city_map=city_map
            )

            rioters = agent_data["Rioter"]
            bystanders = agent_data["Bystander"]
            injured = agent_data["Injured"]

            # Calculate the fraction of rioters at the peak rioting time
            if not rioters.empty:
                max_riot_time = rioters.values.argmax()
                max_rioters = rioters.iloc[max_riot_time]
                total_agents_at_peak = (
                    max_rioters +
                    bystanders.iloc[max_riot_time] +
                    injured.iloc[max_riot_time]
                )

                frac_max_rioters[i] = max_rioters / total_agents_at_peak

            else:
                frac_max_rioters[i] = 0

        return frac_max_rioters

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
                [0.4, 0.8]      # Entry separation (40-80% of width apart)
            ] 
            # TODO: There may need to be tests somewhere to make sure ALL combinations of these parameters make sense. 
                #Sometimes, evaluations are not possible because the parameters are not compatible.
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







        
    


    
    
