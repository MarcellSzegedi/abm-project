"""Module for performing sensitivity tests on the Riot model."""



from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import morris, sobol
from SALib.sample import morris as morris_sample
from SALib.sample import saltelli

from abm.city_map import CityMap
from abm.model import RiotModel
from abm.utils import global_model_parameters as params


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
            num_samples: int = 2, 
            ): 
        """Initializes the sensitivity test class."""
        self.steps = steps
        self.num_samples = num_samples
        self.problem = problem
        self.width = width
        self.height = height
        self.sobol_matrix = None
    
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
        results = []

        for params_vector in sample:
            riot_prob = params_vector[0]
            n_streets = int(params_vector[1])
            street_width = int(params_vector[2])
            exit_space_height = int(params_vector[3])
            entry_separation = params_vector[4]

            params.INITIAL_PROB_OF_BASE = 1.0 - riot_prob 
            params.INITIAL_PROB_OF_RIOT = riot_prob

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
            
            final_rioters = agent_data["Rioter"].iloc[-1]
            total_agents = agent_data.iloc[-1].sum()
            riot_fraction = final_rioters / total_agents if total_agents > 0 else 0
            
            results.append(riot_fraction)

        return np.array(results)

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
    
    def morris_sensitivity_test(self) -> pd.DataFrame:
        """Performs Morris sensitivity analysis on the model.

        :returns dict: Morris sensitivity indices (Mu, Mu*, Sigma) for each parameter.
        """
        sample_parameters = morris_sample.sample(self.problem, self.num_samples)
        sample_riot_fractions = self.evaluate_model(sample_parameters)
        morris_results = morris.analyze(self.problem, sample_parameters, sample_riot_fractions)

        return pd.DataFrame({
            "Parameter": self.problem["names"],
            "Mu": morris_results["mu"],
            "Mu*": morris_results["mu_star"],
            "Sigma": morris_results["sigma"],
            })

    def plot_sobol_indices(self, sobol_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plots the Sobol sensitivity indices.
        
        :param sobol_df: DataFrame containing Sobol sensitivity indices.
        :param save_path: Path to save the plot (optional).
        """
        plt.figure(figsize=(10, 6))
        x = np.arange(len(sobol_df["Parameter"]))
        bar_width = 0.35

        first_order = sobol_df["First-Order"]
        total_order = sobol_df["Total-Order"]

        plt.bar(
            x - bar_width/2, 
            first_order, 
            width=bar_width, 
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
    
    def plot_morris_analysis(self, morris_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plots the Morris sensitivity analysis results.
        
        :param morris_df: DataFrame containing Morris sensitivity indices.
        :param save_path: Path to save the plot (optional).
        """
        plt.figure(figsize=(10, 5))
        plt.scatter(morris_df["Mu*"], morris_df["Sigma"], s=100, alpha=0.7, color='green')
        
        for i, param in enumerate(morris_df["Parameter"]):
            plt.annotate(param, (morris_df["Mu*"].iloc[i], morris_df["Sigma"].iloc[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel(r'$mu *$')
        plt.ylabel(r'$sigma$')
        plt.title('Morris Analysis')
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(f"results/{save_path}")
        
        plt.show()

  
if __name__ == "__main__":
    problem = {
            "num_vars": 5,
            "names": [
                "INITIAL_PROB_OF_RIOT",      # Agent: Initial probability of riot
                "n_streets",                 # Urban: Number of streets
                "street_width",              # Urban: Street width
                "exit_space_height",         # Urban: Exit space size
                "entry_separation_ratio"     # Urban: How far apart entry points are
            ],
            "bounds": [
                [0.05, 0.5],    # Riot probability (5% to 50%)
                [2, 6],         # Number of streets 
                [5, 15],        # Street width (in cells)
                [5, 20],        # Exit space height (in cells)
                [0.4, 0.8]      # Entry separation (40-80% of width apart)
            ]
        }
    
    sensitivity_test = SensitivityTests(problem)
    sobol_df = sensitivity_test.sobol_sensitivity_test()

    sensitivity_test.plot_sobol_indices(
        sobol_df, 
        save_path="sobol_sensitivity_analysis.png"
        )

    sensitivity_test.plot_interactions(
        sensitivity_test.sobol_matrix, 
        problem["names"], 
        save_path="sobol_interaction_heatmap.png"
        )






        
    


    
    
