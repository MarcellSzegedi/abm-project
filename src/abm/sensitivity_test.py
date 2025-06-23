"""Module for performing sensitivity tests on the Riot model."""

from typing import Callable

import numpy as np
import pandas as pd
import concurrent.futures
from SALib.analyze import morris, sobol
from SALib.sample import sobol as sobol_sampling, morris as morris_sampling
from SALib.sample import saltelli

from abm.model import RiotModel
from abm.city_map import CityMap
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
            num_samples: int = 1000, 
            ): 
        """Initializes the sensitivity test class."""
        self.steps = steps
        self.num_samples = num_samples
        self.problem = problem
        self.width = width
        self.height = height
    
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

    def sobol_sensitivity_test(self) -> dict:
        """Performs Sobol sensitivity analysis on the model.
        
        :returns dict: Sobol sensitivity indices (S1, ST, etc.) for each parameter.
        """
        sample_parameters = morris_sampling.sample(self.problem, self.num_samples)
        sample_riot_fractions = self.evaluate_model(sample_parameters)
        sobol_results = sobol.analyze(self.problem, sample_riot_fractions)

        return pd.DataFrame({
            "Parameter": self.problem["names"],
            "First-Order": sobol_results["S1"],
            "Total-Order": sobol_results["ST"],
            "First-Order Error": sobol_results["S1_conf"],
            "Total-Order Error": sobol_results["ST_conf"],
        })
    
    def morris_sensitivity_test(self) -> dict:
        """Performs Morris sensitivity analysis on the model.

        :returns dict: Morris sensitivity indices (Mu, Mu*, Sigma) for each parameter.
        """
        sample_parameters = morris_sampling.sample(self.problem, self.num_samples)
        sample_riot_fractions = self.evaluate_model(sample_parameters)
        morris_results = morris.analyze(self.problem, sample_parameters, sample_riot_fractions)

        return pd.DataFrame({
            "Parameter": self.problem["names"],
            "Mu": morris_results["mu"],
            "Mu*": morris_results["mu_star"],
            "Sigma": morris_results["sigma"],
            })
    

if __name__ == "__main__":
    problem = {
            "num_vars": 5,
            "names": [
                "INITIAL_PROB_OF_RIOT",      # Key agent parameter
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
    # sobol_df = sensitivity_test.sobol_sensitivity_test()
    morris_df = sensitivity_test.morris_sensitivity_test()
    print(morris_df)





        
    


    
    
