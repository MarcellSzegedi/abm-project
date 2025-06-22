"""Module for performing sensitivity tests on the Riot model."""

import numpy as np
import pandas as pd
from SALib.analyze import sobol, morris 
from SALib.sample import saltelli, morris as morris_sampling
from typing import Callable

from abm.utils import global_model_parameters as params
from abm.model import RiotModel


class SensitivityTests: 
    """Class to perform sensitivity tests on the Riot model using Sobol's method."""
    
    def __init__(self, steps: int, model_run: Callable, problem: dict, num_samples: int = 2): 
        """Initializes the sensitivity test class."""
        self.steps = steps
        self.model_run = model_run
        self.problem = problem
        self.num_samples = num_samples
        self.results = None

    def evaluate_model(self, sample: np.ndarray) -> np.ndarray: 
        """Evaluates the model with a given sample of parameters.
        
        Args:
            sample (np.ndarray): Sample of parameters to evaluate final number of rioters.
        
        Returns:
            np.ndarray: Array of final fractions of rioters after the simulation.
        """
        results = []
        for x in sample:
            (
                params.INITIAL_PROB_OF_BASE,
                params.INITIAL_PROB_OF_RIOT,
                params.INITIAL_ROUND_OF_ENTRY_HOME,
                params.INITIAL_ROUND_OF_ENTRY_AWAY,
            ) = x # Sets global parameter values based on sample drawn.
            #TODO: Continue with remaining parameters regarding urban design.

            agent_state, _ = self.model_run(
                width=100,
                height=200,
                entry_point_home=(10, 0),
                entry_point_away=(90, 0),
                n_step=self.steps
            ) # Runs the model with the given parameters.
            #TODO: Add these to global params as to not hardcode? 

            final_rioters = agent_state["Rioter"].iloc[-1] 
            total = agent_state.iloc[-1].sum()
            results.append(final_rioters / total if total else 0)

        return np.array(results)

    def sobol_sensitivity_test(self) -> dict:
        """Performs Sobol sensitivity analysis on the model.
        
        Returns:
            dict: Sobol sensitivity indices (first-order, total-order, etc.)
        """
        test_values = saltelli.sample(self.problem, self.num_samples)
        sample_riot_fractions = self.evaluate_model(test_values)
        return sobol.analyze(self.problem, sample_riot_fractions)
    
    def morris_sensitivity_test(self) -> dict:
        """Performs Morris sensitivity analysis on the model.
        
        Returns:
            dict: Morris sensitivity indices (mu, mu_star, etc.)
        """
        test_values = morris_sampling.sample(self.problem, self.num_samples)
        sample_riot_fractions = self.evaluate_model(test_values)
        return morris.analyze(self.problem, sample_riot_fractions)
    
    def ofat_sensitivity_test(self, pertubations: list[float]) -> pd.DataFrame: 
        """Performs One Factor At a Time (OFAT) sensitivity analysis on the model.
        
        Returns:
            dict: Results of the OFAT sensitivity analysis.
        """
        results = []
        initial_values = {
            "INITIAL_PROB_OF_BASE": params.INITIAL_PROB_OF_BASE,
            "INITIAL_PROB_OF_RIOT": params.INITIAL_PROB_OF_RIOT,
            "INITIAL_ROUND_OF_ENTRY_HOME": params.INITIAL_ROUND_OF_ENTRY_HOME,
            "INITIAL_ROUND_OF_ENTRY_AWAY": params.INITIAL_ROUND_OF_ENTRY_AWAY,
        }

        for delta in pertubations:
            for param in initial_values.keys():
                # Perturb the parameter
                original_value = initial_values[param]
                perturbed_value = original_value + delta
                setattr(params, param, perturbed_value)

                # Run the model
                agent_state, _ = self.model_run(
                    width=100,
                    height=200,
                    entry_point_home=(10, 0),
                    entry_point_away=(90, 0),
                    n_step=self.steps
                )

                final_rioters = agent_state["Rioter"].iloc[-1]
                total = agent_state.iloc[-1].sum()
                fraction_rioters = final_rioters / total if total else 0

                results.append({
                    "parameter": param,
                    "perturbation": delta,
                    "fraction_rioters": fraction_rioters,
                    "difference": fraction_rioters - (initial_values[param] / total if total else 0)
                })

                # Reset the parameter to its original value
                setattr(params, param, original_value)

if __name__ == "__main__":
    model = RiotModel(width=100,
                    height=200,
                    entry_point_home=(10, 0),
                    entry_point_away=(90, 0))

    steps = 1000 
    problem = {
        "num_vars": 4,
        "names": ["INITIAL_PROB_OF_BASE", "INITIAL_PROB_OF_RIOT", "INITIAL_ROUND_OF_ENTRY_HOME", "INITIAL_ROUND_OF_ENTRY_AWAY"],
        "bounds": [[0, 1], [0, 1], [0, 100], [0, 100]]
    }

    sensitivity_test = SensitivityTests(steps, model.run_riot_model, problem)
    sobol_results = sensitivity_test.sobol_sensitivity_test()
    morris_results = sensitivity_test.morris_sensitivity_test()
    pertubations = np.linspace(-0.1, 0.1, 3)  # Example perturbations
    ofat_results = sensitivity_test.ofat_sensitivity_test(pertubations)
    
    print("Sobol Results:", sobol_results)
    print("Morris Results:", morris_results)
    print("OFAT Results:", ofat_results)




        
    


    
    
