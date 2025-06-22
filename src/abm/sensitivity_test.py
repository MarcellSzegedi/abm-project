"""Module for performing sensitivity tests on the Riot model."""

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from typer import Callable

from abm.utils import global_model_parameters as params


class SensitivityTests: 
    """Class to perform sensitivity tests on the Riot model using Sobol's method."""
    
    def __init__(self, steps: int, model_run: Callable, problem: dict, num_samples: int = 1000): 
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
        for i, x in enumerate(sample):
            (
                params.INITIAL_PROB_OF_BASE,
                params.INITIAL_PROB_OF_RIOT,
                params.INITIAL_ROUND_OF_ENTRY_HOME,
                params.INITIAL_ROUND_OF_ENTRY_AWAY,
            ) = x #TODO: Continue with remaining parameters regarding urban design

            agent_state, _ = self.model_run(
                width=100,
                height=200,
                entry_point_home=(10, 0),
                entry_point_away=(90, 0),
                n_step=self.steps
            ) #TODO: Add these to global params as to not hardcode? 

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
        sample_riot_fractions = np.array([self.model_run(params) for params in test_values])
        return sobol.analyze(self.problem, sample_riot_fractions)
    
    
