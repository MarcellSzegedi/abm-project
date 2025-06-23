"""Module for performing sensitivity tests on the Riot model."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        
        Returns:
            dict: Morris sensitivity indices (mu, mu_star, etc.)
        """
        test_values = morris_sampling.sample(self.problem, self.num_samples)
        sample_riot_fractions = self.evaluate_model(test_values)
        morris_results = morris.analyze(self.problem, test_values, sample_riot_fractions)

        return pd.DataFrame({
            "Parameter": self.problem["names"],
            "Mu": morris_results["mu"],
            "Mu*": morris_results["mu_star"],
            "Sigma": morris_results["sigma"],
            })

    
    def ofat_sensitivity_test(self, perturbations: list[float]) -> pd.DataFrame: 
        """Performs One Factor At a Time (OFAT) sensitivity analysis on the model.
        
        Returns:
            dict: Results of the OFAT sensitivity analysis.
        """
        results = []
        param_names = self.problem["names"]
        base_sample = np.array([
            [
                params.INITIAL_PROB_OF_BASE,
                params.INITIAL_PROB_OF_RIOT,
                params.INITIAL_ROUND_OF_ENTRY_HOME,
                params.INITIAL_ROUND_OF_ENTRY_AWAY
            ]
        ])
        baseline_output = self.evaluate_model(base_sample)[0]

        for i, param in enumerate(param_names):
            for delta in perturbations:
                perturbed_sample = base_sample.copy()
                perturbed_sample[0, i] += delta  # apply perturbation to only one parameter

                # Evaluate model
                output = self.evaluate_model(perturbed_sample)[0]

                results.append({
                    "parameter": param,
                    "perturbation": delta,
                    "fraction_rioters": output,
                    "baseline_fraction_rioters": baseline_output,
                    "difference": output - baseline_output
                })

        return pd.DataFrame(results)
    

    def plot_ofat_results(self, ofat_df: pd.DataFrame) -> None:
        """Plots the results of the OFAT sensitivity analysis.
        
        Args:
            ofat_df (pd.DataFrame): DataFrame containing OFAT results.

        THIS WILL GO TO THE CLI EVENTUALLY, BUT FOR NOW IT IS HERE FOR TESTING
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        for param in ofat_df["parameter"].unique():
            subset = ofat_df[ofat_df["parameter"] == param]
            ax.plot(subset["perturbation"], subset["fraction_rioters"], label=param, marker="o")

        ax.set_xlabel("Perturbation")
        ax.set_ylabel("Fraction of Rioters")
        ax.set_title("OFAT Sensitivity Analysis")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model = RiotModel(width=100,
                    height=200,
                    entry_point_home=(10, 0),
                    entry_point_away=(90, 0))

    steps = 10
    problem = {
        "num_vars": 4,
        "names": ["INITIAL_PROB_OF_BASE", "INITIAL_PROB_OF_RIOT", "INITIAL_ROUND_OF_ENTRY_HOME", "INITIAL_ROUND_OF_ENTRY_AWAY"],
        "bounds": [[0, 1], [0, 1], [0, 100], [0, 100]]
    }

    sensitivity_test = SensitivityTests(steps, model.run_riot_model, problem)
    # sobol_df = sensitivity_test.sobol_sensitivity_test()
    # morris_df = sensitivity_test.morris_sensitivity_test()
    pertubations = np.linspace(-0.1, 0.1, 20) 
    ofat_df = sensitivity_test.ofat_sensitivity_test(pertubations)
    sensitivity_test.plot_ofat_results(ofat_df)

    # sobol_df.to_csv("sobol_results.csv", index=False)
    # morris_df.to_csv("morris_results.csv", index=False)
    # ofat_df.to_csv("ofat_results.csv", index=False)

    # Questions for meeting: Sobol uses uses N*(num_vars + 2) samples, where N is the number of samples, do we allow this and run overnight, 
    # or do we use a smaller number of samples? Or parallelise the model runss? 




        
    


    
    
