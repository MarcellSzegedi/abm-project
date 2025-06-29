"""Module for performing sensitivity tests on the Riot model."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from SALib.analyze import sobol
from SALib.sample import saltelli

from abm.city_map import CityMap
from abm.model import RiotModel
import os

class SensitivityTests:
    """Class to perform sensitivity tests on the Riot model using Sobol's method.

    :param problem: Dictionary defining the parameter space for SALib.
    :param width: Width of the city map (default: 50).
    :param height: Height of the city map (default: 100).
    :param n_home_fans: Number of home fans in the simulation (default: 4500).
    :param n_away_fans: Number of away fans in the simulation (default: 500).
    :param steps: Number of simulation steps to run for each sample (default: 250).
    :param num_samples: Number of samples to generate for Sobol analysis (default: 512).
    """

    def __init__(
        self,
        problem: dict,
        width: int = 50,
        height: int = 100,
        n_home_fans: int = 4500,
        n_away_fans: int = 500,
        steps: int = 1000,
        num_samples: int = 512,
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
        home_exit_ranges: list[range] = [range(14, 19), range(22, 27)],
        away_exit_ranges: list[range] = [range(30, 35)],
    ) -> tuple[CityMap, list[tuple[int, int]], list[tuple[int, int]]]:
        city_map = CityMap(
            width=self.width,
            height=self.height,
            n_streets=n_streets,
            street_width=street_width,
            exit_space_height=exit_space_height,
        )

        cluster_spacing = int(self.width * entry_separation_ratio)

        def shift_exit_ranges(exit_ranges, cluster_spacing):
            """Convert ranges to shifted entry points spaced by cluster_spacing."""
            entry_points = []
            total_clusters = len(exit_ranges)
            total_spacing = (total_clusters - 1) * cluster_spacing
            start_x = (self.width - total_spacing) // 2

            for i, r in enumerate(exit_ranges):
                dx = start_x + i * cluster_spacing
                shifted_points = [(min(x + dx, self.width - 1), 0) for x in r]
                entry_points.extend(shifted_points)
            return entry_points

        entry_points_home = shift_exit_ranges(home_exit_ranges, cluster_spacing)
        entry_points_away = shift_exit_ranges(away_exit_ranges, cluster_spacing)

        return city_map, entry_points_home, entry_points_away

    def evaluate_model(self, sample: np.ndarray) -> np.ndarray:
        """Evaluates the model with a given sample of parameters.

        :param sample: A 2D numpy array where each row is a set of parameters to evaluate.
        """
        total_riot_activity = np.zeros(sample.shape[0])

        def run_one_simulation(i, params_vector):
            try:
                n_streets = int(params_vector[0])
                street_width = int(params_vector[1])
                exit_space_height = int(params_vector[2])
                entry_separation = params_vector[3]
                p_injury_ub = params_vector[4]
                riot_willingness_thd = params_vector[5]

                city_map, entry_home, entry_away = self._create_city_map(
                    n_streets, street_width, exit_space_height, entry_separation
                )

                agent_data, _ = RiotModel.run_riot_model(
                    width=self.width,
                    height=self.height,
                    n_home_fans=self.n_home_fans,
                    n_away_fans=self.n_away_fans,
                    entry_points_home=entry_home,
                    entry_points_away=entry_away,
                    p_injury_ub=p_injury_ub,
                    riot_willingness_thd=riot_willingness_thd,
                    n_step=self.steps,
                    city_map=city_map,
                    animate=False,
                    detailed_logging=False,
                )

                rioters = agent_data["Rioter"]
                # injured = agent_data["Injured"]

                # Calculate the total number of rioters in the simulation
                if not rioters.empty:
                    return {i: rioters.sum()}
                else:
                    return {i: 0}

                # return {i: injured.iloc[-1]}

            except Exception:
                return {i: 0}

        results = Parallel(n_jobs=-1)(
            delayed(run_one_simulation)(i, params_vector) for i, params_vector in enumerate(sample)
        )

        for result in results:
            for key, value in result.items():
                total_riot_activity[key] = value

        return total_riot_activity

    def sobol_sensitivity_test(self) -> pd.DataFrame:
        """Performs Sobol sensitivity analysis on the model.

        :returns dict: Sobol sensitivity indices (S1, ST, etc.) for each parameter.
        """
        sample_parameters = saltelli.sample(self.problem, self.num_samples)
        sample_riot_fractions = self.evaluate_model(sample_parameters)
        sobol_results = sobol.analyze(self.problem, sample_riot_fractions)
        self.sobol_matrix = sobol_results["S2"]

        return pd.DataFrame(
            {
                "Parameter": self.problem["names"],
                "First-Order": sobol_results["S1"],
                "Total-Order": sobol_results["ST"],
                "First-Order Error": sobol_results["S1_conf"],
                "Total-Order Error": sobol_results["ST_conf"],
            }
        )

    def plot_sobol_indices(self, sobol_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plots the Sobol sensitivity indices.

        :param sobol_df: DataFrame containing Sobol sensitivity indices.
        :param save_path: Path to save the plot (optional).
        """
        plt.figure(figsize=(10, 7))
        x = np.arange(len(sobol_df["Parameter"]))
        bar_width = 0.35

        first_order = sobol_df["First-Order"]
        total_order = sobol_df["Total-Order"]
        s1_err = sobol_df["First-Order Error"]
        st_err = sobol_df["Total-Order Error"]

        plt.bar(
            x - bar_width / 2,
            first_order,
            width=bar_width,
            yerr=s1_err,
            label="First-Order",
            color="blue",
            edgecolor="black",
            linewidth=1,
            alpha=0.7,
        )

        plt.bar(
            x + bar_width / 2,
            total_order,
            width=bar_width,
            yerr=st_err,
            label="Total-Order",
            color="purple",
            edgecolor="black",
            linewidth=1,
            alpha=0.7,
        )

        plt.xlabel("Parameters")
        plt.ylabel("Sensitivity Index")
        plt.title("Sobol Sensitivity Analysis")
        plt.xticks(x, sobol_df["Parameter"].tolist(), rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs("results", exist_ok=True)
            plt.savefig(f"results/{save_path}")

        plt.show()

    def plot_interactions(
        self, sobol_matrix: np.ndarray, parameter_names: list, save_path: Optional[str] = None
    ):
        """Plots a heatmap of second-order (interaction) Sobol sensitivity indices."""
        plt.figure(figsize=(8, 6))

        sobol_matrix = np.nan_to_num(sobol_matrix)

        im = plt.imshow(sobol_matrix, cmap="viridis", interpolation="nearest")

        plt.colorbar(im, label="Second-Order Index (S2)")
        plt.xticks(
            ticks=np.arange(len(parameter_names)), labels=parameter_names, rotation=45, ha="right"
        )
        plt.yticks(ticks=np.arange(len(parameter_names)), labels=parameter_names)
        plt.title("Sobol Second-Order Interaction Heatmap")
        plt.tight_layout()

        if save_path:
            plt.savefig(f"results/{save_path}")
        plt.show()


if __name__ == "__main__":
    problem: Dict[str, Any] = {
        "num_vars": 6,
        "names": [
            "n_streets",  # Urban: Number of streets
            "street_width",  # Urban: Street width
            "exit_space_height",  # Urban: Exit space size
            "entry_separation_ratio",  # Urban: How far apart entry points are
            "p_injury_ub",  # Agent: Upper bound for injury probability
            "riot_willingness_thd",  # Agent: Threshold for riot willingness
        ],
        "bounds": [
            [2, 6],  # Number of streets
            [5, 15],  # Street width (in cells)
            [5, 15],  # Exit space height (in cells)
            [0.05, 0.3],  # 5% to 30% separation
            [0.0, 1.0],  # Injury probability upper bound (0% to 100%)
            [0.0, 1.0],  # Riot willingness threshold (0% to 100%)
        ],
    }

    sensitivity_test = SensitivityTests(problem)
    sobol_df = sensitivity_test.sobol_sensitivity_test()
    print(sobol_df)  # TODO: remove, for debugging purposes only

    sensitivity_test.plot_sobol_indices(sobol_df, save_path="sobol_sensitivity_analysis.png")

    if sensitivity_test.sobol_matrix is not None:
        sensitivity_test.plot_interactions(
            sensitivity_test.sobol_matrix,
            problem["names"],
            save_path="sobol_interaction_heatmap.png",
        )
    else:
        print("sobol_matrix is None, skipping plot_interactions")
