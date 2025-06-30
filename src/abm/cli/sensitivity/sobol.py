"""Sobol sensitivity test."""

from typing import Any

import typer

from abm.sensitivity_test import SensitivityTests

app = typer.Typer()


@app.command(name="sobol")
def main():
    """Produces the sobol sensitivity analysis."""
    problem: dict[str, Any] = {
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
            [2, 15],  # Exit space height (in cells)
            [0.05, 0.3],  # 5% to 30% separation
            [0.0, 1.0],  # Injury probability upper bound (0% to 20%)
            [0.0, 1.0],  # Riot willingness threshold (0% to 100%)
        ],
    }

    results_type = "Rioter"  # or "Injured"

    sensitivity_test = SensitivityTests(problem, results_type=results_type)
    sobol_df = sensitivity_test.sobol_sensitivity_test()
    sensitivity_test.plot_sobol_indices(sobol_df, save_path="sobol_sensitivity_analysis.png")

    if sensitivity_test.sobol_matrix is not None:
        sensitivity_test.plot_interactions(
            sensitivity_test.sobol_matrix,
            problem["names"],
            save_path="sobol_interaction_heatmap.png",
        )
    else:
        print("sobol_matrix is None, skipping plot_interactions")
