"""Command that plots the Rioters / Injured agents, varying different model parameters."""

from typing import Annotated

import typer

from abm.visualisation.plot import PlotRiot

app = typer.Typer()


@app.command(name="plot-riot-all")
def main(
    results_type: Annotated[
        str, typer.Option("--results-type", help="The state to plot.")
    ] = "Rioter",
):
    """Command to plot the number of rioters / injured agents by varying multiple model parameters.

    Model parameters varied:
    1. Street width.
    2. Exit spacing.
    3. Number of streets.
    4. Exit door spacing.

    Args:
        results_type: The state to plot, either 'Rioter' or 'Injured'.
    """
    plot_riot = PlotRiot(results_type=results_type)
    plot_riot.plot_all()
