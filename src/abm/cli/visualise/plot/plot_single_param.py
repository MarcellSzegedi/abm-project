"""Command that plots the Rioters / Injured agents, varying one model parameter."""

from typing import Annotated

import typer

from abm.visualisation.plot import PlotRiot

app = typer.Typer()


@app.command(name="plot-riot-single")
def main(
    category: Annotated[
        str,
        typer.Option(
            "--category",
            help="Parameter to be plotted. Can be 'entry_spacing', 'street_width',"
            "'exit_space_height', 'n_streets'.",
        ),
    ],
    values: Annotated[
        list[float],
        typer.Option("--values", help="Chosen ranges or values for chosen parameters."),
    ],
    results_type: Annotated[
        str, typer.Option("--results-type", help="The state to plot.")
    ] = "Rioter",
):
    """Command to plot the number of rioters or injured agents by varying one model parameter.

    Args:
        category: Parameter to be plotted.
        values: Chosen ranges or values for the chosen parameter.
        results_type: Can be 'Rioter' or 'Injured', based on the state to be plotted.
    """
    plot_riot = PlotRiot(results_type=results_type)
    plot_riot.plot_single(category, values)
