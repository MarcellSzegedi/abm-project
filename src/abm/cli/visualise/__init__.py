"""Entry point for the visualisation commands."""

import typer

from .plot.plot_all_param import app as app_plot_all_param
from .plot.plot_single_param import app as app_plot_single_param

app = typer.Typer()

app.add_typer(app_plot_single_param)
app.add_typer(app_plot_all_param)
