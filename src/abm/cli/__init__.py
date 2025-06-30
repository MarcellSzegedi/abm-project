"""Main entry point of the CLI."""

import typer

from .utils.version import app as app_version
from .visualise import app as app_visualise

app = typer.Typer()

app.add_typer(app_version)
app.add_typer(app_visualise)
