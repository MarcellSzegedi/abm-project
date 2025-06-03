"""Main entry point of the CLI."""

import typer

from .utils.version import app as app_version

app = typer.Typer()

app.add_typer(app_version)
