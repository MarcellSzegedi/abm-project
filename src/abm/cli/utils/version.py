"""Version of the ABM package."""

import typer

app = typer.Typer()
__version__ = "0.1.0"


@app.command()
def version():
    """Show the version of the package."""
    print(f"The version of the ABM package is {__version__}.")
