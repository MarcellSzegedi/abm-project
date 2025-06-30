"""Command to create an animation of a single simulation of the RiotModel."""

import logging
from typing import Annotated

import typer

from abm.model import RiotModel
from abm.model_settings import RiotModelSettings
from abm.utils.logging_config import setup_logging
from abm.utils.utility_func import unpack_riot_model_settings
from abm.visualisation.animation import animate_model

setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command(name="animate")
def main(
    n_step: Annotated[
        int, typer.Option("--n-step", help="Number of steps to simulate maximum.", min=10)
    ] = 1000,
):
    """Creates an animation of a single simulation of the RiotModel."""
    model_settings = RiotModelSettings(n_step=n_step, animate=True)
    settings = unpack_riot_model_settings(model_settings)
    _, frames = RiotModel.run_riot_model(**settings)
    logger.info("Simulation done, animation started.")
    animate_model(
        frames, model_settings.city_map.grid, model_settings.height, model_settings.width
    )
