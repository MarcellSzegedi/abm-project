"""Logging configuration."""

import logging


def setup_logging() -> None:
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="riot_simulation.log",
        filemode="w",
    )
