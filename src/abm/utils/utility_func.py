"""Support functions for the abm model."""

from typing import TYPE_CHECKING, Any

from abm.model_settings import RiotModelSettings

if TYPE_CHECKING:
    from abm.model import RiotModel


def count_agents_in_state(model: "RiotModel", target_state: str) -> int:
    """Count how many agents is in a given state."""
    return sum(1 for agent in model.scheduler.agents if agent.state == target_state)


def unpack_riot_model_settings(settings: RiotModelSettings) -> dict[str, Any]:
    """Unpack a RiotModelSettings into a dictionary."""
    keys = [
        "width",
        "height",
        "n_home_fans",
        "n_away_fans",
        "entry_points_home",
        "entry_points_away",
        "p_injury_ub",
        "riot_willingness_thd",
        "n_step",
        "city_map",
        "animate",
        "detailed_logging",
    ]
    return {k: getattr(settings, k) for k in keys}
