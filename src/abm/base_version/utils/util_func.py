"""Support functions for the abm model."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abm.base_version.model import RiotModel


def count_agents_in_state(model: "RiotModel", target_state: str) -> int:
    """Count how many agents is in a given state."""
    return sum(1 for agent in model.scheduler.agents if agent.state == target_state)


def count_agents_in_team(model: "RiotModel", target_team: bool) -> int:
    """Count how many agents is in a given team."""
    return sum(1 for agent in model.scheduler.agents if agent.team == target_team)
