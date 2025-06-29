"""Support functions for the abm model."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abm.model import RiotModel


def count_agents_in_state(model: "RiotModel", target_state: str) -> int:
    """Count how many agents is in a given state."""
    return sum(1 for agent in model.scheduler.agents if agent.state == target_state)
