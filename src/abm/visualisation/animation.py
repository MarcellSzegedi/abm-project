"""Functions to animate the flow of the ABM."""

import warnings
from typing import TYPE_CHECKING, TypedDict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

from abm.utils.global_model_parameters import AGENT_ANIM_PLACEMENT_COL, AGENT_ANIM_PLACEMENT_ROW

if TYPE_CHECKING:
    from abm.model import RiotModel

rcParams["animation.embed_limit"] = 50 * 1024 * 1024


class CellInfoContainer(TypedDict):
    """Dictionary to hold the relevant information about a cell for animation."""

    n_agents: int
    agents: list[dict[str, int | str | tuple[float, float, float]] | None]


def get_grid_data(model: "RiotModel") -> list[CellInfoContainer]:
    """Collects information about the agents for every frame of the model."""
    one_frame_content = []
    for cell in model.grid.coord_iter():
        cell_content, (col, row) = cell
        temp_cell_container: CellInfoContainer = {"n_agents": len(cell_content), "agents": []}
        for agent in cell_content:
            color = (0.0, 0.0, 1.0) if agent.team else (1.0, 0.0, 0.0)
            if agent.state == "injured":
                color = (0.0, 0.0, 0.0)

            temp_cell_container["agents"].append(
                {
                    "x": col,
                    "y": row,
                    "edgecolor": color,
                    "facecolor": color if agent.state != "bystander" else "none",
                    "marker": "o",
                }
            )
        one_frame_content.append(temp_cell_container)

    return one_frame_content


def animate_model(
    model_results: list[list[CellInfoContainer]] | None,
    city_map: npt.NDArray[np.bool],
    height: int,
    width: int,
    filename: str = "abm_animation.html",
) -> None:
    """Animates the results of the ABM."""
    if model_results is not None:
        fig, ax = plt.subplots(figsize=(width * 0.5, height * 0.5))

        def update(frame):
            ax.clear()
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect("equal")
            ax.set_xticks(range(width))
            ax.set_yticks(range(height))
            ax.grid(True, which="both", color="lightgray", linestyle="--", linewidth=0.5)

            rows, cols = city_map.shape
            for y in range(rows):
                for x in range(cols):
                    if not city_map[y, x]:
                        rect = patches.Rectangle(
                            (x, y), 1, 1, facecolor="darkgrey", edgecolor="none", zorder=0
                        )
                        ax.add_patch(rect)

            container = []

            for cell in frame:
                for i, agent in enumerate(cell["agents"]):
                    sct = ax.scatter(
                        agent["x"] + AGENT_ANIM_PLACEMENT_COL[cell["n_agents"]][i],
                        agent["y"] + AGENT_ANIM_PLACEMENT_ROW[cell["n_agents"]][i],
                        s=50,
                        edgecolor=agent["edgecolor"],
                        facecolor=agent["facecolor"],
                        marker=agent["marker"],
                        linewidth=0.5,
                        zorder=1,
                    )
                    container.append(sct)

            return container

        ani = FuncAnimation(fig, update, frames=model_results, blit=True, repeat=False)
        html_str = ani.to_jshtml()
        with open(filename, "w") as f:
            f.write(html_str)
        print(f"Animation saved to {filename}")
    else:
        warnings.warn("No results to animate.", UserWarning)
