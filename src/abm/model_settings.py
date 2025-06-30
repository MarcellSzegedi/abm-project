"""ABM model settings."""

from pydantic import BaseModel, Field, model_validator

from abm.city_map import CityMap


class RiotModelSettings(BaseModel):
    """Model settings for RiotModel.

    Args:
    width: Width of the grid measured as the number of cells.
    height: Height of the grid measured as the number of cells.
    n_home_fans: Total number of home fans to spawn on the grid throughout the simulation.
    n_away_fans: Total number of away fans to spawn on the grid throughout the simulation.
    entry_points_home: Coordinates of the spawn points of the home fan agents
                        in (col, row) format.
    entry_points_away: Coordinates of the spawn points of the away fan agents
                        in (col, row) format.
    p_injury_ub: Upper bound of the probability of injury for the agents.
    riot_willingness_thd: Threshold of the willingness to riot for the agents.
    n_step: Number of simulation steps.
    city_map: Boolean numpy array representing the city map,
                        with True values in the places where agents can move
                        and False where they cannot.
    animate: True if the simulation should be animated, False otherwise.
    detailed_logging: True if the simulation should be logged, False otherwise.
    """

    width: int = Field(50, description="Width of the grid measured as the number of cells.", gt=0)
    height: int = Field(
        100, description="Height of the grid measured as the number of cells.", gt=0
    )
    n_home_fans: int = Field(4500, description="Total number of home fans", gt=0)
    n_away_fans: int = Field(500, description="Total number of away fans", gt=0)
    entry_points_home: list[tuple[int, int]] = Field(
        [(i, 0) for i in range(14, 19)] + [(i, 0) for i in range(22, 27)],
        description="Coordinates of the spawn points of the home fan agents in (col, row) format.",
    )
    entry_points_away: list[tuple[int, int]] = Field(
        [(i, 0) for i in range(30, 35)],
        description="Coordinates of the spawn points of the away fan agents in (col, row) format.",
    )
    p_injury_ub: float = Field(
        0.1, description="Upper bound of the probability of injury for the agents.", ge=0, le=1
    )
    riot_willingness_thd: float = Field(
        0.1, description="Threshold of the willingness to riot for the agents.", ge=0, le=1
    )
    n_step: int = Field(1000, description="Number of simulation steps.", gt=0)
    n_streets: int = Field(4, description="Number of streets in the city map.", gt=0)
    street_width: int = Field(7, description="Width of the streets in the city map.", gt=0)
    exit_space_height: int = Field(10, description="Height of the exit", gt=0)
    city_map: CityMap | None = Field(
        None,
        description="Boolean numpy array representing the city map, with True values in the "
        "places where agents can move and False where they cannot.",
    )
    animate: bool = Field(False, description="True if the simulation should be animated.")
    detailed_logging: bool = Field(False, description="True if the simulation should be logged.")
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def add_city_map(self):
        """Adds the city_map variable instance to the model settings."""
        if self.city_map is None:
            self.city_map = CityMap(
                self.width, self.height, self.n_streets, self.street_width, self.exit_space_height
            )
        return self
