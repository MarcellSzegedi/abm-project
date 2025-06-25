"""Contains the city map class for the ABM's base version."""

import matplotlib.pyplot as plt
import numpy as np


class CityMap:
    """Represents a city grid with streets and an exit space for agent-based modeling.

    :param width: Width of the grid.
    :param height: Height of the grid.
    :param n_streets: Number of vertical streets.
    :param street_width: Width of each street.
    :param exit_space_height: Height of the exit space at the top of the grid.
    """

    def __init__(self, width, height, n_streets, street_width, exit_space_height=10):
        """Initialize the CityMap."""
        self.width = width
        self.height = height
        self.n_streets = n_streets
        self.street_width = street_width
        self.exit_space_height = exit_space_height

        self.grid = np.zeros((height, width), dtype=np.bool)
        self._set_exit_space()
        self._generate_vertical_streets()

    def _set_exit_space(self) -> None:
        """Set an exit space at the top of the grid where agents can move freely."""
        self.grid[0 : self.exit_space_height, :] = True

    def _generate_vertical_streets(self) -> None:
        """Splits the grid into multiple vertical streets of given width."""
        if self.n_streets <= 0:
            raise ValueError(
                f"Number of streets must be a positive integer. {self.n_streets} given."
            )

        if self.street_width <= 0:
            raise ValueError(
                f"Street width must be a positive integer. {self.street_width} given."
            )

        # Calculate the number of cells needed for the streets
        total_street_width = self.n_streets * self.street_width
        if total_street_width > self.width:
            raise ValueError("Total width of streets exceeds grid width.")

        # Calculate the number of columns available for buildings
        available_columns = self.width - total_street_width
        building_width = available_columns // (self.n_streets + 1)
        if building_width < 1 and self.n_streets > 1:
            raise ValueError("Building width must be at least 1.")

        # Create vertical streets with spacing
        for i in range(self.n_streets):
            start_col = building_width * (i + 1) + i * self.street_width
            end_col = start_col + self.street_width
            self.grid[:, start_col:end_col] = True

    def display(self):
        """Display the grid as a heatmap using matplotlib."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap="Greys", origin="upper")
        plt.title("City Grid")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.colorbar(label="Cell Value")
        plt.show()
