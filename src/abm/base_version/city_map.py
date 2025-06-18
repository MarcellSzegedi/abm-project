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
        self.set_exit_space(exit_space_height)
        self.generate_vertical_streets(n_streets, street_width)

    def set_exit_space(self, exit_space_height=10):
        """Set an exit space at the top of the grid where agents can move freely.

        :param exit_space_height: Height of the exit space.
        """
        self.grid[0:exit_space_height, :] = True

    def generate_vertical_streets(self, n_streets, street_width):
        """Splits the grid into multiple vertical streets of given width.

        :param n_streets: Number of streets to generate.
        :param street_width: Width of each street.
        """
        if n_streets <= 0:
            raise ValueError(f"Number of streets must be a positive integer. {n_streets} given.")

        if street_width <= 0:
            raise ValueError(f"Street width must be a positive integer. {street_width} given.")

        # Calculate the number of cells needed for the streets
        total_street_width = n_streets * street_width
        if total_street_width > self.width:
            raise ValueError("Total width of streets exceeds grid width.")

        # Calculate the number of columns available for buildings
        available_columns = self.width - total_street_width
        building_width = available_columns // (n_streets + 1)
        if building_width < 1:
            raise ValueError("Building width must be be at least 1.")

        # Create vertical streets
        for i in range(n_streets):
            start_col = building_width * (i + 1) + i * street_width
            end_col = start_col + street_width

            # Create vertical streets with spacing
            for i in range(n_streets):
                start_col = building_width * (i + 1) + i * street_width
                end_col = start_col + street_width
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


# grid = CityMap(100, 200, 5, 10, exit_space_height=10)
# grid.display()
