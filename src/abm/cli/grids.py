"""This module provides the `CityGrid` class for creating and managing a grid-based representations of a city layout.

The grid is implemented as a numpy array where cells with value 1 represent streets (accessible
areas), and cells with a value of 0 represent houses or obstructions (non-accessible areas).

The `CityGrid` class allows for:
- Customizable grid sizes.
- Automatic generation of random layouts based on the number of streets and their widths.
- Predefined layouts resembling real-world city configurations ('grid', 'radial', 'linear').
- Simple text-based visualization of the grid.

This module is designed for use in agent-based modeling and simulation projects.
"""  # noqa: E501

import matplotlib.pyplot as plt
import numpy as np

grid_size = 100


class CityGrid:
    def __init__(self):
        """Initialize the CityGrid."""
        self.size = (grid_size, grid_size)
        self.grid = np.zeros(self.size, dtype=int)

    def set_exits(self, exit_width=10):
        """Set two exits at the top corners of the grid."""
        self.grid[0, :exit_width] = 2
        self.grid[0, -exit_width:] = 2

    def set_exit_space(self, lenght=10):
        """Set an exit space at the top of the grid where agents can move freely.

        :param lenght: Length of the exit space.
        """
        self.grid[1:lenght, :] = 1

    def set_parking(self):
        """Set parking spaces at the bottom of the grid."""
        self.grid[-1, :] = 2

    def generate_random_layout(self, num_streets, street_width):
        """Splits the grid into multiple vertical streets of given width.

        :param num_streets: Number of streets to generate.
        :param street_width: Width of each street.
        """
        if num_streets <= 0 or street_width <= 0:
            raise ValueError("Number of streets and street width must be positive integers.")

        # Calculate the number of columns needed for the streets
        total_street_width = num_streets * street_width
        if total_street_width > self.size[1]:
            raise ValueError("Total width of streets exceeds grid width.")

        # Calculate the number of columns available for buildings
        available_columns = self.size[1] - total_street_width
        gaps_between_streets = available_columns // (num_streets + 1)

        # Create vertical streets
        for i in range(num_streets):
            start_col = i * street_width
            end_col = start_col + street_width
            # Create a large square street at the top of the grid
            start_col = gaps_between_streets * (i + 1) + i * street_width
            end_col = start_col + street_width

            # Calculate spacing between streets
            spacing = (self.size[1] - total_street_width) // (num_streets + 1)

            # Create vertical streets with spacing
            for i in range(num_streets):
                start_col = spacing * (i + 1) + i * street_width
                end_col = start_col + street_width
                self.grid[1:-1, start_col:end_col] = 1

    def display(self):
        """Display the grid as a heatmap using matplotlib."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap="Greys", origin="upper")
        plt.title("City Grid")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.colorbar(label="Cell Value")
        plt.show()


grid = CityGrid()
grid.set_exits()
grid.set_exit_space(lenght=20)
grid.set_parking()
grid.generate_random_layout(num_streets=5, street_width=10)
grid.display()
