"""Entry point of the utils directory in CLI."""

import numpy as np
import numpy.typing as npt
import scipy


def main(alma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Example function to test mypy static type checking."""
    return alma + 1


if __name__ == "__main__":
    main("alma")
