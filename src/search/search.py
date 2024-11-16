"""
This file contains the main interface for the search module.
"""

import numpy as np
from holonomic_quad import HoloQuadState
from states import Position
from dataclasses import dataclass


@dataclass
class SearchResult:
    sucess: bool
    result: list[HoloQuadState] | None


example_search_output = SearchResult(
    True,
    [
        HoloQuadState(x=0, y=0, theta=0, x_velocity=0, y_velocity=0, omega=0),
        HoloQuadState(x=1, y=1, theta=-np.pi / 4, x_velocity=0, y_velocity=0, omega=0),
        HoloQuadState(x=1, y=3, theta=-np.pi / 2, x_velocity=0, y_velocity=0, omega=0),
        HoloQuadState(x=3, y=3, theta=0, x_velocity=0, y_velocity=0, omega=0),
    ],
)
"""
Example of an output from the search function. There probably would be 
non-zero values for x_velocity, y_velocity, and omega, but I expect us to throw
away the values.

Additionally, search will output a waypoint every dt seconds. We will probably want
to throw away some number of these.
"""


def search(
    map: np.ndarray,
    map_cell_size: float,
    initial_state: Position,
    final_state: Position,
    dt: float,
) -> SearchResult:
    """High level interface for searching through the world

    Performs a search from initial_state to final_state through a map.
    The robot is defined by the HoloQuad implementation.

    Args:
        map (np.ndarray): 2d array of the map with height values (m).
            Map[0,0] placed at (0,0) in space, directed in +x +y.
            `impassible` cells should have +infinite value
        map_cell_size (float): size of each cell in the map (m).
            cells are assumed to be square.
        initial_state (Position): Initial state in world space.
        final_state (Position): Final state in world space
        dt (float): dt.

    Returns:
        SearchResult: Boolean indicating sucess, and
            a list of states leading to the solution (assuming sucess)
    """
    raise NotImplementedError()
