"""
This file contains the main interface for the search module.
"""

import numpy as np
from holonomic_quad import HoloQuadState, HoloQuad
from heightmap import HeightMap
from states import Position
import dataclasses
from a_star import SearchException, a_star
import shapely


def _interpolate_nan(array: np.ndarray) -> np.ndarray:
    """
    Interpolates to fill any gaps of np.nan in an array

    Args:
        array (np.ndarray): The array to fix up

    Returns:
        np.ndarray: An array with the np.nans interpolated over
    """
    # Find the indices of valid (non-NaN) and invalid (NaN) entries
    indices = np.arange(len(array))
    valid = ~np.isnan(array)
    invalid = np.isnan(array)

    # Perform linear interpolation for the NaN values
    array[invalid] = np.interp(indices[invalid], indices[valid], array[valid])

    return array


def simplify_path(
    states: list[HoloQuadState],
    max_segment_length: float = None,
    tolerance: float = 0.5,
) -> list[Position]:
    """
    Simplifies a set of states

    Args:
        states (list[HoloQuadState]): The set of states to simplify
        max_segment_length (float): Maximum length of a segment, see shapely.LineString.segmentize for specifics.
            This may or may not actually correspond to the euclidiean length of each segment.
        tolerance (float, optional): The tolerance for simplifying, see shapely.simplify for specifics. Defaults to 0.5.

    Returns:
        list[HoloQuadState]: The simplified path
    """
    # here we are treating theta as another linear dimension.
    # this is a little funny, but should avoid simplifying out
    # rapid turns
    positions = [(state.x, state.y, state.theta) for state in states]
    line = shapely.LineString(positions)
    simple_line = shapely.simplify(line, tolerance, preserve_topology=True)

    if max_segment_length is not None:
        # now for some awful reason, this doesn't interpolate the thetas too,
        # they are left as nan, here is the workaround
        simple_line = simple_line.segmentize(max_segment_length)
        coordinates = np.asarray(simple_line.coords)
        print(coordinates)
        coordinates[:, 2] = _interpolate_nan(coordinates[:, 2])
        print(coordinates)

        simple_positions = [Position(*row) for row in coordinates]
    else:
        simple_positions = [Position(*position) for position in simple_line.coords]

    return simple_positions


def search(
    map: np.ndarray,
    map_cell_size: float,
    initial_state: Position,
    final_state: Position,
    dt: float,
    **kwargs,
) -> list[HoloQuadState]:
    """High level interface for searching through the world

    Performs a search from initial_state to final_state through a map.
    The robot is defined by the HoloQuad implementation.

    Raises:
        SearchException: On a failed A* search

    Args:
        map (np.ndarray): 2d array of the map with height values (m).
            Map[0,0] placed at (0,0) in space, directed in +x +y.
            `impassible` cells should have +infinite value
        map_cell_size (float): size of each cell in the map (m).
            cells are assumed to be square.
        initial_state (Position): Initial state in world space.
        final_state (Position): Final state in world space
        dt (float): dt.
        **kwargs (any): passed to a_star

    Returns:
        list[HoloQuadState]: a list of states leading to the solution (assuming sucess)
    """
    state_space_bin_sizes = {
        "x_velocity": 0.2,
        "y_velocity": 0.2,
        "omega": 0.05,
        "x": 0.3,
        "y": 0.3,
        "theta": 0.1,
    }
    heightmap = HeightMap.default_from_heightmap(height_map=map, scale=map_cell_size)
    initial_state = HoloQuadState(**dataclasses.asdict(initial_state))
    final_state = HoloQuadState(**dataclasses.asdict(final_state))
    search_agent = HoloQuad(length=2.5, width=1.5, state=initial_state)

    result: list[HoloQuad] = a_star(
        agent=search_agent,
        static_objects=[heightmap],
        target_state=final_state,
        dt=dt,
        state_space_bin_sizes=state_space_bin_sizes,
        **kwargs,
    )

    states = [agent.state for agent in result]
    return states


if __name__ == "__main__":
    from visualization import Camera
    from visualization import DisplayServer
    from a_star import SearchConditionException
    import time

    camera = Camera(-3, 0, 30)

    while True:
        try:
            map = np.random.rand(10, 10) - 0.5
            # half at 0 height
            map[map < 0] = 0
            # 10% impassable
            map[map > 0.40] = np.inf
            map *= 0.1

            initial_state = Position(x=1, y=1, theta=0)
            final_state = Position(x=15, y=15, theta=0)
            states = search(
                map=map,
                map_cell_size=2,
                initial_state=initial_state,
                final_state=final_state,
                dt=0.5,
                visualize=100,
                camera=camera,
            )
            break
        except SearchConditionException:
            # try again, map was generated invalid
            ...

    heightmap = HeightMap.default_from_heightmap(height_map=map, scale=2)

    simple_states = simplify_path(states, max_segment_length=4)
    print(f"{len(states) = }")
    print(f"{len(simple_states) = }")

    with DisplayServer() as display_server:
        for state in states:
            dog = HoloQuad(length=1.5, width=0.75, state=state)
            target_dog = HoloQuad(
                length=1.5,
                width=0.75,
                state=HoloQuadState(**dataclasses.asdict(final_state)),
            )
            display_server.set_camera(camera)
            display_server.display([heightmap, dog, target_dog])
            time.sleep(1 / 20)
        for state in simple_states:
            dog = HoloQuad(length=1.5, width=0.75, state=state)
            target_dog = HoloQuad(
                length=1.5,
                width=0.75,
                state=HoloQuadState(**dataclasses.asdict(final_state)),
            )
            display_server.set_camera(camera)
            display_server.display([heightmap, dog, target_dog])
            time.sleep(1)
