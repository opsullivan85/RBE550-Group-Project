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
from typing import Optional


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
    max_segment_length: Optional[float] = None,
    tolerance: float = 0.5,
    turning_weight: float = 1.0,
) -> list[Position]:
    """
    Simplifies a set of states

    Args:
        states (list[HoloQuadState]): The set of states to simplify
        max_segment_length (float): Maximum length of a segment, see shapely.LineString.segmentize for specifics.
            This may or may not actually correspond to the euclidiean length of each segment.
        tolerance (float, optional): The tolerance for simplifying, see shapely.simplify for specifics. Defaults to 0.5.
        turning_weight (float, optional): Added weight to turning, changes how strictly the simplification
            respects changes in angle. Defaults to 1.

    Returns:
        list[HoloQuadState]: The simplified path
    """
    # here we are treating theta as another linear dimension.
    # this is a little funny, but should avoid simplifying out
    # rapid turns
    positions = [(state.x, state.y, state.theta * turning_weight) for state in states]
    line = shapely.LineString(positions)
    simple_line = shapely.simplify(line, tolerance, preserve_topology=True)

    if max_segment_length is not None:
        # now for some awful reason, this doesn't interpolate the thetas too,
        # they are left as nan, here is the workaround
        simple_line = simple_line.segmentize(max_segment_length)
        coordinates = np.asarray(simple_line.coords)
        coordinates[:, 2] = _interpolate_nan(coordinates[:, 2])

        simple_positions = [
            Position(row[0], row[1], row[2] / turning_weight) for row in coordinates
        ]
    else:
        simple_positions = [
            Position(position[0], position[1], position[2] / turning_weight)
            for position in simple_line.coords
        ]

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
            Any cells taller than 0.5 are taken to be impassable
        map_cell_size (float): size of each cell in the map (m).
            cells are assumed to be square.
        initial_state (Position): Initial state in world space.
        final_state (Position): Final state in world space
        dt (float): dt.
        **kwargs (any): passed to a_star

    Returns:
        list[HoloQuadState]: a list of states leading to the solution (assuming sucess)
    """
    map[map > 0.5] = np.inf
    # Somewhere in this pipeline (or on the other side) X and Y are getting swapped.
    # this should fix it
    map = map.T
    state_space_bin_sizes = {
        "x_velocity": 0.4,
        "y_velocity": 0.4,
        "omega": 0.1,
        "x": 0.17,
        "y": 0.17,
        "theta": 0.1,
    }
    heightmap = HeightMap.default_from_heightmap(height_map=map, scale=map_cell_size)
    initial_state = HoloQuadState(**dataclasses.asdict(initial_state))
    final_state = HoloQuadState(**dataclasses.asdict(final_state))
    search_agent = HoloQuad(length=1, width=0.5, state=initial_state)

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

    camera = Camera(0, 0, 120)

    while True:
        try:
            # map = np.random.rand(10, 10) - 0.5
            # # half at 0 height
            # map[map < 0] = 0
            # # 10% impassable
            # map[map > 0.40] = np.inf
            # map *= 0.1

            # hard coding in the default map for testing
            map = eval(
                """np.asarray([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.08323599,
        0.08323599, 0.        , 0.        , 0.        , 0.        ,
        0.06506913, 0.06506913, 0.        , 0.        , 0.        ,
        0.05816431, 0.05816431, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.08323599,
        0.08323599, 0.        , 0.        , 0.14749025, 0.14749025,
        0.06506913, 0.06506913, 0.        , 0.        , 0.        ,
        0.05816431, 0.05816431, 0.        , 0.15594869, 0.15594869,
        0.09900517, 0.09900517, 0.11811679, 0.11811679],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.14749025, 0.14749025,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.15594869, 0.15594869,
        0.09900517, 0.09900517, 0.11811679, 0.11811679],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.09908702, 0.09908702,
        0.05824306, 0.05824306, 0.        , 0.13210194, 0.13210194,
        0.        , 0.15568852, 0.15568852, 0.        , 0.        ,
        0.03718125, 0.03718125, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.09908702, 0.09908702,
        0.05824306, 0.05824306, 0.        , 0.13210194, 0.13210194,
        0.        , 0.15568852, 0.15568852, 0.        , 0.        ,
        0.03718125, 0.03718125, 0.        , 0.        , 0.        ,
        0.75      , 0.75      , 0.75      , 0.75      ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.10044771, 0.10044771, 0.        , 0.        , 0.        ,
        0.05159382, 0.05159382, 0.18201683, 0.18201683, 0.        ,
        0.        , 0.1313313 , 0.1313313 , 0.17220177, 0.17220177,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.75      , 0.75      , 0.75      , 0.75      ],
       [0.        , 0.        , 0.        , 0.11467654, 0.11467654,
        0.10044771, 0.10044771, 0.        , 0.        , 0.        ,
        0.05159382, 0.05159382, 0.18201683, 0.18201683, 0.        ,
        0.        , 0.1313313 , 0.1313313 , 0.17220177, 0.17220177,
        0.        , 0.75      , 0.75      , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.75      , 0.75      ],
       [0.        , 0.        , 0.        , 0.11467654, 0.11467654,
        0.        , 0.        , 0.        , 0.        , 0.06686668,
        0.06686668, 0.13536971, 0.13536971, 0.        , 0.        ,
        0.        , 0.        , 0.75      , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.75      , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.75      , 0.75      ],
       [0.        , 0.        , 0.        , 0.        , 0.19664341,
        0.19664341, 0.13489101, 0.13489101, 0.        , 0.06686668,
        0.06686668, 0.13536971, 0.13536971, 0.        , 0.14430801,
        0.14430801, 0.        , 0.75      , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.75      , 0.75      , 0.75      ,
        0.1256897 , 0.1256897 , 0.        , 0.        ],
       [0.        , 0.14067642, 0.14067642, 0.        , 0.19664341,
        0.19664341, 0.13489101, 0.13489101, 0.        , 0.        ,
        0.        , 0.01390308, 0.01390308, 0.        , 0.14430801,
        0.14430801, 0.        , 0.75      , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.75      , 0.75      , 0.75      ,
        0.1256897 , 0.1256897 , 0.        , 0.        ],
       [0.        , 0.14067642, 0.14067642, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.01390308, 0.01390308, 0.        , 0.        ,
        0.        , 0.        , 0.75      , 0.75      , 0.75      ,
        0.75      , 0.        , 0.        , 0.        , 0.        ,
        0.75      , 0.75      , 0.75      , 0.75      ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.10975975, 0.10975975, 0.        , 0.17815363, 0.17815363,
        0.        , 0.        , 0.0979387 , 0.0979387 , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.75      , 0.75      , 0.75      , 0.75      ],
       [0.75      , 0.75      , 0.75      , 0.75      , 0.        ,
        0.10975975, 0.10975975, 0.        , 0.17815363, 0.17815363,
        0.        , 0.        , 0.0979387 , 0.0979387 , 0.06265617,
        0.06265617, 0.11780045, 0.11780045, 0.        , 0.        ,
        0.15774467, 0.15774467, 0.        , 0.        , 0.        ,
        0.75      , 0.75      , 0.75      , 0.75      ],
       [0.75      , 0.75      , 0.75      , 0.75      , 0.        ,
        0.        , 0.14517052, 0.14517052, 0.        , 0.        ,
        0.        , 0.06921558, 0.06921558, 0.        , 0.06265617,
        0.06265617, 0.11780045, 0.11780045, 0.03240397, 0.03240397,
        0.15774467, 0.15774467, 0.        , 0.        , 0.        ,
        0.75      , 0.75      , 0.75      , 0.75      ],
       [0.75      , 0.75      , 0.75      , 0.75      , 0.        ,
        0.        , 0.14517052, 0.14517052, 0.        , 0.14052511,
        0.14052511, 0.06921558, 0.06921558, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.03240397, 0.03240397,
        0.        , 0.        , 0.        , 0.        , 0.1878334 ,
        0.1878334 , 0.        , 0.        , 0.        ],
       [0.75      , 0.75      , 0.75      , 0.75      , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.14052511,
        0.14052511, 0.        , 0.        , 0.        , 0.        ,
        0.1725298 , 0.1725298 , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.1878334 ,
        0.1878334 , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.18984559, 0.18984559, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.1725298 , 0.1725298 , 0.        , 0.        , 0.16822644,
        0.16822644, 0.        , 0.        , 0.19534766, 0.19534766,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.18984559, 0.18984559, 0.        ,
        0.04259364, 0.04259364, 0.        , 0.        , 0.05068203,
        0.05068203, 0.        , 0.        , 0.03157939, 0.03157939,
        0.        , 0.        , 0.        , 0.        , 0.16822644,
        0.16822644, 0.        , 0.        , 0.19534766, 0.19534766,
        0.        , 0.        , 0.09066194, 0.09066194],
       [0.        , 0.        , 0.11005495, 0.11005495, 0.        ,
        0.04259364, 0.04259364, 0.        , 0.        , 0.05068203,
        0.05068203, 0.        , 0.        , 0.03157939, 0.03157939,
        0.        , 0.        , 0.        , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.09066194, 0.09066194],
       [0.        , 0.        , 0.11005495, 0.11005495, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.1939593 , 0.1939593 , 0.        , 0.15245602, 0.15245602,
        0.        , 0.        , 0.        , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.1939593 , 0.1939593 , 0.        , 0.15245602, 0.15245602,
        0.        , 0.        , 0.        , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.        , 0.05504507, 0.05504507,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.19267718, 0.19267718, 0.        ,
        0.        , 0.        , 0.        , 0.10959926, 0.10959926,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.0711654 , 0.0711654 , 0.        , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.        , 0.05504507, 0.05504507,
        0.        , 0.10686601, 0.10686601, 0.        ],
       [0.        , 0.        , 0.19267718, 0.19267718, 0.        ,
        0.        , 0.        , 0.        , 0.10959926, 0.10959926,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.0711654 , 0.0711654 , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.10015994, 0.10015994,
        0.        , 0.10686601, 0.10686601, 0.        ],
       [0.        , 0.        , 0.        , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.15948085, 0.15948085, 0.03164148,
        0.03164148, 0.        , 0.11187482, 0.11187482, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.0749406 , 0.0749406 , 0.10015994, 0.10015994,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.15948085, 0.15948085, 0.03164148,
        0.03164148, 0.        , 0.11187482, 0.11187482, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.0749406 , 0.0749406 , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.04592101, 0.04592101, 0.        , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.19405281, 0.19405281, 0.        ,
        0.        , 0.02584989, 0.02584989, 0.        , 0.12249049,
        0.12249049, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.04592101, 0.04592101, 0.        , 0.75      , 0.75      ,
        0.75      , 0.75      , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.19405281, 0.19405281, 0.        ,
        0.        , 0.02584989, 0.02584989, 0.        , 0.12249049,
        0.12249049, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.01528923, 0.01528923, 0.        ,
        0.        , 0.06554814, 0.06554814, 0.        , 0.11046357,
        0.11046357, 0.11555896, 0.11555896, 0.        , 0.        ,
        0.07294721, 0.07294721, 0.01418327, 0.01418327, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.01528923, 0.01528923, 0.        ,
        0.        , 0.06554814, 0.06554814, 0.        , 0.11046357,
        0.11046357, 0.11555896, 0.11555896, 0.        , 0.        ,
        0.07294721, 0.07294721, 0.01418327, 0.01418327, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ]])"""
            )

            initial_state = Position(x=0, y=0, theta=0)
            final_state = Position(x=4, y=4, theta=0)
            states = search(
                map=map,
                map_cell_size=0.17,
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

    heightmap = HeightMap.default_from_heightmap(height_map=map.T, scale=0.17)

    simple_states = simplify_path(
        states, max_segment_length=0.8, tolerance=0.17, turning_weight=0.25
    )
    print(f"{len(states) = }")
    print(f"{len(simple_states) = }")

    with DisplayServer() as display_server:
        for state in states:
            dog = HoloQuad(length=1, width=0.5, state=state)
            target_dog = HoloQuad(
                length=1,
                width=0.5,
                state=HoloQuadState(**dataclasses.asdict(final_state)),
            )
            display_server.set_camera(camera)
            display_server.display([heightmap, dog, target_dog])
            time.sleep(1 / 20)
        for state in simple_states:
            dog = HoloQuad(length=1, width=0.5, state=state)
            target_dog = HoloQuad(
                length=1,
                width=0.5,
                state=HoloQuadState(**dataclasses.asdict(final_state)),
            )
            display_server.set_camera(camera)
            display_server.display([heightmap, dog, target_dog])
            time.sleep(1)
