"""
This file contains the main interface for the search module.
"""

import numpy as np
from holonomic_quad import HoloQuadState, HoloQuad
from heightmap import HeightMap
from states import Position
import dataclasses
from a_star import SearchException, a_star


@dataclasses.dataclass
class SearchResult:
    sucess: bool
    failure_msg: str | None = None
    result: list[HoloQuadState] | None = None


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
    **kwargs,
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
        **kwargs (any): passed to a_star

    Returns:
        SearchResult: Boolean indicating sucess, and
            a list of states leading to the solution (assuming sucess)
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
    try:
        result: list[HoloQuad] = a_star(
            agent=search_agent,
            static_objects=[heightmap],
            target_state=final_state,
            dt=dt,
            state_space_bin_sizes=state_space_bin_sizes,
            **kwargs,
        )
    except SearchException as e:
        return SearchResult(False, failure_msg=str(e))

    states = [agent.state for agent in result]
    return SearchResult(True, result=states)


if __name__ == "__main__":
    from visualization import Camera
    from visualization import DisplayServer
    from a_star import SearchConditionException
    import time

    camera = Camera(-3, 0, 30)

    map = np.random.rand(10, 10) - 0.5
    # half at 0 height
    map[map < 0] = 0
    # 10% impassable
    map[map > 0.40] = np.inf
    map *= 0.1

    initial_state = Position(x=1, y=1, theta=0)
    final_state = Position(x=15, y=15, theta=np.pi / 2)
    ret = search(
        map=map,
        map_cell_size=2,
        initial_state=initial_state,
        final_state=final_state,
        dt=0.5,
        visualize=100,
        camera=camera,
    )

    if not ret.sucess:
        raise Exception(ret.failure_msg)

    states = ret.result
    heightmap = HeightMap.default_from_heightmap(height_map=map, scale=2)
    with DisplayServer() as display_server:
        while states:
            dog = HoloQuad(length=1.5, width=0.75, state=states.pop(0))
            target_dog = HoloQuad(
                length=1.5,
                width=0.75,
                state=HoloQuadState(**dataclasses.asdict(final_state)),
            )
            display_server.set_camera(camera)
            display_server.display([heightmap, dog, target_dog])
            time.sleep(1 / 20)
