import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from simulation import SimAgent, SimObject
from states import State, discretize_state
from camera import Camera
from util import PriorityQueue
from visualization import DisplayServer, display
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SearchException(Exception):
    """Generic exception for a search"""

    ...


@dataclass(frozen=True, order=True)
class SearchNode:
    data: any = field(compare=False)
    parent: Optional["SearchNode"] = field(default=None, compare=False)
    depth: int = field(init=False, compare=True)

    def __post_init__(self):
        if self.parent is not None:
            # workaround since this class is frozen
            object.__setattr__(self, "depth", self.parent.depth + 1)
        else:
            object.__setattr__(self, "depth", 0)

    def to_list(self) -> list[any]:
        node = self
        data = [self.data]
        while node.parent is not None:
            node = node.parent
            data.append(node.data)
        data.reverse()
        return data


def a_star(
    agent: SimAgent,
    static_objects: list[SimObject],
    target_state: State,
    dt: float,
    state_space_bin_sizes: dict[str, float],
    visualize: int = 0,
    camera: Optional[Camera] = None,
    max_iterations: int = None,
) -> list[SimAgent]:
    """Performs A* for a single agent in a world of static objects.
    kills leaves when a collision is detected.

    Args:
        agent (SimObject): The agent to control
        static_objects (list[SimObject]): The static objects to avoid
        target_state (State): The target state for `agent`
        dt (float): Time period to advance the simulation by
        state_space_bin_sizes (dict[str, float]): dictionary of values to descretize the state space with.
            see `states.discretize_state`
        visualize (int): Allows for vizualizing the search process. visualizes every n nodes explored.
        camera (Optional[Camera]): Specifies a camera to be used in the case of visualization.
        max_iterations (int): Allows for capping the search at a specific number of iterations

    Returns:
        list[SimAgent]: The list of agents at each timestep leading upto the solution
    """
    if visualize:
        display_server = DisplayServer()
        display_server.__enter__()
        display_server.set_camera(camera or Camera())

    # check for initial intersection
    if not agent.is_valid(others=static_objects):
        if visualize:
            display_server.__exit__(None, None, None)
        logging.debug(
            f"Failed a_star serach from {agent.state = } to {target_state = }. Agent intially intersecting obstacles."
        )
        raise SearchException(
            f"Could not find path from {agent.state = } to {target_state = }. Agent intially intersecting obstacles."
        )

    # check for target state intersection
    if not dataclasses.replace(agent, state=target_state).is_valid(
        others=static_objects
    ):
        if visualize:
            display_server.__exit__(None, None, None)
        logging.debug(
            f"Failed a_star serach from {agent.state = } to {target_state = }. Target state intersecting obstacles."
        )
        raise SearchException(
            f"Could not find path from {agent.state = } to {target_state = }. Target state intersecting obstacles."
        )
    explored: set[State] = set()
    """Set of discretized agent states that have already been checked"""
    frontier: PriorityQueue = PriorityQueue()
    """Agent states to check soon"""
    frontier.put(SearchNode(data=agent), 0)

    discrete_target_state = discretize_state(target_state, state_space_bin_sizes)
    _cache_hits = 0

    # visualize the target state
    if visualize:
        target_agent = dataclasses.replace(agent, state=target_state)
        display(
            [[agent] + static_objects + [target_agent]],
            clear_at_start=True,
            clear_between=False,
            display_server=display_server,
        )

    while not frontier.is_empty():
        # make sure we aren't over the iteration limit
        if max_iterations is not None and len(explored) >= max_iterations:
            if visualize:
                display_server.__exit__(None, None, None)
            logging.debug(
                f"Failed a_star serach from {agent.state = } to {target_state = }, max_iterations reached."
            )
            raise SearchException(
                f"Could not find path from {agent.state = } to {target_state = }, max_iterations reached."
            )

        head_node: SearchNode = frontier.get()
        head_agent: SimAgent = head_node.data

        # skip if descrete state was already explored
        descrete_state = discretize_state(head_agent.state, state_space_bin_sizes)
        if descrete_state in explored:
            _cache_hits += 1
            continue
        explored.add(descrete_state)

        # skip if any intersections
        if not head_agent.is_valid(others=static_objects):
            continue

        # visualize every `visualize` number of states
        if visualize and not (len(explored) % visualize):
            display(
                [[head_agent]],
                clear_at_start=False,
                clear_between=False,
                display_server=display_server,
            )

        # sucess contition
        if descrete_state == discrete_target_state:
            if visualize:
                display_server.__exit__(None, None, None)

            logging.debug(
                f"Succeeded a_star serach from {agent.state} to {target_state}."
                f"\n\tSolution length = {head_node.depth}"
                f"\n\tStates explored = {len(explored)}"
                f"\n\tCache hits = {_cache_hits}"
                f"\n\tFrontier size = {len(frontier._elements)}"
            )
            return head_node.to_list()

        for neighbor_control in head_agent.control_neighbors():
            # we specifically want to disallow side effects here
            neighbor_agent = head_agent.simulate(
                dt=dt, control=neighbor_control, side_effects=False
            )

            heuristic: float = neighbor_agent.heuristic(
                others=static_objects,
                prev_agent=head_agent,
                search_depth=head_node.depth,
                target_state=target_state,
            )
            neighbor_node = SearchNode(data=neighbor_agent, parent=head_node)
            frontier.put(neighbor_node, heuristic)

    if visualize:
        display_server.__exit__(None, None, None)
    logging.debug(
        f"Failed a_star serach from {agent.state = } to {target_state = }, frontier empty"
    )
    raise SearchException(
        f"Could not find path from {agent.state = } to {target_state = }, frontier empty."
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
