from abc import ABC
from dataclasses import dataclass, fields


class State(ABC):
    """Defines the state of a simulated agent
    Subclasses should be immutable
        `@dataclass(frozen=True)`
    """

    ...


@dataclass(frozen=True)
class Point(State):
    """Positional state, in world space

    Args:
        x (float, Optional): X position. Defaults to 0
        y (float, Optional): Y position. Defaults to 0
    """

    x: float = 0
    """X position"""
    y: float = 0
    """Y position"""


@dataclass(frozen=True)
class Position(Point):
    """Positional state, in world space

    Args:
        x (float, Optional): X position. Defaults to 0
        y (float, Optional): Y position. Defaults to 0
        theta (float, Optional): Theta value in radians. Defaults to 0
    """

    theta: float = 0
    """Theta value in radians"""


def discretize_state(state: State, bin_sizes: dict[str, float | None]) -> State:
    """Discretizes any continuous state into bins based on bin sizes.

    Args:
        state (State): The state to discretize.
        bin_sizes (dict[str, float]): A dictionary mapping field names to bin sizes. None means don't discretize

    Returns:
        State: The same class of state with descretized values.

    Examples:
        >>> dds = DiffDriveState(x=0.1, y=0.4, theta=1.74, v=0.9)
        >>> bin_sizes = {"x": 0.5, "y": 0.5, "theta": 0.5, "v": 0.5}
        >>> discretize_state(state=dds, bin_sizes=bin_sizes)
        DiffDriveState(x=0.0, y=0.5, theta=1.5, v=1.0)

        >>> bin_sizes = {"x": 0.5, "y": 0.5, "theta": 0.5}
        >>> discretize_state(state=dds, bin_sizes=bin_sizes)
        Traceback (most recent call last):
        ...
        Exception: Missing bin size for 'v'. Please include in `bin_sizes` dict.
    """
    discretized_values = {}

    # Loop through each field in the state dataclass
    for field in fields(state):
        if not field.init:
            continue
        field_name = field.name
        field_value = getattr(state, field_name)
        # bin_size = bin_sizes.get(field_name, 1.0)  # Default bin size if not specified
        try:
            bin_size = bin_sizes[field_name]  # Default bin size if not specified
        except KeyError as e:
            raise Exception(
                f"Missing bin size for {e}. Please include in `bin_sizes` dict."
            )

        # Discretize the field
        if bin_size is None:
            rounded_field_val = field_value
        else:
            rounded_field_val = round(field_value / bin_size) * bin_size
        discretized_values[field_name] = rounded_field_val
    return type(state)(**discretized_values)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
