from abc import ABC


class Control(ABC):
    """Defines a control imput.
    Subclasses should be immutable
        `@dataclass(frozen=True)`
    """

    ...
