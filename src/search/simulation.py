from itertools import combinations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import pyglet
import shapely

from control import Control
from states import Position, State


class SimulationException(Exception):
    """Generic exception for a simulation"""

    ...


class Drawable(ABC):
    """ABC for things that get drawn"""

    @abstractmethod
    def drawables(
        self, batch: pyglet.graphics.Batch
    ) -> list[pyglet.shapes.ShapeBase | pyglet.sprite.Sprite]:
        """pyglet.shapes object to be used for rendering.
        Be careful if trying to cache. Pyglet objects must be created
        after the window.
        """
        ...


class SimObject(Drawable):
    """ABC for simulation object.
    Subclasses should probably be immutable
        `@dataclass(frozen=True)`
    """

    @property
    @abstractmethod
    def state(self) -> State:
        """The internal state of the object"""
        ...

    @property
    @abstractmethod
    def collision_polygons(self) -> list[shapely.geometry.Polygon]:
        """Collision polygon of the object"""
        ...


class SimAgent(SimObject):
    """ABC for simulation agent.
    Subclasses should probably be immutable
        `@dataclass(frozen=True)`
    """

    @property
    @abstractmethod
    def _previous_control(self) -> Optional[Control]:
        """Most recently applied control input"""
        ...

    @abstractmethod
    def simulate(
        self, dt: float, control: Optional[Control] = None, side_effects: float = True
    ) -> "SimAgent":
        """Simulates a time step (dt) based on a state and control inputs

        Args:
            dt (float): Time passed
            control (Optional[Control]): Control input, or None if the object doesn't need control.
                Defaults to None
            side_effects (float): Allows for preventing side effects. Useful for agents who
                have mutable attributes that they modify, that we don't want to change for searches

        Returns:
            SimAgent: Copy of self with updated state
        """
        ...

    @abstractmethod
    def control_neighbors(self) -> list[Control]:
        """Gets a feasable set of next control inputs."""
        ...

    @abstractmethod
    def heuristic(
        self,
        others: list[SimObject],
        prev_agent: "SimAgent",
        search_depth: int,
        target_state: State,
    ) -> float:
        """Heuristic function"""
        ...

    def is_valid(self, others: list[SimObject]) -> bool:
        """Checks if the state of the agent is valid.
        default implementation is just collision check

        Args:
            others (list[SimObject]): other agents in the simulation

        Returns:
            bool: if the agent state is valid
        """
        return not any_intersects(self, others=others)


@dataclass(frozen=True)
class StaticRect(SimObject):
    """Simulation object for a static rectangle. Just serves to collide with other things

    Args:
        height (float): Height of rectangle (y direction)
        width (float): Width of rectangle (x direction)
        color (str): Color of rectangle. Defaults to Black
        state (Position): State
    """

    height: float
    """Height of rectangle (y direction)"""
    width: float
    """Width of rectangle (x direction)"""
    color: tuple[int, int, int, int] | tuple[int, int, int] = (255, 255, 255)
    """Color of rectangle. Defaults to Black"""
    state: Position = Position()
    """State"""

    @cached_property
    def collision_polygons(self) -> list[shapely.geometry.Polygon]:
        # [:, np.newaxis] makes them column vectors
        verts = [
            np.asarray((self.width / 2, self.height / 2, 1))[:, np.newaxis],
            np.asarray((-self.width / 2, self.height / 2, 1))[:, np.newaxis],
            np.asarray((-self.width / 2, -self.height / 2, 1))[:, np.newaxis],
            np.asarray((self.width / 2, -self.height / 2, 1))[:, np.newaxis],
            np.asarray((self.width / 2, self.height / 2, 1))[:, np.newaxis],
        ]
        t = self.state.theta
        transform = np.asarray(
            (
                (math.cos(t), -math.sin(t), self.state.x),
                (math.sin(t), math.cos(t), self.state.y),
                (0, 0, 1),
            )
        )
        # apply transform, drop last element
        verts = [((transform @ vert).T)[0, :-1] for vert in verts]
        return [shapely.geometry.Polygon(verts)]

    def drawables(
        self, batch: pyglet.graphics.Batch
    ) -> list[pyglet.shapes.ShapeBase | pyglet.sprite.Sprite]:
        sprite = pyglet.shapes.Rectangle(
            x=self.state.x,
            y=self.state.y,
            width=self.width,
            height=self.height,
            color=self.color,
            batch=batch,
        )
        sprite.anchor_position = (
            self.width / 2,
            self.height / 2,
        )
        # for some reason this uses degrees
        sprite.rotation = math.degrees(self.state.theta)
        return [sprite]


def any_intersects(
    agent: SimObject,
    others: list[SimObject],
    check_self_intersection: bool = True,
) -> bool:
    """Checks if `agent` intersects with any of the `static_objects`

    Args:
        agent (SimObject): Agent to check for intersections with
        others (list[SimObject]): Other objects
        check_self_intersections (bool): Whether or not to check for self intersection

    Returns:
        bool: If there is an intersection between the `agent` and any `static_objects`
    """
    # TODO: cache results where applicable

    # self intersections
    if check_self_intersection:
        for a, b in combinations(agent.collision_polygons, 2):
            if a.intersects(b):
                return True

    # external intersections
    for obj in others:
        for obj_part in obj.collision_polygons:
            for agent_part in agent.collision_polygons:
                if agent_part.intersects(obj_part):
                    return True

    return False
