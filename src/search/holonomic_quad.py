import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
import pyglet
import shapely

from control import Control
from simulation import SimAgent, SimObject
from states import Position
from util import transform_2d

_max_acceleration: float = 20
"""Maximum acceleration control imput. Should be positive"""
_max_deceleration: float = -20
"""Maximum deceleration control imput. Should be negative"""


@dataclass(frozen=True)
class DiffDriveControl(Control):
    """Control input for a differential drive robot

    Args:
        left_wheel_acc (float): acceleration of the left wheel
        right_wheel_acc (float): acceleration of the right wheel
    """

    linear_acceleration: np.ndarray[float]
    """linear acceleration vector (2,). units/s"""
    angular_acceleration: float
    """angular acceleration in z direction, following right hand rule. radians/s"""


@dataclass(frozen=True)
class DiffDriveState(Position):
    """State of a differential drive robot

    Args:
        x (float, Optional): X position of the robot. Defaults to 0
        y (float, Optional): Y position of the robot. Defaults to 0
        theta (float, Optional): Theta of the robot in radians. Defaults to 0
        vl (float, Optional): Velocity of left wheel of the robot. Defaults to 0
        vr (float, Optional): Velocity of right wheel of the robot. Defaults to 0
    """

    vl: float = 0
    """Velocity of left wheel of the robot"""
    vr: float = 0
    """Velocity of right wheel of the robot"""
    v: float = field(init=False)
    """Velocity of the robot"""

    def __post_init__(self):
        # workaround since this class is frozen
        object.__setattr__(self, "v", (self.vl + self.vr) / 2)


@dataclass(frozen=True)
class DiffDrive(SimAgent):
    """Simulation object for a differential drive agent

    Args:
        height (float): Height of vehicle (y direction)
        width (float): Width of vehicle (x direction)
        wheel_base_width (float): Width of the wheel base (x direction)
        wheel_base_offset (float): y direction offset of center of rotation from geometric center (y direction)
        color (tuple[int, int, int, int] | tuple[int, int, int]): Color of rectangle. Defaults to Black
        state (Position): State
    """

    height: float
    """Height of vehicle (y direction)"""
    width: float
    """Width of vehicle (x direction)"""
    wheel_base_width: float
    """Width of the wheel base (x direction)"""
    # TODO this still doesn't work correctly
    wheel_base_offset: float
    """y direction offset of center of rotation from geometric center (y direction).
        changes how the car is drawn
    """
    color: tuple[int, int, int, int] | tuple[int, int, int] = (255, 255, 255)
    """Color of rectangle. Defaults to Black"""
    state: DiffDriveState = DiffDriveState()
    """State"""
    # Normally this would require a default factory, but
    # DiffDriveControl is static so it doesn't matter
    previous_control: Optional[DiffDriveControl] = None
    """Most recently applied control input. None implies no previous control input"""

    @cached_property
    def collision_polygons(self) -> list[shapely.geometry.Polygon]:
        verts = [
            [self.width / 2, self.height / 2, 1],
            [-self.width / 2, self.height / 2, 1],
            [-self.width / 2, -self.height / 2, 1],
            [self.width / 2, -self.height / 2, 1],
            [self.width / 2, self.height / 2, 1],
        ]
        transformed_verts = transform_2d(
            verts=verts,
            x=self.state.x,
            y=self.state.y - self.wheel_base_offset,
            theta=self.state.theta,
        )
        return [shapely.geometry.Polygon(transformed_verts)]

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
            self.height / 2 + self.wheel_base_offset,
        )
        # for some reason this uses degrees and goes clockwise
        sprite.rotation = -math.degrees(self.state.theta)
        return [sprite]

    def simulate(
        self, dt: float, control: DiffDriveControl, side_effects: float = True
    ) -> "DiffDrive":
        vl = self.state.vl + dt * control.left_wheel_acc
        vr = self.state.vr + dt * control.right_wheel_acc
        v = (vl + vr) / 2

        thetadot = (vr - vl) / self.wheel_base_width
        theta = self.state.theta + dt * thetadot

        xdot = v * math.sin(-theta)
        x = self.state.x + dt * xdot
        ydot = v * math.cos(-theta)
        y = self.state.y + dt * ydot

        new_state = dataclasses.replace(self.state, x=x, y=y, theta=theta, vl=vl, vr=vr)
        return dataclasses.replace(self, state=new_state, previous_control=control)

    def heuristic(
        self,
        others: list[SimObject],
        prev_agent: "DiffDrive",
        search_depth: int,
        target_state: DiffDriveState,
    ) -> float:
        heuristic = 0
        # Heuristic is primarily distance
        agent_pos = np.asarray([self.state.x, self.state.y])
        target_pos = np.asarray([target_state.x, target_state.y])
        offset_vector = target_pos - agent_pos
        distance = np.linalg.norm(np.abs(offset_vector), ord=2)
        heuristic += distance

        # add a little of breath search preference
        heuristic += 0.1 * search_depth

        # penalize going too fast when near the target
        velocity_l_contribution = max(
            0,
            (self.state.vl - target_state.vl) ** 2 / (-2 * _max_deceleration)
            - distance,
        )
        heuristic += 2 * velocity_l_contribution
        velocity_r_contribution = max(
            0,
            (self.state.vr - target_state.vr) ** 2 / (-2 * _max_deceleration)
            - distance,
        )
        heuristic += 2 * velocity_r_contribution

        # encourage the agent to align with the target angle
        # when close
        theta_contribution = abs(self.state.theta - target_state.theta) / (distance + 2)
        heuristic += 10 * theta_contribution

        return heuristic

    def control_neighbors(self) -> list[DiffDriveControl]:
        num_acceleration_vals = 3
        accelerations = np.linspace(
            _max_deceleration, _max_acceleration, num_acceleration_vals
        )
        return [
            DiffDriveControl(left_wheel_acc=lacc, right_wheel_acc=racc)
            for lacc, racc in itertools.product(accelerations, accelerations)
        ]
