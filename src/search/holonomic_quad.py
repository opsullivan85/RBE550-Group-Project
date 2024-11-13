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
from camera import Camera
from simulation import SimAgent, SimObject
from states import Position
from util import transform_2d, unit_vectors_2d


@dataclass(frozen=True)
class HoloQuadControl(Control):
    """Control input for a quadruped robot

    Args:
        x_acceleration (float): Linear acceleration in X. units/s^2
        y_acceleration (float): Linear acceleration in y. units/s^2
        angular_acceleration (float): Angular acceleration in Z,
            following right hand rule. rad/s
    """

    x_acceleration: float = 0
    """Linear acceleration in X. units/s^2"""
    y_acceleration: float = 0
    """Linear acceleration in Y. units/s^2"""
    angular_acceleration: float = 0
    """Angular acceleration in Z, following right hand rule. radians/s^2"""


@dataclass(frozen=True)
class HoloQuadState(Position):
    """State of a quadruped robot

    Args:
        x (float, Optional): X position of the robot. Defaults to 0
        y (float, Optional): Y position of the robot. Defaults to 0
        theta (float, Optional): Theta of the robot in rad. Defaults to 0
        x_velocity (float): Velocity in X. units/s
        y_velocity (float): Velocity in Y. units/s
        omega (float): Angular velocity in z direction,
            following right hand rule. rad/s
    """

    x_velocity: float = 0
    """Velocity in X. units/s"""
    y_velocity: float = 0
    """Velocity in Y. units/s"""
    omega: float = 0
    """Angular velocity in z direction, following right hand rule. radians/s"""


@dataclass(frozen=True)
class HoloQuad(SimAgent):
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
    """Height of robot (y direction)"""
    width: float
    """Width of robot (x direction)"""
    color: tuple[int, int, int, int] | tuple[int, int, int] = (255, 255, 255)
    """Color of rectangle. Defaults to Black"""
    state: HoloQuadState = HoloQuadState()
    """State"""
    _previous_control: Optional[HoloQuadControl] = None
    """Most recently applied control input. None implies no previous control input"""
    max_linear_acceleration: float = 1
    """The maximum linear acceleration. Symetric acceleration and deceleration is assumed. units/s^2"""
    max_angular_acceleration: float = 0.2
    """The maximum angular acceleration. Symetric acceleration and deceleration is assumed. rad/2^2"""
    max_v: float = 1
    """The maximum linear velocity. units/s"""
    max_omega: float = 0.2
    """The maximum angular velocity. rad/2^2"""

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
            y=self.state.y,
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
            self.height / 2,
        )
        # for some reason this uses degrees and goes clockwise
        sprite.rotation = -math.degrees(self.state.theta)
        return [sprite]

    def simulate(
        self, dt: float, control: HoloQuadControl, side_effects: float = True
    ) -> "HoloQuadControl":
        new_x_velocity = self.state.x_velocity + dt * control.x_acceleration
        new_y_velocity = self.state.y_velocity + dt * control.y_acceleration
        new_omega = self.state.omega + dt * control.angular_acceleration

        # Cap velocity and omega
        velocity_magnitude = np.hypot(new_x_velocity, new_y_velocity)
        if velocity_magnitude > self.max_v:
            scale = self.max_v / velocity_magnitude
            new_x_velocity *= scale
            new_y_velocity *= scale
        new_omega = min(max(-self.max_omega, new_omega), self.max_omega)

        new_x = self.state.x + dt * new_x_velocity
        new_y = self.state.y + dt * new_y_velocity
        new_theta = self.state.theta + dt * new_omega
        new_theta = new_theta % np.pi

        new_state = dataclasses.replace(
            self.state,
            x=new_x,
            y=new_y,
            theta=new_theta,
            x_velocity=new_x_velocity,
            y_velocity=new_y_velocity,
            omega=new_omega,
        )

        return dataclasses.replace(self, state=new_state, _previous_control=control)

    def heuristic(
        self,
        others: list[SimObject],
        prev_agent: "HoloQuad",
        search_depth: int,
        target_state: HoloQuadState,
    ) -> float:
        # Heuristic is primarily distance
        agent_pos = np.asarray([self.state.x, self.state.y])
        target_pos = np.asarray([target_state.x, target_state.y])
        offset_vector = target_pos - agent_pos
        distance = np.linalg.norm(np.abs(offset_vector), ord=2)

        # penalize going too fast when near the target
        # equation is specifically derived from kinematic equations
        # ensures that a full deceleration will ensure 0 velocity
        # by the time that we reach the target (assuming a direct path)
        x_velocity_contribution = max(
            0,
            (self.state.x_velocity - target_state.x_velocity) ** 2
            / (-2 * (-self.max_linear_acceleration))
            - distance,
        )
        y_velocity_contribution = max(
            0,
            (self.state.y_velocity - target_state.y_velocity) ** 2
            / (-2 * (-self.max_linear_acceleration))
            - distance,
        )
        velocity_contribution = np.hypot(
            x_velocity_contribution, y_velocity_contribution
        )

        # encourage the agent to align with the target angle when close
        theta_contribution = abs(self.state.theta - target_state.theta) / (
            distance + 0.2
        )

        # discourage turning
        lazy_turning = self._previous_control.angular_acceleration

        # discourage changing acceleration
        lazy_acceleration = np.hypot(
            self._previous_control.x_acceleration, self._previous_control.y_acceleration
        )

        heuristic = sum(
            (
                distance * 10,
                velocity_contribution * 2,
                theta_contribution * 10,
                search_depth * 0.1,
                lazy_turning * 0.1,
                lazy_acceleration * 0.1,
            )
        )

        return heuristic

    def control_neighbors(
        self, num_linear_acc: int = 8, num_angular_acc: int = 8
    ) -> list[HoloQuadControl]:
        """Generates a set of control inputs.

        Note that the output size scales with num_linear_acc x num_angular_acc

        Args:
            num_linear_acc (int, optional): Number of linear accelerations to generate. Defaults to 8.
                Linear accelerations will lie equally spaced on a circle with
                radius = self.max_angular_acceleration.
            num_angular_acc (int, optional): Number of angular accelerations to generate. Defaults to 8.
                Angular accelerations will lie equally spaced between
                -self.max_angular_acceleration and self.max_angular_acceleration

        Returns:
            list[HoloQuadControl]: list of HoloQuadControl
        """
        linear_acceleration_vectors = (
            unit_vectors_2d(num_linear_acc, self.state.theta)
            * self.max_linear_acceleration
        )
        linear_acceleration_vectors = np.vstack(
            (linear_acceleration_vectors, np.asarray((0, 0)))
        )
        angular_accelerations = np.linspace(
            -self.max_omega, self.max_omega, endpoint=True, num=num_angular_acc
        )
        controls = [
            HoloQuadControl(
                x_acceleration=lin_acc[0],
                y_acceleration=lin_acc[1],
                angular_acceleration=angular_acc,
            )
            for lin_acc, angular_acc in itertools.product(
                linear_acceleration_vectors, angular_accelerations
            )
        ]
        return controls


if __name__ == "__main__":
    from a_star import a_star

    agent = HoloQuad(2, 1)
    target_state = HoloQuadState(x=5, y=3)
    state_space_bin_sizes = {
        "x_velocity": 0.1,
        "y_velocity": 0.1,
        "omega": 0.05,
        "x": 0.1,
        "y": 0.1,
        "theta": 0.05,
    }
    a_star(
        agent=agent,
        static_objects=[],
        target_state=target_state,
        dt=0.1,
        state_space_bin_sizes=state_space_bin_sizes,
    )
