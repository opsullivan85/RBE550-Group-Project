from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Union

import numpy as np
from search import ENABLE_VISUALIZATION

if ENABLE_VISUALIZATION:
    import pyglet
import shapely

from control import Control
from simulation import SimAgent, SimObject
from states import Point, Position, State
from util import transform_2d


def height_to_color(map: "HeightMap", height: float) -> tuple[int, int, int]:
    val = 255 - int(((np.arctan(height * 10) + np.pi / 2) / np.pi) * 255)
    return (val, val, val)


@dataclass(frozen=True)
class HeightMapState(State):
    heights: np.ndarray


@dataclass(frozen=True)
class HeightMap(SimObject):
    height: float
    """Height of the map (y direction), in grid cells"""
    width: float
    """Width of the map (x direction), in grid cells"""
    position: Point
    """Location of the (0,0) cell in world space"""
    scale: float
    """How big each grid cell is"""
    state: HeightMapState = None
    """State"""
    _str_tree: shapely.STRtree = None
    """Holds an SR map for collision detection"""

    def __post_init__(self):
        object.__setattr__(self, "_str_tree", shapely.STRtree(self.collision_polygons))

    @classmethod
    def default_from_heightmap(
        cls: "HeightMap",
        height_map: np.ndarray,
        scale: float,
    ) -> "HeightMap":
        state = HeightMapState(heights=height_map)
        return cls(
            height=state.heights.shape[0],
            width=state.heights.shape[1],
            position=Position(),
            scale=scale,
            state=state,
        )

    @cached_property
    def collision_polygons(self) -> list[shapely.geometry.Polygon]:
        polygons = []
        with np.nditer(
            self.state.heights, flags=["multi_index"], op_flags=["readonly"]
        ) as it:
            for height in it:
                # only check collisions for infinite obstacles
                if height != np.inf:
                    continue

                col, row = it.multi_index

                verts = (
                    (0, 0, 1),
                    (self.scale, 0, 1),
                    (self.scale, self.scale, 1),
                    (0, self.scale, 1),
                    (0, 0, 1),
                )
                transformed_verts = transform_2d(
                    verts=verts,
                    x=col * self.scale + self.position.x,
                    y=row * self.scale + self.position.y,
                    theta=0,
                )
                polygon = shapely.geometry.Polygon(transformed_verts)
                shapely.prepare(polygon)
                polygons.append(polygon)

        return polygons

    def drawables(
        self, batch: "pyglet.graphics.Batch"
    ) -> list[Union["pyglet.shapes.ShapeBase" "pyglet.sprite.Sprite"]]:
        sprites = []
        with np.nditer(
            self.state.heights, flags=["multi_index"], op_flags=["readonly"]
        ) as it:
            for height in it:
                col, row = it.multi_index

                sprite = pyglet.shapes.Rectangle(
                    x=col * self.scale + self.position.x,
                    y=row * self.scale + self.position.y,
                    width=self.scale,
                    height=self.scale,
                    color=height_to_color(self, height),
                    batch=batch,
                )
                sprites.append(sprite)
        return sprites

    def to_grid_space(self, ws: Point) -> np.ndarray:
        """Converts to grid space

        Note: No bounds checking is performed here

        Args:
            ws (Point): point in world space to convert

        Returns:
            np.ndarray: (2,) array of coordinates in grid space
        """
        x, y = ws.x, ws.y
        x = (x - self.position.x) // self.scale
        y = (y - self.position.y) // self.scale
        return np.asarray((x, y), dtype="int64")

    def to_grid_space_safe(self, ws: Point) -> np.ndarray:
        """Converts to grid space

        Raises: IndexError if the point is out of bounds

        Args:
            ws (Point): point in world space to convert

        Returns:
            np.ndarray: (2,) array of coordinates in grid space
        """
        gs = self.to_grid_space(ws=ws)
        if gs[0] < 0 or gs[1] < 0 or gs[0] >= self.width or gs[1] >= self.height:
            raise IndexError(f"World space point is out of array bounds. {ws = }")
        return gs

    def to_world_space(self, gs: np.ndarray) -> Point:
        """Converts to the center of the cell in world space

        Args:
            gs (np.ndarray): (2,) array of coordinates in grid space to convert

        Returns:
            Point: point in world space at the center of the cell
        """
        x, y = gs
        x = x * self.scale + self.position.x + self.scale / 2
        y = y * self.scale + self.position.y + self.scale / 2
        return Point(x, y)

    def position_is_free(self, ws: Point) -> bool:
        """Checks if a point in world space corresponds to a free cell

        Args:
            ws (Point): World space point to check

        Returns:
            bool: If the corresponding location is free
        """
        try:
            gs = self.to_grid_space_safe(ws=ws)
        except IndexError:
            return False
        # make sure not on occupied cell in map
        return self.state.heights[gs[0], gs[1]] != np.inf

    def any_intersects(self, others: list[SimObject]) -> bool:
        """checks if any of the other objects are intersecting with the map

        Args:
            others (list[SimObject]): other agents in the simulation

        Returns:
            bool: if the agent state is valid
        """
        for other in others:
            for geometry in other.collision_polygons:
                candidate_idxs = self._str_tree.query(geometry)
                candidates: list[shapely.Polygon] = [
                    self._str_tree.geometries[idx] for idx in candidate_idxs
                ]
                if any(poly.intersects(geometry) for poly in candidates):
                    return True
        return False
