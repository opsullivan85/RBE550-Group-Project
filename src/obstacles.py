import math
import numpy as np
import copy
import random as rand

from enum import Enum
from dataclasses import dataclass


@dataclass
class Obstacle:
    """ Represents an obstacle in the grid environment.
    """
    # size in number of grid cells
    size_x_cell: int
    size_y_cell: int
    # size in the z direction in meters
    size_z: float

    # position in units of grid cells
    pos_x_cell: int = 0
    pos_y_cell: int = 0
    scale_m_p_cell: float = 0
    
    def getSizeX(self):
        """ Convert the x direction size into in units of meters, given a scale
        of meters per grid cell.  """
        return self.scale_m_p_cell * self.size_x_cell

    def getSizeY(self):
        """ Convert the y direction size into units of meters, given a scale of
        meters per grid cell.  """
        return self.scale_m_p_cell * self.size_y_cell

    def getSizeZ(self):
        return self.size_z
    
    def getPosX(self):
        return self.scale_m_p_cell * self.pos_x_cell

    def getPosY(self):
        return self.scale_m_p_cell * self.pos_y_cell

    def getCornerPts(self) -> list(tuple()):
        """ Get a list of the corner points in the continuous world frame coordinates.
        """
        # the frame of the cell of an obstacle is at the bottom left corner

        # corner points in cell units relative to the frame of the obstacle
        corner_pts_cell = [(0,0), (0,self.size_y_cell), (self.size_x_cell,self.size_y_cell), (self.size_x_cell,0)]
        # convert to continuous points in the world frame
        return [(self.getPosX() + x_cell*self.scale_m_p_cell, self.getPosY() +
                 y_cell*self.scale_m_p_cell) for x_cell, y_cell in corner_pts_cell]

    def updateHeight(self):
        return

    
class Step(Obstacle):
    """ Fixed height step obstacle.
    """
    def __init__(self, pos_x_cell=0, pos_y_cell=0):
        super().__init__(size_x_cell=2, size_y_cell=2, size_z = 0.10, pos_x_cell=pos_x_cell, pos_y_cell=pos_y_cell)


class VariableHeightStep(Obstacle):
    """ Variable height step obstacle. Update the height after construction.
    """
    def __init__(self, pos_x_cell=0, pos_y_cell=0):
        super().__init__(size_x_cell=2, size_y_cell=2, size_z = 0.10, pos_x_cell=pos_x_cell, pos_y_cell=pos_y_cell)

    def updateHeight(self):
        self.size_z = rand.random() * 0.2
        
    
class Column(Obstacle):
    """ Column or pillar obstacle """
    def __init__(self, pos_x_cell=0, pos_y_cell=0):
        super().__init__(size_x_cell=4, size_y_cell=4, size_z=0.75, pos_x_cell=pos_x_cell, pos_y_cell=pos_y_cell)
    

class Grid:
    """ The grid cell environment. Contains a 2D grid cell array and obstacles
    that cover specific cells.  """

    def __init__(self, size_x, size_y, res_m_p_cell):
        self._size_x_cell = round(size_x / res_m_p_cell)
        self._size_y_cell = round(size_y / res_m_p_cell)
        self._res_m_p_cell = res_m_p_cell
        self._obstacles = []
        self._free_zones = []

        # y is the row index and x is the column index
        self._grid = np.zeros((self._size_y_cell, self._size_x_cell))

    def insertFreeZone(self, pos_x, pos_y, radius):
        """ No obstacles will be generated that overlap with a free zone, which
        is a circle with its center located at the provided position.
        """
        self._free_zones.append((pos_x, pos_y, radius))
        
    def getRes(self):
        return self._res_m_p_cell

    def getSizeXCell(self):
        """ Get the number of cells in the x direction of the grid.
        """
        return self._size_x_cell

    def getSizeYCell(self):
        """ Get the number of cells in the y direction of the grid.
        """
        return self._size_y_cell
    
    def getSizeX(self):
        """ Get the size of the grid in the x direction in units of meters.
        """
        return self._size_x_cell * self._res_m_p_cell

    def getSizeY(self):
        """ Get the size of the grid in the y direction in units of meters.
        """
        return self._size_y_cell * self._res_m_p_cell

    def tryInsertObstacle(self, ob) -> bool:
        """ Attempt to insert an obstacle into the grid. If the obstacle will
        overlap with an already existing obstacle, or parts of the obstacle are
        outside the bounds of the grid, then the obstacle is not inserted.  """
        if not self._inBounds(ob):
            return False

        if self._overlapping(ob):
            return False
        
        self._obstacles.append(ob)
        for i in range(ob.size_x_cell):
            for j in range(ob.size_y_cell):
                self._grid[j+ob.pos_y_cell, i+ob.pos_x_cell] = ob.size_z
        
        return True

    def toArray(self):
        """ Return a copy of the internal obstacle grid.
        """
        return self._grid.copy()
    
    def getObstacles(self):
        return self._obstacles

    def _inBounds(self, obstacle) -> bool:
        """ Check if all parts of an obstacle are within the bounding edges of the grid.
        """
        for i in range(obstacle.size_x_cell):
            for j in range(obstacle.size_y_cell):
                if self._grid.shape[0] <= j+obstacle.pos_y_cell:
                    return False
                if self._grid.shape[1] <= i+obstacle.pos_x_cell:
                    return False
        return True

    def _overlapping(self, obstacle):
        """ Check if any part of an obstacle will overlap with an already
        existing obstacle or free space zone in the grid.
        """
        # check if any cell in the grid where the obstacle will be place is
        # already occupied
        for i in range(obstacle.size_x_cell):
            for j in range(obstacle.size_y_cell):
                if self._grid[j+obstacle.pos_y_cell, i+obstacle.pos_x_cell]:
                    return True

        # check for overlap with a free zone
        for zone in self._free_zones:
            for p in obstacle.getCornerPts():
                diff = np.array([zone[0], zone[1]]) - np.array(p)
                if np.linalg.norm(diff) < zone[2]:
                    return True


        return False


def randInsertObstacle(grid, obstacle_options: list, weights: list):
    """Randomly choose an obstacle and location to place it in the provided grid.
    :param ob_i If specified, gives the index of the obstacle type to insert.
    """
    # psuedorandomly choose obstacle type based on weights
    ob = copy.deepcopy(rand.choices(population=obstacle_options, weights=weights)[0])
    
    for _ in range(grid.getSizeXCell() * grid.getSizeYCell()):
        # choose random location for obstacle
        ob.pos_x_cell = rand.randrange(grid.getSizeXCell())
        ob.pos_y_cell = rand.randrange(grid.getSizeYCell())
        ob.scale_m_p_cell = grid.getRes()
        ob.updateHeight()

        # try place obstacle
        if grid.tryInsertObstacle(ob):
            return ob
    return None


class Terrain(Enum):
    SIMPLE=1
    ROUGH=2


def fillObstacles(grid, density, terrain = Terrain.SIMPLE):
    """Fill a grid with randomly selected obstacles at random locations until
    the obstacle density is reached. The density is a value between 0.0 and
    1.0. The set of obstacles used and their probability distribution are based
    on the provided terrain type.

    """
    # seed random number generation so that environments are consistent for
    # particular densities
    rand.seed(1)
    target_area = density * grid.getSizeXCell() * grid.getSizeYCell()

    # obstacles types and distribution weights
    obstacle_options = [Step(), Column()]
    weights = [2, 1]

    if terrain == Terrain.ROUGH:
        obstacle_options = [VariableHeightStep(), Column()]
        weights = [16, 1]
    
    # track current density and increment based on selected obstacle, keep going
    # until the desired density is reached
    curr_area = 0.0
    while curr_area < target_area:
        ob = randInsertObstacle(grid, obstacle_options=obstacle_options,
                                weights=weights)
        # if ob insertion failed, then stop
        if ob == None:
            return
        # increment density
        curr_area += ob.size_y_cell * ob.size_x_cell
        
    
def obstacleToSdf(grid, ob, desc: str) -> str:
    """Convert an obstacle to an SDF link string.
    :param grid The grid that the obstacle is in.
    :param ob The obstacle
    :param desc A unique description of the obstacle to use in parts of the SDF
    """
    # By default the frame is at the center of the box, so first offset the box
    # such that the frame is at the corner of the box, then include the position
    # of the frame transformed from the grid frame to the world frame.
    # pose_x = ob.getSizeX()/2+ob.getPosX()-grid.getSizeX()/2
    # pose_y = ob.getSizeY()/2+ob.getPosY()-grid.getSizeY()/2
    # pose_z = ob.getSizeZ()/2
    pose_x = ob.getSizeX()/2+ob.getPosX()
    pose_y = ob.getSizeY()/2+ob.getPosY()
    pose_z = ob.getSizeZ()/2
    return f"""
          <link name="{desc}">
            <pose>{pose_x} {pose_y} {pose_z} 0 0 0</pose>
            <collision name="collision1">
              <geometry>
                <box>
                  <size>{ob.getSizeX()} {ob.getSizeY()} {ob.getSizeZ()}</size>
                </box>
              </geometry>
            </collision>
            <visual name="visual1">
              <geometry>
                <box>
                  <size>{ob.getSizeX()} {ob.getSizeY()} {ob.getSizeZ()}</size>
                </box>
              </geometry>
            </visual>
          </link>
          <joint name="fixed_joint_{desc}" type="fixed">
            <parent>world</parent>
            <child>{desc}</child>
          </joint>
          """    
    
# Convert a grid and its obstacles into an SDF string.
def gridToSdf(grid) -> str:

    # create a single sdf string with each obstacle link
    obstacle_links = ""
    i = 0
    for ob in grid.getObstacles():
        obstacle_links += obstacleToSdf(grid, ob, f"obstacle_{i}")
        i += 1
    
    # create final sdf with ground plane and obstacles
    world_sdf = f"""<?xml version="1.0" ?>
    <sdf version="1.4">
      <world name="simple_world">
        <model name="weld_models">
          {obstacle_links}
        </model>
      </world>
    </sdf>
    """
    return world_sdf
