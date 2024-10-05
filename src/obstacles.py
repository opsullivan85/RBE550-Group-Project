import numpy as np
import copy
import random as rand

from dataclasses import dataclass


@dataclass
class Obstacle:
    # size in number of grid cells
    size_x_cell: int
    size_y_cell: int
    # size in the z direction in meters
    size_z: float

    # position in units of grid cells
    pos_x_cell: int = 0
    pos_y_cell: int = 0
    
    def getSizeX(self, scale_m_p_cell):
        """ Convert the x direction size into in units of meters, given a scale
        of meters per grid cell.  """
        return scale_m_p_cell * self.size_x_cell

    def getSizeY(self, scale_m_p_cell):
        """ Convert the y direction size into units of meters, given a scale of
        meters per grid cell.  """
        return scale_m_p_cell * self.size_y_cell

    def getSizeZ(self):
        return self.size_z
    
    def getPosX(self, scale_m_p_cell):
        return scale_m_p_cell * self.pos_x_cell

    def getPosY(self, scale_m_p_cell):
        return scale_m_p_cell * self.pos_y_cell

    
class Step(Obstacle):
    def __init__(self, pos_x_cell=0, pos_y_cell=0):
        super().__init__(size_x_cell=4, size_y_cell=4, size_z = 0.075, pos_x_cell=pos_x_cell, pos_y_cell=pos_y_cell)
    
class Column(Obstacle):
    def __init__(self, pos_x_cell=0, pos_y_cell=0):
        super().__init__(size_x_cell=4, size_y_cell=4, size_z=0.75, pos_x_cell=pos_x_cell, pos_y_cell=pos_y_cell)
    

class Grid:
    def __init__(self, size_x, size_y, res_m_p_cell):
        self._size_x_cell = round(size_x / res_m_p_cell)
        self._size_y_cell = round(size_y / res_m_p_cell)
        self._res_m_p_cell = res_m_p_cell
        self._obstacles = []

        # x is in the direction of increasing rows
        # y is in the direction of increasing columns
        self._grid = np.zeros((self._size_y_cell, self._size_x_cell))
        
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
                self._grid[j+ob.pos_y_cell, i+ob.pos_x_cell] = 1
        
        return True
    
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
        """ Check if any part of an obstacle will overlap with an already existing obstacle in the grid.
        """
        # check if any cell in the grid where the obstacle will be place is
        # already occupied
        for i in range(obstacle.size_x_cell):
            for j in range(obstacle.size_y_cell):
                if self._grid[j+obstacle.pos_y_cell, i+obstacle.pos_x_cell]:
                    return True
        return False


def randInsertObstacle(grid):
    """Randomly choose an obstacle and location to place it in the provided grid. 
    """
    rand.seed()
    obstacle_options = [Step(), Column()]
    # randomly choose an obstacle
    ob = copy.deepcopy(rand.choice(obstacle_options))

    for _ in range(grid.getSizeXCell() * grid.getSizeYCell()):
        # choose random location for obstacle
        ob.pos_x_cell = rand.randrange(grid.getSizeXCell())
        ob.pos_y_cell = rand.randrange(grid.getSizeYCell())

        # try place obstacle
        if grid.tryInsertObstacle(ob):
            return ob
    return None


def fillObstacles(grid, density):
    """
    Fill a grid with randomly selected obstacles at random locations until the
    obstacle density is reached. The density is a value between 0.0 and 1.0.
    """
    # seed random number generation so that environments are consistent for particular densities
    rand.seed(1)
    target_area = density * grid.getSizeX() * grid.getSizeY()
    
    # track current density and increment based on selected obstacle, keep going
    # until the desired density is reached
    curr_area = 0.0
    while curr_area < target_area:
        ob = randInsertObstacle(grid)
        # if ob insertion failed, then stop
        if ob == None:
            return
        # increment density
        curr_area += ob.getSizeX(grid.getRes()) * ob.getSizeY(grid.getRes())
        
    
def obstacleToSdf(grid, ob, desc: str) -> str:
    """Convert an obstacle to an SDF link string.
    :param grid The grid that the obstacle is in.
    :param ob The obstacle
    :param desc A unique description of the obstacle to use in parts of the SDF
    """
    res_m_p_cell = grid.getRes()
    # By default the frame is at the center of the box, so first offset the box
    # such that the frame is at the corner of the box, then include the position
    # of the frame transformed from the grid frame to the world frame.
    pose_x = ob.getSizeX(res_m_p_cell)/2+ob.getPosX(res_m_p_cell)-grid.getSizeX()/2
    pose_y = ob.getSizeY(res_m_p_cell)/2+ob.getPosY(res_m_p_cell)-grid.getSizeY()/2
    pose_z = ob.getSizeZ()/2
    return f"""
          <link name="{desc}">
            <pose>{pose_x} {pose_y} {pose_z} 0 0 0</pose>
            <collision name="collision1">
              <geometry>
                <box>
                  <size>{ob.getSizeX(res_m_p_cell)} {ob.getSizeY(res_m_p_cell)} {ob.getSizeZ()}</size>
                </box>
              </geometry>
            </collision>
            <visual name="visual1">
              <geometry>
                <box>
                  <size>{ob.getSizeX(res_m_p_cell)} {ob.getSizeY(res_m_p_cell)} {ob.getSizeZ()}</size>
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
    ground_link = f"""
          <link name="ground">
            <pose>0 0 -0.1 0 0 0</pose>
            <collision name="collision1">
              <geometry>
                <box>
                  <size>{grid.getSizeX()} {grid.getSizeY()} 0.1</size>
                </box>
              </geometry>
            </collision>
            <visual name="visual1">
              <geometry>
                <box>
                  <size>{grid.getSizeX()} {grid.getSizeY()} 0.1</size>
                </box>
              </geometry>
            </visual>
          </link>
          <joint name="fixed_ground_joint" type="fixed">
            <parent>world</parent>
            <child>ground</child>
          </joint>
    """

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
          {ground_link}
          {obstacle_links}
        </model>
      </world>
    </sdf>
    """
    return world_sdf
