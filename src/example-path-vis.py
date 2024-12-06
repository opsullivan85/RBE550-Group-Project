# Example of using path visualization.

from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

from dataclasses import dataclass
from pathlib import Path


import math
import numpy as np
import obstacles as ob
from path_vis import PathVisualizer

# create the grid, fill it with obstacles, and convert to SDF
env_size_x=5.0
env_size_y=5.0
grid = ob.Grid(size_x=env_size_x, size_y=env_size_y, res_m_p_cell=0.25)
grid.insertFreeZone(pos_x=0.0, pos_y=0.0, radius=1)
grid.insertFreeZone(pos_x=grid.getSizeX(), pos_y=grid.getSizeY(), radius=1)

ob.fillObstacles(grid, density=0.8, terrain=ob.Terrain.ROUGH)
world_sdf = ob.gridToSdf(grid)

# manually specify path
path = np.array([[0,0],
                 [1,1],
                 [1,2.5],
                 [-1,1],
                 [env_size_x, env_size_y]])

# create SDF
path_vis = PathVisualizer("test", path)
path_sdf = path_vis.toSdf()

# create border visualization
border = np.array( [[0,0,-math.pi/4],
                    [0, env_size_y, math.pi/4],
                    [env_size_x, env_size_y, -math.pi/4],
                    [env_size_x, 0, math.pi/4],
                    [0, 0, -math.pi/4]] )
border_sdf = PathVisualizer("border", border, pretty=False, height=0).toSdf()    

# create a model visualizer and add the sdf models
meshcat = StartMeshcat()
visualizer = ModelVisualizer(meshcat=meshcat)
visualizer.parser().AddModelsFromString(file_contents=world_sdf, file_type="sdf")
visualizer.parser().AddModelsFromString(file_contents=path_sdf, file_type="sdf")
visualizer.parser().AddModelsFromString(file_contents=border_sdf, file_type="sdf")

# run
# Start the interactive visualizer.
# Click the "Stop Running" button in MeshCat when you're finished.
visualizer.Run()
