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


import numpy as np
import obstacles as ob
from path_vis import PathVisualizer

# manually specify path
path = np.array([[0,1],
                 [1,1],
                 [1,2.5],
                 [-1,1],
                 [0,2]])

# create SDF
path_vis = PathVisualizer("test", path, line_thickness=0.01, sphere_radius=0.05)
path_sdf = path_vis.toSdf()

# create the grid, fill it with obstacles, and convert to SDF
grid = ob.Grid(size_x=5.0, size_y=5.0, res_m_p_cell=0.25)
ob.fillObstacles(grid, density=0.15)
world_sdf = ob.gridToSdf(grid)

# create a model visualizer and add the sdf models
meshcat = StartMeshcat()
visualizer = ModelVisualizer(meshcat=meshcat)
visualizer.parser().AddModelsFromString(file_contents=world_sdf, file_type="sdf")
visualizer.parser().AddModelsFromString(file_contents=path_sdf, file_type="sdf")

# run
# Start the interactive visualizer.
# Click the "Stop Running" button in MeshCat when you're finished.
visualizer.Run()
