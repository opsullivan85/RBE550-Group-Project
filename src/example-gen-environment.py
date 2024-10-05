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


import obstacles as ob

# create the grid, fill it with obstacles, and convert to SDF
grid = ob.Grid(size_x=5.0, size_y=5.0, res_m_p_cell=0.25)
ob.fillObstacles(grid, density=0.15)
world_sdf = ob.gridToSdf(grid)

with open(str(Path(__file__).parent.joinpath("world.sdf")), "w") as file:
    file.write(world_sdf)

meshcat = StartMeshcat()

# create a model visualizer and add the world sdf
visualizer = ModelVisualizer(meshcat=meshcat)
visualizer.parser().AddModels(file_contents=world_sdf, file_type="sdf")

# Start the interactive visualizer.
# Click the "Stop Running" button in MeshCat when you're finished.
visualizer.Run()
