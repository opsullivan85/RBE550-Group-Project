#!/usr/bin/env python

from pydrake.all import *
import quadruped_drake
from quadruped_drake.controllers import *
from quadruped_drake.planners import BasicTrunkPlanner, TowrTrunkPlanner
import pydot
import obstacles as ob

import os
import sys
from pathlib import Path


def createWorld(world_map_path):
    # create the obstacle environment and save it to a temporary file for towr to process
    grid = ob.Grid(size_x=5.0, size_y=5.0, res_m_p_cell=0.17)
    ob.fillObstacles(grid, density=0.13)
    world_sdf = ob.gridToSdf(grid)
    with open(world_map_path, "w") as f:
        f.write(world_sdf)

    return Parser(plant, scene_graph, "world").AddModels(
        file_contents=world_sdf, file_type="sdf"
    )


def addGround(plant):
    # Add a flat ground with friction
    X_BG = RigidTransform()
    X_BG.set_translation(np.array([0.0, 0.0, -0.01]))  # Offset halfspace ground for now
    surface_friction = CoulombFriction(static_friction=0.7, dynamic_friction=0.7)
    plant.RegisterCollisionGeometry(
        plant.world_body(),  # the body for which this object is registered
        X_BG,  # The fixed pose of the geometry frame G in the body frame B
        HalfSpace(),  # Defines the geometry of the object
        "ground_collision",  # A name
        surface_friction,
    )  # Coulomb friction coefficients
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_BG,
        HalfSpace(),
        "ground_visual",
        np.array([0.5, 0.5, 0.5, 0.0]),
    )  # Color set to be completely transparent


def addTrunkGeometry(scene_graph):
    # Add custom visualizations for the trunk model
    trunk_source_id = scene_graph.RegisterSource("trunk")
    trunk_frame = GeometryFrame("trunk")
    scene_graph.RegisterFrame(trunk_source_id, trunk_frame)

    trunk_shape = Box(0.4, 0.2, 0.1)
    trunk_color = np.array([0.1, 0.1, 0.1, 0.4])
    X_trunk = RigidTransform()
    X_trunk.set_translation(np.array([0.0, 0.0, 0.0]))

    trunk_geometry = GeometryInstance(X_trunk, trunk_shape, "trunk")
    if show_trunk_model:
        trunk_geometry.set_illustration_properties(
            MakePhongIllustrationProperties(trunk_color)
        )
    scene_graph.RegisterGeometry(trunk_source_id, trunk_frame.id(), trunk_geometry)

    trunk_frame_ids = {"trunk": trunk_frame.id()}

    for foot in ["lf", "rf", "lh", "rh"]:
        foot_frame = GeometryFrame(foot)
        scene_graph.RegisterFrame(trunk_source_id, foot_frame)

        foot_shape = Sphere(0.02)
        X_foot = RigidTransform()
        foot_geometry = GeometryInstance(X_foot, foot_shape, foot)
        if show_trunk_model:
            foot_geometry.set_illustration_properties(
                MakePhongIllustrationProperties(trunk_color)
            )

        scene_graph.RegisterGeometry(trunk_source_id, foot_frame.id(), foot_geometry)
        trunk_frame_ids[foot] = foot_frame.id()

    return trunk_source_id, trunk_frame_ids


def makePlanner(planning_method, world_map_path):
    # high-level trunk-model planner
    if planning_method == "basic":
        planner = builder.AddSystem(BasicTrunkPlanner(trunk_frame_ids))
    elif planning_method == "towr":
        planner = builder.AddSystem(
            TowrTrunkPlanner(
                trunk_frame_ids,
                x_init=x_init,
                y_init=y_init,
                yaw_init=theta_init,
                x_final=x_final,
                y_final=y_final,
                yaw_final=theta_final,
                world_map=world_map,
            )
        )
    else:
        print("Invalid planning method %s" % planning_method)
        sys.exit(1)
    return planner


def makeController(control_method, plant, dt):
    # low-level whole-body controller
    if control_method == "B":
        controller = builder.AddSystem(BasicController(plant, dt))
    elif control_method == "ID":
        controller = builder.AddSystem(IDController(plant, dt))
    elif control_method == "MPTC":
        controller = builder.AddSystem(MPTCController(plant, dt))
    elif control_method == "PC":
        controller = builder.AddSystem(PCController(plant, dt))
    elif control_method == "CLF":
        controller = builder.AddSystem(CLFController(plant, dt))
    else:
        print("Invalid control method %s" % control_method)
        sys.exit(1)
    return controller

    
def connectSystems(builder, scene_graph, plant, planner, controller, trunk_source_id):
    # Set up the Scene Graph
    builder.Connect(
        scene_graph.get_query_output_port(), plant.get_geometry_query_input_port()
    )
    builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        planner.GetOutputPort("trunk_geometry"),
        scene_graph.get_source_pose_port(trunk_source_id),
    )

    # Connect the trunk-model planner to the controller
    if not control_method == "B":
        builder.Connect(
            planner.GetOutputPort("trunk_trajectory"), controller.get_input_port(1)
        )

    # Connect the controller to the simulated plant
    builder.Connect(
        controller.GetOutputPort("quad_torques"), plant.get_actuation_input_port(quad)
    )
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("quad_state"))


def setupVisualization(builder, scene_graph, publish_period = None):
    # Only provide a publish period if necessary because it can slow things down
    # if it has a very low value. Originally it was set to the dt value of the
    # simulation which probably isn't necessary.

    #Set up the Meshcat Visualizer
    meshcat = StartMeshcat()
    meshcat_params = MeshcatVisualizerParams()
    if publish_period != None:
        meshcat_params.publish_period = dt  # Set the publishing rate

    # Add the visualizer to the builder
    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_params
    )



quadruped_drake_path = str(Path(quadruped_drake.__file__).parent)

############### Common Parameters ###################
show_trunk_model = True

planning_method = "basic"  # "towr" or "basic"
control_method = "ID"  # ID = Inverse Dynamics (standard QP),
# B = Basic (simple joint-space PD),
# MPTC = task-space passivity
# PC = passivity-constrained
# CLF = control-lyapunov-function based

sim_time = 1000.0 # make it long
dt = 1e-3
target_realtime_rate = 1.0

show_diagram = False
make_plots = False

x_init: float = 0
y_init: float = 0
theta_init: float = 0
x_final: float = 1.5 / 2
y_final: float = -0.2
theta_final: float = 3.1415 / 8
world_map_path: str = "/home/ws/src/world.sdf"

#####################################################

# Drake only loads things relative to the drake path, so we have to do some hacking
# to load an arbitrary file
robot_description_path = (
    f"{quadruped_drake_path}/models/mini_cheetah/mini_cheetah_mesh.urdf"
)
drake_path = getDrakePath()
robot_description_file = "drake/" + os.path.relpath(
    robot_description_path, start=drake_path
)

robot_urdf = FindResourceOrThrow(robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
quad = Parser(plant=plant).AddModelFromFile(robot_urdf, "quad")

world = createWorld(world_map_path)
addGround(plant)

# Turn off gravity
# g = plant.mutable_gravity_field()
# g.set_gravity_vector([0,0,0])

plant.Finalize()
assert plant.geometry_source_is_registered()

trunk_source_id, trunk_frame_ids = addTrunkGeometry(scene_graph)

# Create high-level trunk-model planner and low-level whole-body controller
planner = makePlanner(planning_method, world_map_path)
controller = makeController(control_method, plant, dt)

connectSystems(builder, scene_graph, plant, planner, controller, trunk_source_id)

# Add loggers
logger = LogVectorOutput(controller.GetOutputPort("output_metrics"), builder)

setupVisualization(builder, scene_graph)

# Compile the diagram: no adding control blocks from here on out
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# Visualize the diagram
if show_diagram:
    plt.figure()
    plot_system_graphviz(diagram, max_depth=2)
    plt.show()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(target_realtime_rate)

# Set initial states
quad_model_id = plant.GetModelInstanceByName("quad")
PositionView = namedview(
        "Positions",
        plant.GetPositionNames(
            quad_model_id, always_add_suffix=False
        ),
    )
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

q0 = PositionView(plant.GetPositions(plant_context, quad_model_id))
q0.body_qw = 1.0
q0.body_qx = 0.0
q0.body_qy = 0.0
q0.body_qz = 0.0
q0.body_x = x_init
q0.body_y = y_init
q0.body_z = 0.3
q0.torso_to_abduct_fl_j = 0.0
q0.abduct_fl_to_thigh_fl_j = -0.8
q0.thigh_fl_to_knee_fl_j = 1.6
q0.torso_to_abduct_fr_j = 0.0
q0.abduct_fr_to_thigh_fr_j = -0.8
q0.thigh_fr_to_knee_fr_j = 1.6
q0.torso_to_abduct_hl_j = 0.0
q0.abduct_hl_to_thigh_hl_j = -0.8
q0.thigh_hl_to_knee_hl_j = 1.6
q0.torso_to_abduct_hr_j = 0.0
q0.abduct_hr_to_thigh_hr_j = -0.8
q0.thigh_hr_to_knee_hr_j = 1.6
plant.SetPositions(plant_context, quad_model_id, q0[:])

qd0 = np.zeros(plant.num_velocities())
plant.SetVelocities(plant_context, qd0)

# Visualize the diagram.
pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=1))[0].write_svg("diagram.svg")

# Run the simulation!
simulator.AdvanceTo(sim_time)
