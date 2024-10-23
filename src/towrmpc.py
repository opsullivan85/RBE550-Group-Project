#!/usr/bin/env python

from pydrake.all import *
import quadruped_drake
from quadruped_drake.controllers import *
from quadruped_drake.planners import BasicTrunkPlanner, TowrTrunkPlanner
import os
import sys
from pathlib import Path

quadruped_drake_path = str(Path(quadruped_drake.__file__).parent)

############### Common Parameters ###################
show_trunk_model = True
use_lcm = False

control_method = "ID"  # ID = Inverse Dynamics (standard QP),
# B = Basic (simple joint-space PD),
# MPTC = task-space passivity
# PC = passivity-constrained
# CLF = control-lyapunov-function based

sim_time = 6.0
dt = 1e-3
target_realtime_rate = 1.0

show_diagram = False
make_plots = False

x_init: float = 0
y_init: float = -0.1
theta_init: float = 0
x_final: float = 1.5 / 10
y_final: float = -0.1
theta_final: float = 3.1415 / 8 / 10
world_map: str = "/home/ws/src/savedworlds/world.sdf"

#####################################################

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
world = Parser(plant, scene_graph, "world").AddModels(
    "/home/ws/src/savedworlds/world.sdf"
)

# Add flat ground with friction
X_BG = RigidTransform()
X_BG.set_translation(np.array([0.0, 0.0, -0.01]))
surface_friction = CoulombFriction(static_friction=0.7, dynamic_friction=0.7)
plant.RegisterCollisionGeometry(
    plant.world_body(),
    X_BG,
    HalfSpace(),
    "ground_collision",
    surface_friction,
)
plant.RegisterVisualGeometry(
    plant.world_body(),
    X_BG,
    HalfSpace(),
    "ground_visual",
    np.array([0.5, 0.5, 0.5, 0.0]),
)

plant.Finalize()
assert plant.geometry_source_is_registered()

# Add custom visualizations for the trunk model
trunk_source = scene_graph.RegisterSource("trunk")
trunk_frame = GeometryFrame("trunk")
scene_graph.RegisterFrame(trunk_source, trunk_frame)

trunk_shape = Box(0.4, 0.2, 0.1)
trunk_color = np.array([0.1, 0.1, 0.1, 0.4])
X_trunk = RigidTransform()
X_trunk.set_translation(np.array([0.0, 0.0, 0.0]))

trunk_geometry = GeometryInstance(X_trunk, trunk_shape, "trunk")
if show_trunk_model:
    trunk_geometry.set_illustration_properties(
        MakePhongIllustrationProperties(trunk_color)
    )
scene_graph.RegisterGeometry(trunk_source, trunk_frame.id(), trunk_geometry)

trunk_frame_ids = {"trunk": trunk_frame.id()}

for foot in ["lf", "rf", "lh", "rh"]:
    foot_frame = GeometryFrame(foot)
    scene_graph.RegisterFrame(trunk_source, foot_frame)

    foot_shape = Sphere(0.02)
    X_foot = RigidTransform()
    foot_geometry = GeometryInstance(X_foot, foot_shape, foot)
    if show_trunk_model:
        foot_geometry.set_illustration_properties(
            MakePhongIllustrationProperties(trunk_color)
        )

    scene_graph.RegisterGeometry(trunk_source, foot_frame.id(), foot_geometry)
    trunk_frame_ids[foot] = foot_frame.id()

default_foot_positions = quadruped_drake.planners.FootPositions()
planner = builder.AddSystem(
    TowrTrunkPlanner(
        trunk_frame_ids,
        x_init=x_init,
        y_init=y_init,
        theta_init=theta_init,
        x_final=x_final,
        y_final=y_final,
        theta_final=theta_final,
        world_map=world_map,
        foot_positions=default_foot_positions
    )
)

if control_method == "B":
    controller = builder.AddSystem(BasicController(plant, dt, use_lcm=use_lcm))
elif control_method == "ID":
    controller = builder.AddSystem(IDController(plant, dt, use_lcm=use_lcm))
elif control_method == "MPTC":
    controller = builder.AddSystem(MPTCController(plant, dt, use_lcm=use_lcm))
elif control_method == "PC":
    controller = builder.AddSystem(PCController(plant, dt, use_lcm=use_lcm))
elif control_method == "CLF":
    controller = builder.AddSystem(CLFController(plant, dt, use_lcm=use_lcm))
else:
    print("Invalid control method %s" % control_method)
    sys.exit(1)

builder.Connect(
    scene_graph.get_query_output_port(), plant.get_geometry_query_input_port()
)
builder.Connect(
    plant.get_geometry_poses_output_port(),
    scene_graph.get_source_pose_port(plant.get_source_id()),
)
builder.Connect(
    planner.GetOutputPort("trunk_geometry"),
    scene_graph.get_source_pose_port(trunk_source),
)

if not control_method == "B":
    builder.Connect(
        planner.GetOutputPort("trunk_trajectory"), controller.get_input_port(1)
    )

builder.Connect(
    controller.GetOutputPort("quad_torques"), plant.get_actuation_input_port(quad)
)
builder.Connect(plant.get_state_output_port(), controller.GetInputPort("quad_state"))

logger = LogVectorOutput(controller.GetOutputPort("output_metrics"), builder)

meshcat = StartMeshcat()
meshcat_params = MeshcatVisualizerParams()
meshcat_params.publish_period = dt

visualizer = ModelVisualizer(meshcat=meshcat)
model_visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, meshcat_params
)

diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

simulator = Simulator(diagram, diagram_context)
if use_lcm:
    simulator.set_target_realtime_rate(0.0)
else:
    simulator.set_target_realtime_rate(target_realtime_rate)

plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
q0 = np.asarray(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        x_init,
        y_init,
        0.3,
        0.0,
        -0.8,
        1.6,
        0.0,
        -0.8,
        1.6,
        0.0,
        -0.8,
        1.6,
        0.0,
        -0.8,
        1.6,
    ]
)
qd0 = np.zeros(plant.num_velocities())
plant.SetPositions(plant_context, q0)
plant.SetVelocities(plant_context, qd0)

# Set up the loop for MPC updates
time = 0.0
while time < sim_time:
    # Extract current plant state
    q = plant.GetPositions(plant_context)
    v = plant.GetVelocities(plant_context)
    print(f"{q = }")
    print(f"{v = }")
    # Update planner with the current state
    planner.SetPositions(plant, plant_context, q)
    planner.SetVelocities(plant, plant_context, v)
    
    # Advance the simulation by one time step
    simulator.AdvanceTo(time + dt)
    
    # Update time
    time += dt

if make_plots:
    import matplotlib.pyplot as plt

    log = logger.FindLog(diagram_context)

    t = log.sample_times()[10:]
    V = log.data()[0, 10:]
    err = log.data()[1, 10:]
    res = log.data()[2, 10:]
    Vdot = log.data()[3, 10:]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, Vdot, linewidth="2")
    plt.axhline(0, linestyle="dashed", color="grey")
    plt.ylabel("$\dot{V}$")

    plt.subplot(3, 1, 2)
    plt.plot(t, V, linewidth="2")
    plt.ylabel("$V$")

    plt.subplot(3, 1, 3)
    plt.plot(t, err, linewidth="2")
