"""This is a wrapper library for quadruped_drake.
The way quadruped_drake was made you just would
import its sub-modules individually.
    ex: `import controllers` instead of `import quadruped_drake.controllers`

This file allows us to do `import qd.controllers` and prevents
us from needing to repeat the sys.path.append line everywhere.

As this is currently written, it assumes that quadruped_drake is located
in "/home/repos/quadruped_drake".
"""

import sys
from importlib import import_module
import os
import subprocess as sub

quadruped_drake_path = "/home/repos/quadruped_drake"
sys.path.append(quadruped_drake_path)

# add folders in quadruped_drake and set them as sub-modules of this.
# more folders may need to be added these two were the most immediately obvious
for module_name in ("controllers", "planners"):
    module = import_module(module_name)  # allows for `import qd.planners``
    globals()[module_name] = module  # allows for `import qd; qd.planners`
    sys.modules[f"{__name__}.{module_name}"] = module

sys.path.remove(quadruped_drake_path)

#####################################################


# This is super sub-optimal
# quadruped_drake.planners.towr.TowrTrunkPlanner.GenerateTrunkTrajectory
# uses a relative import that doesn't work unless the file you are running
# is in the quadruped_drake project direcrtory. This manually updates
# the function to fix it, then writes over the one defined in quadruped_drake.
# As far as I can tell this is the only instance of a hard-coded relative path in quadruped_drake.
# Just forking quadruped_drake would probably be nicer than this.
def GenerateTrunkTrajectory(self):
    """
    Call a TOWR cpp script to generate a trunk model trajectory.
    Read in the resulting trajectory over LCM.
    """
    # Run the trajectory optimization (TOWR)
    # syntax is trunk_mpc gait_type={walk,trot,pace,bound,gallop} optimize_gait={0,1} distance_x distance_y
    my_env = os.environ
    my_env["LD_LIBRARY_PATH"] = (
        ""  # need to set this so only the custom version of towr gets used, not
    )
    # the one in catkin_ws (if it exits)
    # LOCAL CHANGE MADE HERE, ADDED {quadruped_drake_path}
    sub.Popen(
        [f"{quadruped_drake_path}/build/towr/trunk_mpc", "walk", "0", "1.5", "0.0"],
        env=my_env,
    )

    # Read the result over LCM
    self.traj_finished = False  # clear out any stored data
    self.towr_timestamps = []  # from previous trunk trajectories
    self.towr_data = []

    while not self.traj_finished:
        self.lc.handle()


# over-write the function
sys.modules[
    f"{__name__}.planners"
].towr.TowrTrunkPlanner.GenerateTrunkTrajectory = GenerateTrunkTrajectory

# we don't want these in the __dir__
del sys
del GenerateTrunkTrajectory
del import_module
del module_name
del module
