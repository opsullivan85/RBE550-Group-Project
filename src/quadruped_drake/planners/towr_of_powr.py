from planners.simple import *
import subprocess as sub
import os
from pathlib import Path

import lcm
from lcm_types.trunklcm import trunk_state_t


class MultiTrunkPlanner(BasicTrunkPlanner):
    """
    Trunk planner which uses TOWR (https://github.com/ethz-adrl/towr/) to generate
    target motions of the base and feet.
    """

    def __init__(
        self,
        trunk_geometry_frame_id,
        waypoints,
        x_start: float = 0,
        y_start: float = 0,
        z_start: float = 0.3,
        roll_start: float = 0,
        pitch_start: float = 0,
        yaw_start: float = 0,
        foot_positions_start = np.zeros((4,3)),
        world_map: str = "/home/ws/src/savedworlds/world.sdf",
        waypt_duration=1.0,
    ):

        #foot_positions = foot_positions or FootPositions()
        BasicTrunkPlanner.__init__(self, 
                                   trunk_geometry_frame_id, 
                                   x_init=x_start, 
                                   y_init=y_start, 
                                   z_init=z_start,
                                   roll_init=roll_start,
                                   pitch_init=pitch_start,
                                   yaw_init=yaw_start,
                                   foot_positions=foot_positions_start)

        # Time to wait in a standing position before starting the motion
        self.wait_time = 2.5
        self.init_footpos = foot_positions_start
        self.worldmap = world_map
        self.waypoints = waypoints

        # Set up LCM subscriber to read optimal trajectory from TOWR
        self.lc = lcm.LCM()
        subscription = self.lc.subscribe("trunk_state", self.lcm_handler)
        subscription.set_queue_capacity(0)  # disable the queue limit, since we'll process many messages from TOWR

        # Set up storage of optimal trajectory
        self.traj_finished = False
        self.towr_timestamps = []
        self.towr_data = []

        self.global_data = []
        self.global_timestamps = []

        # Call TOWR repeatedly to generate a nominal trunk trajectory
        self.multijectory(waypoints=waypoints, duration=waypt_duration)
        print("Hooo-yah!")


    def lcm_handler(self, channel, data):
        """
        Handle an incoming LCM message. Essentially, we save the data
        to self.towr_data and self.timestamps.
        """
        msg = trunk_state_t.decode(data)

        self.towr_timestamps.append(msg.timestamp)
        self.towr_data.append(msg)

        self.traj_finished = msg.finished  # indicate when the trajectory is over so
        # we can stop listening to LCM

    def GenerateTrunkTrajectory(
        self,
        x_init: float,
        y_init: float,
        z_init: float,
        roll_init: float,
        pitch_init: float,
        yaw_init: float,
        x_final: float,
        y_final: float,
        z_final: float,
        roll_final: float,
        pitch_final: float,
        yaw_final: float,
        world_map: str,
        foot_positions: type(np.array([])),
        duration: float,
    ):
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
        proc = sub.Popen(
            [
                str(
                    Path(__file__)
                    .parent.parent.joinpath("build")
                    .joinpath("towr")
                    .joinpath("trunk_mpc")
                ),
                "walk",
                "0",
                str(x_init),
                str(y_init),
                str(yaw_init),
                str(x_final),
                str(y_final),
                str(yaw_final),
                str(world_map),
                str(foot_positions[0, 0]),  #fl_x
                str(foot_positions[0, 1]),  #fl_y
                str(foot_positions[0, 2]),  #fl_z
                str(foot_positions[1, 0]),  #fr_x
                str(foot_positions[1, 1]),  #fr_y
                str(foot_positions[1, 2]),  #fr_z
                str(foot_positions[2, 0]),  #bl_x
                str(foot_positions[2, 1]),  #bl_y
                str(foot_positions[2, 2]),  #bl_z
                str(foot_positions[3, 0]),  #br_x
                str(foot_positions[3, 1]),  #br_y
                str(foot_positions[3, 2]),  #br_z
                str(z_init),
                str(z_final),
                str(roll_init),
                str(pitch_init),
                str(roll_final),
                str(pitch_final),
                str(duration),
            ],
            env=my_env,
        )
        
        # Read the result over LCM
        self.traj_finished = False  # clear out any stored data
        self.towr_timestamps = []  # from previous trunk trajectories
        self.towr_data = []

        while not self.traj_finished:
            self.lc.handle()
        
        proc.kill()

    def multijectory(self, waypoints, duration):
        #Initialization for global start pose
        x_trunk_prev = self.x_init
        y_trunk_prev = self.y_init
        z_trunk_prev = self.z_init
        roll_trunk_prev = self.roll_init
        pitch_trunk_prev = self.pitch_init
        yaw_trunk_prev = self.yaw_init
        foot_pos_prev = self.init_footpos
        t_prev = 0.0
        
        j = 0
        for w in waypoints:
            print("COMPUTING WAYPOINT #", j)
            j+=1
            x_target, y_target, theta_target = w    # Just using tuples for now
            
            self.GenerateTrunkTrajectory(
                x_init=x_trunk_prev,
                y_init=y_trunk_prev,
                z_init=z_trunk_prev,
                roll_init=roll_trunk_prev,
                pitch_init=pitch_trunk_prev,
                yaw_init=yaw_trunk_prev,
                x_final=x_target,
                y_final=y_target,
                z_final=0.3,            # Is this extra bad? Do we need zed from the A* planner?
                roll_final=0.0,
                pitch_final=0.0,
                yaw_final=theta_target,
                world_map=self.worldmap,
                foot_positions=foot_pos_prev,
                duration=duration,
            )
            
            #Pick off final trunk state for trunk x, y, z, roll, pitch, yaw, and foot positions
            prev_data = self.towr_data[-1]
            trunk_p = np.array(prev_data.base_p)
            trunk_rpy = np.array(prev_data.base_rpy)
            lf_p = np.array(prev_data.lf_p)
            rf_p = np.array(prev_data.rf_p)
            lh_p = np.array(prev_data.lh_p)
            rh_p = np.array(prev_data.rh_p)
            
            x_trunk_prev, y_trunk_prev, z_trunk_prev = trunk_p[0], trunk_p[1], trunk_p[2]
            roll_trunk_prev, pitch_trunk_prev, yaw_trunk_prev = trunk_rpy[0], trunk_rpy[1], trunk_rpy[2]
            foot_pos_prev = np.vstack([lf_p, rf_p, lh_p, rh_p])
            print(foot_pos_prev)
            
            self.global_data.extend(self.towr_data)
            
            #Fix up timestamps for global timestamps
            crimestamps = [t + t_prev for t in self.towr_timestamps]
            self.global_timestamps.extend(crimestamps)
            t_prev = self.global_timestamps[-1]
        
        # Compute maximum magnitude of the control inputs (accelerations)
        self.u2_max = self.ComputeMaxControlInputs()
            
            

    def ComputeMaxControlInputs(self):
        """
        Compute ||u_2||_inf, the maximum L2 norm of the control input, which
        we take to be foot and body accelerations.

        This can be used for approximate-simulation-based control strategies,
        which require Vdot <= gamma(||u_2||_inf)
        """
        u2_max = 0
        for data in self.global_data:
            u2_i = np.linalg.norm(
                np.hstack(
                    [
                        data.lf_pdd,
                        data.rf_pdd,
                        data.lh_pdd,
                        data.rh_pdd,
                        data.base_rpydd,
                        data.base_pdd,
                    ]
                )
            )
            if u2_i > u2_max:
                u2_max = u2_i
        return u2_max

    
    def SetTrunkOutputs(self, context, output):
        self.output_dict = output.get_mutable_value()
        t = context.get_time()

        if t < self.wait_time:
            # Just stand for a bit
            self.SimpleStanding()

        else:
            # Find the timestamp in the (stored) TOWR trajectory that is closest
            # to the current time
            t -= self.wait_time
            # TODO: t-= t_previous
            closest_index = np.abs(np.array(self.global_timestamps) - t).argmin()
            closest_towr_t = self.global_timestamps[closest_index]
            data = self.global_data[closest_index]
            
            # Unpack the TOWR-generated trajectory into the dictionary format that
            # we'll pass to the controller

            # Foot positions
            self.output_dict["p_lf"] = np.array(data.lf_p)
            self.output_dict["p_rf"] = np.array(data.rf_p)
            self.output_dict["p_lh"] = np.array(data.lh_p)
            self.output_dict["p_rh"] = np.array(data.rh_p)
            
            # Foot velocities
            self.output_dict["pd_lf"] = np.array(data.lf_pd)
            self.output_dict["pd_rf"] = np.array(data.rf_pd)
            self.output_dict["pd_lh"] = np.array(data.lh_pd)
            self.output_dict["pd_rh"] = np.array(data.rh_pd)

            # Foot accelerations
            self.output_dict["pdd_lf"] = np.array(data.lf_pdd)
            self.output_dict["pdd_rf"] = np.array(data.rf_pdd)
            self.output_dict["pdd_lh"] = np.array(data.lh_pdd)
            self.output_dict["pdd_rh"] = np.array(data.rh_pdd)

            # Foot contact states: [lf,rf,lh,rh], True indicates being in contact.
            self.output_dict["contact_states"] = [
                data.lf_contact,
                data.rf_contact,
                data.lh_contact,
                data.rh_contact,
            ]

            # Foot contact forces, where each row corresponds to a foot [lf,rf,lh,rh].
            self.output_dict["f_cj"] = np.vstack(
                [
                    np.array(data.lf_f),
                    np.array(data.rf_f),
                    np.array(data.lh_f),
                    np.array(data.rh_f),
                ]
            ).T

            # Body pose
            self.output_dict["rpy_body"] = np.array(data.base_rpy)
            self.output_dict["p_body"] = np.array(data.base_p)

            # Body velocities
            self.output_dict["rpyd_body"] = np.array(data.base_rpyd)
            self.output_dict["pd_body"] = np.array(data.base_pd)

            # Body accelerations
            self.output_dict["rpydd_body"] = np.array(data.base_rpydd)
            self.output_dict["pdd_body"] = np.array(data.base_pdd)

            # Maximum control input (accelerations) magnitude across the trajectory
            self.output_dict["u2_max"] = self.u2_max