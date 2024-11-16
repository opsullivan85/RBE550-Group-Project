from scipy.spatial.transform import Rotation as R
import numpy as np
import itertools
import math

class PathVisualizer:
    def __init__(self, path: np.ndarray, line_thickness: int, sphere_radius: int):
        self._path = path
        self._line_thickness = line_thickness
        self._sphere_radius = sphere_radius

    def toSdf(self):
        # create and SDF string of the entire path

        links_sdf = ""
        i = 0
        for start, end in itertools.pairwise(self._path):
            links_sdf += self.pathSegToSdf(start, end, i)
            i += 1

        # add sphere for last point
        links_sdf += self.pathPointToSdf(self._path[-1], link_id=i)
        
        sdf = f"""<?xml version="1.0" ?>
        <sdf version="1.4">
            <model name="path">
              {links_sdf}
            </model>
        </sdf>
        """
        return sdf

    def pathPointToSdf(self, pt: np.ndarray, **kwargs):
        # create a sphere at the point

        link_id = kwargs.get('link_id', None)
        
        assert pt.shape == (2,)

        
        # create sphere sdf visualization strings
        material_blue = """
            <material>
              <ambient>0 0 1 1</ambient>
              <diffuse>0 0 1 1</diffuse>
              <specular>0.1 0.1 0.1 1</specular>
            </material>
        """

        sphere_vis = f"""
          <visual name="sphere_vis">
            <geometry>
              <sphere>
                <radius>{self._sphere_radius}</radius>
              </sphere>
            </geometry>
            {material_blue}
          </visual>
        """
        if link_id is not None:
            sphere_vis = f"""
                <link name="link_{link_id}">
                  <pose>{pt[0]} {pt[1]} 0 0 0 0</pose>
                  {sphere_vis}
                </link>
            """
        return sphere_vis
    
    def pathLineToSdf(self, start: np.ndarray, end: np.ndarray, **kwargs):
        # create a cyclinder that goes from the start point to the end point
        
        link_id = kwargs.get('link_id', None)

        # Find the euler angles to transform a vertical cylinder to go from the
        # start to end point.

        line_vec = np.append(end-start, 0) # add a zero z component
        line_len = np.linalg.norm(line_vec)

        # first calculate the angle-axis representation
        u_line = line_vec / np.linalg.norm(line_vec)
        u_z = np.array([0,0,1])
        rot_axis = np.cross(u_z, u_line)
        norm_cross = np.linalg.norm(rot_axis);
        angle = math.atan(norm_cross / np.dot(u_z, u_line))

        # extrinsic rotations about x, y, z (in this order)
        euler = R.from_rotvec(angle * rot_axis).as_euler('xyz')
        
        material_green = """
            <material>
              <ambient>0 1 0 1</ambient>
              <diffuse>0 1 0 1</diffuse>
              <specular>0.1 0.1 0.1 1</specular>
            </material>
        """
        line_vis = f"""
          <visual name="vis_{link_id}">
            <pose>{line_vec[0]/2} {line_vec[1]/2} {line_vec[2]/2} {euler[0]} {euler[1]} {euler[2]}</pose>
            <geometry>
              <cylinder>
                <radius>{self._line_thickness/2}</radius>
                <length>{line_len}</length>
              </cylinder>
            </geometry>
            {material_green}
          </visual>
        """
        return line_vis
        

    def pathSegToSdf(self, start: np.ndarray, end: np.ndarray, id_str: int):
        # Create an SDF string of a two point path segment. A sphere is created
        # for the start point, and a cyclinder goes from the start to the end.

        # create line visualization strings
        sphere_vis = self.pathPointToSdf(start)
        line_vis = self.pathLineToSdf(start, end, link_id=id_str)

        link_sdf = f"""
        <link name="link_{id_str}">
          <pose>{start[0]} {start[1]} 0 0 0 0</pose>
          {sphere_vis}
          {line_vis}
        </link>
        """
        return link_sdf
