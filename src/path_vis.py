from scipy.spatial.transform import Rotation as R
import numpy as np
import itertools
import math
from enum import Enum


class PathVisualizer:
    _black ="""
            <material>
              <ambient>0 0 0 1</ambient>
              <diffuse>0 0 0 1</diffuse>
              <specular>0.1 0.1 0.1 1</specular>
            </material>
        """
    _blue = """
            <material>
              <ambient>0 0 1 1</ambient>
              <diffuse>0 0 1 1</diffuse>
              <specular>0.1 0.1 0.1 1</specular>
            </material>
        """
    _green = """
            <material>
              <ambient>0 1 0 1</ambient>
              <diffuse>0 1 0 1</diffuse>
              <specular>0.1 0.1 0.1 1</specular>
            </material>
        """
    
    def __init__(self, name: str, path: np.ndarray, pretty=True, height=0.3, line_thickness: int = 0.01, sphere_radius: int = 0.05):
        self._name = name
        self._path = path
        self._pretty = pretty
        self._height = height
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
        if self._pretty:
            links_sdf += self.pathPointToSdf(self._path[-1], link_id=i)
        
        sdf = f"""<?xml version="1.0" ?>
        <sdf version="1.4">
            <model name="path_{self._name}">
              {links_sdf}
            </model>
        </sdf>
        """
        return sdf

    def pathPointToSdf(self, pt: np.ndarray, **kwargs):
        # create a cylinder at the point

        if pt.size == 2:
            return ""
        
        link_id = kwargs.get('link_id', None)
        
        # create sphere sdf visualization strings
        material = self._black
        if self._pretty:
            material = self._blue

        waypoint_vis = f"""
          <visual name="waypoint_vis">
            <pose>0 0 0 0 {math.pi/2} {pt[2]}</pose>
            <geometry>
              <cylinder>
                <radius>{self._sphere_radius}</radius>
                <length>{self._sphere_radius*3}</length>
              </cylinder>
            </geometry>
            {material}
          </visual>
        """
        if link_id is not None:
            link_name = f'link_{link_id}'
            waypoint_vis = f"""
                <link name="{link_name}">
                  <pose>{pt[0]} {pt[1]} {self._height} 0 0 0</pose>
                  {waypoint_vis}
                </link>
                <joint name="fixed_joint_{link_id}" type="fixed">
                  <parent>world</parent>
                  <child>{link_name}</child>
                </joint>

            """
        return waypoint_vis
    
    def pathLineToSdf(self, start: np.ndarray, end: np.ndarray, **kwargs):
        # create a cyclinder that goes from the start point to the end point
        
        link_id = kwargs.get('link_id', None)

        # only use the linear components of the start and end points
        start_pt = start[0:2];
        end_pt = end[0:2]
        
        # Find the euler angles to transform a vertical cylinder to go from the
        # start to end point.

        line_vec = np.append(end_pt-start_pt, 0)
        line_len = np.linalg.norm(line_vec)

        # first calculate the angle-axis representation
        u_line = line_vec / np.linalg.norm(line_vec)
        u_z = np.array([0,0,1])
        rot_axis = np.cross(u_z, u_line)
        norm_cross = np.linalg.norm(rot_axis);
        angle = math.atan2(norm_cross, np.dot(u_z, u_line))

        # extrinsic rotations about x, y, z (in this order)
        euler = R.from_rotvec(angle * rot_axis).as_euler('xyz')
        
        material = self._black
        if self._pretty:
            material = self._green

        line_vis = f"""
          <visual name="vis_{link_id}">
            <pose>{line_vec[0]/2} {line_vec[1]/2} {line_vec[2]/2} {euler[0]} {euler[1]} {euler[2]}</pose>
            <geometry>
              <cylinder>
                <radius>{self._line_thickness/2}</radius>
                <length>{line_len}</length>
              </cylinder>
            </geometry>
            {material}
          </visual>
        """
        return line_vis
        

    def pathSegToSdf(self, start: np.ndarray, end: np.ndarray, id_str: int):
        # Create an SDF string of a two point path segment. A sphere is created
        # for the start point, and a cyclinder goes from the start to the end.

        # create line visualization strings
        waypoint_vis = ""
        if self._pretty:
            waypoint_vis = self.pathPointToSdf(start)
        line_vis = self.pathLineToSdf(start, end, link_id=id_str)

        link_name = f"link_{id_str}"
        link_sdf = f"""
        <link name="{link_name}">
          <pose>{start[0]} {start[1]} {self._height} 0 0 0</pose>
          {waypoint_vis}
          {line_vis}
        </link>
        <joint name="fixed_joint_{id_str}" type="fixed">
          <parent>world</parent>
          <child>{link_name}</child>
        </joint>
        """
        return link_sdf
