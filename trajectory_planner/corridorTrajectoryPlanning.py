# -- coding: utf-8 --
# @Time : 2021/11/5 下午7:58
# @Author : fujiawei0724
# @File : corridorTrajectoryPlanning.py
# @Software: PyCharm

"""
This code contains the trajectory planning method in the constraints of corridors.
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from bezierSplineTrajectory import trajectory_generator
from drivingCorridor import Cube

class Constraints:
    """
    Brief:
        The constraints for start point and end point
    Args:
        s means the longitudinal dimension, d means the lateral dimension, d_ denotes the first derivative of time, dd_ denotes the second derivative of time
    """
    def __init__(self, s, d_s, dd_s, d, d_d, dd_d):
        self.s_ = s
        self.d_s_ = d_s
        self.dd_s_ = dd_s
        self.d_ = d
        self.d_d_ = d_d
        self.dd_d_ = dd_d

class Point3D:
    def __init__(self, s, d, t):
        self.s_ = s
        self.d_ = d
        self.t_ = t

class SemanticCube(Cube):
    """
    Brief:
        Add more methods on the base of Cube
    """
    def __init__(self, s_start, s_end, d_start, d_end, t_start, t_end):
        super(SemanticCube, self).__init__(s_start, s_end, d_start, d_end, t_start, t_end)

    # Judge whether a point is in the cube (contain boundaries)
    def isInside(self, point: Point3D):
        if self.s_start_ <= point.s_ <= self.s_end_ and self.d_start_ <= point.d_ <= self.d_end_ and self.t_start_ <= point.t_ <= self.t_end_:
            return True
        return False





class CvxoptInterface:
    pass

class BSplineOptimizer:
    """
    Brief:
        Optimize the parameters of b-spline based trajectory
    Arg:
        cubes: the cubes need to constrain the positions of scatter points in the interpolation
        ref_stamps: the time stamps of the points in interpolation
            Note that the dense seed path points' detailed information only used in corridor generation, in the optimization, the reference time stamps of seed path points are enough
        start_constraint: start point's constraint
        end_constraint: end point's constraint
    Returns:
        The optimized scatter points' information (s, d, t)
    """
    def __init__(self):
        # Initialize shell
        self.start_constraints_ = None
        self.end_constraints_ = None
        self.semantic_cubes_ = None
        self.ref_stamps_ = None

    def load(self, start_constraints: Constraints, end_constraints: Constraints, semantic_cubes, ref_stamps):
        pass


if __name__ == '__main__':
    pass





