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
import copy
from bezierSplineTrajectory import trajectory_generator
from drivingCorridor import Cube


class EqualConstraint:
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


class UnequalConstraint:
    """
    Brief:
        The description os unequal constraint for a Point3D
    Args:
        Given a time stamp (comes from reference time stamps), its constraints could be implied by 4 position constraints.
            Note that for unequal constraints in this problem, only position constraints are taken in consideration
    """

    def __init__(self, s_start, s_end, d_start, d_end):
        self.s_start_ = s_start
        self.s_end_ = s_end
        self.d_start_ = d_start
        self.d_end_ = d_end


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

    # Load the initial data
    def load(self, start_constraints: EqualConstraint, end_constraints: EqualConstraint, semantic_cubes, ref_stamps):
        self.start_constraints_ = start_constraints
        self.end_constraints_ = end_constraints
        self.semantic_cubes_ = semantic_cubes
        self.ref_stamps_ = ref_stamps

    # Run optimization
    def runOnce(self):
        # ~Stage I: check, prepare, and supple data
        assert len(self.semantic_cubes_) == len(self.ref_stamps_) - 1 and len(self.ref_stamps_) >= 3
        # Add additional time stamps to approximate start point and end point
        all_ref_stamps = self.calculateAllRefStamps()

        # ~Stage I:

    # Convert the unequal constraints for each time stamps
    def calculateUnequalConstraints(self):
        unequal_constraints = []
        for i in range(1, len(self.ref_stamps_) - 1):
            pass

    # Merge unequal constraints using semantic cubes
    def generateUnequalConstraints(self, semantic_cubes, ref_stamp):
        pass

    # Add additional time stamps
    def calculateAllRefStamps(self):
        all_ref_stamps = [0. for _ in range(len(self.ref_stamps_) + 4)]
        add_stamp_1 = 2. * self.ref_stamps_[0] - self.ref_stamps_[2]
        add_stamp_2 = 2. * self.ref_stamps_[0] - self.ref_stamps_[1]
        add_stamp_3 = 2. * self.ref_stamps_[-1] - self.ref_stamps_[-2]
        add_stamp_4 = 2. * self.ref_stamps_[-1] - self.ref_stamps_[-3]
        for i in range(len(all_ref_stamps)):
            if i == 0:
                all_ref_stamps[i] = add_stamp_1
            elif i == 1:
                all_ref_stamps[i] = add_stamp_2
            elif i == len(all_ref_stamps) - 2:
                all_ref_stamps[i] = add_stamp_3
            elif i == len(all_ref_stamps) - 1:
                all_ref_stamps[i] = add_stamp_4
            else:
                all_ref_stamps[i] = copy.copy(self.ref_stamps_[i - 2])
        return all_ref_stamps



if __name__ == '__main__':
    pass
