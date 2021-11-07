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
from enum import Enum, unique
from bezierSplineTrajectory import BSplineTrajectory
from drivingCorridor import Cube

@unique
class Dimension(Enum):
    s = 0
    d = 1

class EqualConstraint:
    """
    Brief:
        The constraints for start point and end point.
    Args:
        s means the longitudinal dimension, d means the lateral dimension, d_ denotes the first derivative of time, dd_ denotes the second derivative of time.
    """

    def __init__(self, s, d_s, dd_s, d, d_d, dd_d):
        self.s_ = s
        self.d_s_ = d_s
        self.dd_s_ = dd_s
        self.d_ = d
        self.d_d_ = d_d
        self.dd_d_ = dd_d

    def deintegrate(self, dimension_id: Dimension):
        if dimension_id == Dimension.s:
            return np.array([self.s_, self.d_s_, self.dd_s_])
        elif dimension_id == Dimension.d:
            return np.array([self.d_, self.d_d_, self.dd_d_])
        assert False


class UnequalConstraint:
    """
    Brief:
        The description os unequal constraint for a Point3D.
    Args:
        Given a time stamp (comes from reference time stamps), its constraints could be implied by 4 position constraints.
            Note that for unequal constraints in this problem, only position constraints are taken in consideration.
    """

    def __init__(self, s_start, s_end, d_start, d_end):
        if s_end <= s_start or d_start <= d_end:
            raise UnequalConstraintError('Unequal constraint does not exist.')
        self.s_start_ = s_start
        self.s_end_ = s_end
        self.d_start_ = d_start
        self.d_end_ = d_end

    def deintegrate(self, dimension_id: Dimension):
        if dimension_id == Dimension.s:
            return np.array([self.s_start_, self.s_end_])
        elif dimension_id == Dimension.d:
            return np.array([self.d_start_, self.d_end_])
        else:
            assert False

class UnequalConstraintSequence:
    def __init__(self, unequal_constraints):
        self.data_ = unequal_constraints

    def deintegrate(self, dimension_id: Dimension):
        constraints = []
        for unq_cons in self.data_:
            constraints.append(unq_cons.deintegrate(dimension_id))
        return np.array(constraints)



class UnequalConstraintError(Exception):
    pass


class Point3D:
    def __init__(self, s, d, t):
        self.s_ = s
        self.d_ = d
        self.t_ = t


class SemanticCube(Cube):
    """
    Brief:
        Add more methods on the base of Cube.
    """

    def __init__(self, s_start, s_end, d_start, d_end, t_start, t_end):
        super(SemanticCube, self).__init__(s_start, s_end, d_start, d_end, t_start, t_end)

    # Judge whether a point is in the cube (contain boundaries)
    def isInside(self, point: Point3D):
        if self.s_start_ <= point.s_ <= self.s_end_ and self.d_start_ <= point.d_ <= self.d_end_ and self.t_start_ <= point.t_ <= self.t_end_:
            return True
        return False


class CvxoptInterface:
    """
    Brief:
        Solve 2 dimensions quadratic programming problem using cvxopt.
    """
    def __init__(self):
        self.variables_num_ = None
        self.start_constraint_ = None
        self.end_constraint_ = None
        self.unequal_constraints_ = None

    # Load the real data
    def load(self, variables_num, start_constraint, end_constraint, unequal_constraints):
        self.variables_num_ = variables_num
        self.start_constraint_ = start_constraint
        self.end_constraint_ = end_constraint
        self.unequal_constraints_ = unequal_constraints

    # Run optimization
    def runOnce(self):
        pass



class BSplineOptimizer:
    """
    Brief:
        Optimize the parameters of b-spline based trajectory.
    Arg:
        cubes: the cubes need to constrain the positions of scatter points in the interpolation.
        ref_stamps: the time stamps of the points in interpolation.
            Note that the dense seed path points' detailed information only used in corridor generation, in the optimization, the reference time stamps of seed path points are enough.
        start_constraint: start point's constraint.
        end_constraint: end point's constraint.
    Returns:
        The optimized scatter points' information (s, d, t).
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
        # Calculate unequal constraints for intermediate Point3D based on their reference time stamps
        unequal_constraints = self.generateUnequalConstraints()

        # ~Stage I:

    # Merge unequal constraints using semantic cubes
    def generateUnequalConstraints(self):
        unequal_constraints = []
        for i, ref_stamp in enumerate(self.ref_stamps_):
            if i == 0 or i == len(self.ref_stamps_) - 1:
                # The start point and end point only have equal constraints
                continue

            # Calculate the first semantic cube affects the Point3D in reference stamp
            first_index = max(i - 5, 0)
            s_up, s_low, d_up, d_low = [], [], [], []
            for j in range(first_index, i + 1):
                s_up.append(self.semantic_cubes_[j].s_end_)
                s_low.append(self.semantic_cubes_[j].s_start_)
                d_up.append(self.semantic_cubes_[j].d_end_)
                d_low.append(self.semantic_cubes_[j].d_start_)
            cur_unequal_constraint = None
            try:
                cur_unequal_constraint = UnequalConstraint(max(s_low), min(s_up), max(d_low), min(d_up))
            except UnequalConstraintError:
                print('ref stamp: {}, construct unequal constraint error, s_start: {}, s_end: {}, d_start: {}, d_end: {}'.format(ref_stamp, max(s_low), min(s_up), max(d_low), min(d_up)))
                assert False
            unequal_constraints.append(cur_unequal_constraint)

        return UnequalConstraintSequence(unequal_constraints)

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
