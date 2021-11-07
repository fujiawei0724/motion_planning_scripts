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
import copy
from cvxopt import solvers, matrix
from enum import Enum, unique
from mpl_toolkits.mplot3d import Axes3D

from bezierSplineTrajectory import BSplineTrajectory
from drivingCorridor import Cube
from drivingCorridor import Visualization
from drivingCorridor import Tools


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
        if s_end <= s_start or d_end <= d_start:
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


# class Point3D:
#     def __init__(self, s, d, t):
#         self.s_ = s
#         self.d_ = d
#         self.t_ = t


class SemanticCube(Cube):
    """
    Brief:
        Add more methods on the base of Cube.
    """

    def __init__(self, s_start, s_end, d_start, d_end, t_start, t_end):
        super(SemanticCube, self).__init__(s_start, s_end, d_start, d_end, t_start, t_end)

    # # Judge whether a point is in the cube (contain boundaries)
    # def isInside(self, point: Point3D):
    #     if self.s_start_ <= point.s_ <= self.s_end_ and self.d_start_ <= point.d_ <= self.d_end_ and self.t_start_ <= point.t_ <= self.t_end_:
    #         return True
    #     return False


class OptimizationTools:
    Hessian_matrix = np.array([[1.0 / 10.0, -1.0 / 12.0, -1.0 / 3.0, 1.0 / 2.0, -1.0 / 6.0, -1.0 / 60.0],
                               [-1.0 / 12.0, 1.0 / 2.0, -5.0 / 6.0, 1.0 / 3.0, 1.0 / 4.0, -1.0 / 6.0],
                               [-1.0 / 3.0, -5.0 / 6.0, 4.0, -11.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0],
                               [1.0 / 2.0, 1.0 / 3.0, -11.0 / 3.0, 4.0, -5.0 / 6.0, -1.0 / 3.0],
                               [-1.0 / 6.0, 1.0 / 4.0, 1.0 / 3.0, -5.0 / 6.0, 1.0 / 2.0, -1.0 / 12.0],
                               [-1.0 / 60.0, -1.0 / 6.0, 1.0 / 2.0, -1.0 / 3.0, -1.0 / 12.0, 1.0 / 10.0]])

    @staticmethod
    def calculateStartTime(segment_ref_stamps):
        return (1.0 / 120.0) * segment_ref_stamps[0] + (26.0 / 120.0) * segment_ref_stamps[1] + (33.0 / 60.0) * \
               segment_ref_stamps[2] + (13.0 / 60.0) * segment_ref_stamps[3] + (1.0 / 120.0) * segment_ref_stamps[4]

    @staticmethod
    def calculateEndTime(segment_ref_stamps):
        return (1.0 / 120.0) * segment_ref_stamps[1] + (13.0 / 60.0) * segment_ref_stamps[2] + (33.0 / 60.0) * \
               segment_ref_stamps[3] + (26.0 / 120.0) * segment_ref_stamps[4] + (1.0 / 120.0) * segment_ref_stamps[5]

    @staticmethod
    def calculateTimeSpan(segment_ref_stamps):
        return OptimizationTools.calculateEndTime(segment_ref_stamps) - OptimizationTools.calculateStartTime(
            segment_ref_stamps)


class CvxoptInterface:
    """
    Brief:
        Solve 2 dimensions quadratic programming problem using cvxopt.
    """

    def __init__(self):
        self.all_ref_stamps_ = None
        self.start_constraint_ = None
        self.end_constraint_ = None
        self.unequal_constraints_ = None

    # Load the real data
    def load(self, all_ref_stamps, start_constraint, end_constraint, unequal_constraints):
        self.all_ref_stamps_ = all_ref_stamps
        self.start_constraint_ = start_constraint
        self.end_constraint_ = end_constraint
        self.unequal_constraints_ = unequal_constraints

    # Run optimization
    def runOnce(self):
        points_num = len(self.all_ref_stamps_)

        # Determine P and q matrix (objective function)
        P = self.calculatePMatrix()
        q = matrix(np.zeros((points_num, )))

        # Determine G and h matrix (unequal constraints)
        G, h = self.calculateGhMatrix()

        # Determine A and b matrix (equal constraints)
        A, b = self.calculateAbMatrix()

        res = solvers.qp(P, q, G, h, A, b)

        # Construct final result
        optimized_y = np.array(res['x'][2:-2]).reshape((1, -1))
        # ref_stamps = np.array(self.all_ref_stamps_[2:-2])
        # optimized_points2d = np.vstack((ref_stamps, optimized_y)).T

        return optimized_y

    # Calculate P matrix
    def calculatePMatrix(self):
        points_num = len(self.all_ref_stamps_)

        # Initialize P matrix
        P = np.zeros((points_num, points_num))

        # Calculate segment number
        segment_number = points_num - 5

        # Calculate matrix P iteratively
        for i in range(0, segment_number):
            # Calculate time span
            segment_reference_stamps = self.all_ref_stamps_[i:i + 6]
            time_span = OptimizationTools.calculateTimeSpan(segment_reference_stamps)

            # TODO: check "time_span ** (-3)" or "time_span ** (-5)"
            P[i:i + 6, i:i + 6] += OptimizationTools.Hessian_matrix * (time_span ** (-3))

        return matrix(P)

    # Calculate A matrix and b matrix
    def calculateAbMatrix(self):
        points_num = len(self.all_ref_stamps_)

        # Initialize A matrix
        A = np.zeros((8, points_num))
        b = np.zeros((8,))

        # Added points constrain conditions
        A[0][0], A[0][2], A[0][4] = 1.0, -2.0, 1.0
        A[1][1], A[1][2], A[1][3] = 1.0, -2.0, 1.0
        A[2][points_num - 1], A[2][points_num - 3], A[2][points_num - 5] = 1.0, -2.0, 1.0
        A[3][points_num - 2], A[3][points_num - 3], A[3][points_num - 4] = 1.0, -2.0, 1.0

        # Start point and end point position constraint conditions
        A[4][2], A[5][points_num - 3] = 1.0, 1.0
        b[4], b[5] = self.start_constraint_[0], self.end_constraint_[0]

        # Start point and end point velocity constraint conditions
        A[6][0], A[6][1], A[6][3], A[6][4] = -1.0 / 24.0, -5.0 / 12.0, 5.0 / 12.0, 1.0 / 24.0
        b[6] = self.start_constraint_[1]
        A[7][points_num - 5], A[7][points_num - 4], A[7][points_num - 2], A[7][points_num - 1] = -1.0 / 24.0, -5.0 / 12.0, 5.0 / 12.0, 1.0 / 24.0
        b[7] = self.end_constraint_[1]

        # TODO: for quintic B-spline, the acceleration of the start point and point must be set to zero, add an algorithm to handle this problem.

        return matrix(A), matrix(b)

    # Calculate G matrix and h matrix
    def calculateGhMatrix(self):
        # Calculate the shape of matrix
        points_num = len(self.all_ref_stamps_)
        assert points_num - 6 == len(self.unequal_constraints_)
        unequal_constraints_num = len(self.unequal_constraints_) * 2

        # Initialize matrix
        G = np.zeros((unequal_constraints_num, points_num))
        h = np.zeros((unequal_constraints_num, ))

        # Fill matrix data
        # TODO: check the correctness of the matrix index
        for i, unequal_constraint in enumerate(self.unequal_constraints_):
            G[i*2][i+3] = 1
            h[i*2] = unequal_constraint[1]
            G[i*2+1][i+3] = -1
            h[i*2+1] = -unequal_constraint[0]

        return matrix(G), matrix(h)




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

        # ~Stage II: divide the problem into two dimensions and solute them respectively
        # Construct optimizer
        cvx_itf = CvxoptInterface()

        # Calculate longitudinal dimension
        s_start_constraints = self.start_constraints_.deintegrate(Dimension.s)
        s_end_constraints = self.end_constraints_.deintegrate(Dimension.s)
        s_unequal_constraints = unequal_constraints.deintegrate(Dimension.s)
        cvx_itf.load(all_ref_stamps, s_start_constraints, s_end_constraints, s_unequal_constraints)
        optimized_s = cvx_itf.runOnce()

        # Calculate latitudinal dimension
        d_start_constraints = self.start_constraints_.deintegrate(Dimension.d)
        d_end_constraints = self.end_constraints_.deintegrate(Dimension.d)
        d_unequal_constraints = unequal_constraints.deintegrate(Dimension.d)
        cvx_itf.load(all_ref_stamps, d_start_constraints, d_end_constraints, d_unequal_constraints)
        optimized_d = cvx_itf.runOnce()

        # ~Stage III: merge two series of 2D points to generate 3D points
        final_t_stamps = copy.deepcopy(self.ref_stamps_)
        points_data = np.vstack((optimized_s, optimized_d, final_t_stamps))
        points = points_data.T

        return points

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
                print(
                    'ref stamp: {}, construct unequal constraint error, s_start: {}, s_end: {}, d_start: {}, d_end: {}'.format(
                        ref_stamp, max(s_low), min(s_up), max(d_low), min(d_up)))
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
                all_ref_stamps[i] = self.ref_stamps_[i - 2]
        return all_ref_stamps


if __name__ == '__main__':
    # Prepare data
    start_constraints = EqualConstraint(0., 5., 0., 0.5, 0., 0.)
    end_constraints = EqualConstraint(30., 5., 0., 3.5, 0., 0.)

    cube_1 = SemanticCube(0.0, 12.0, -2.0, 2.0, 0.0, 2.0)
    cube_2 = SemanticCube(2.0, 14.0, -2.0, 2.0, 0.4, 2.4)
    cube_3 = SemanticCube(4.0, 16.0, -2.0, 2.0, 0.8, 2.8)
    cube_4 = SemanticCube(6.0, 18.0, -2.0, 2.0, 1.2, 3.2)
    cube_5 = SemanticCube(8.0, 20.0, -4.5, 2.0, 1.6, 3.6)
    cube_6 = SemanticCube(10.0, 22.0, -4.5, 2.0, 2.0, 4.0)
    cube_7 = SemanticCube(12.0, 24.0, -4.5, 2.0, 2.4, 4.4)
    cube_8 = SemanticCube(14.0, 26.0, -4.5, 2.0, 2.8, 4.8)
    cube_9 = SemanticCube(16.0, 28.0, -4.5, 2.0, 3.2, 5.2)
    cube_10 = SemanticCube(20.0, 30.0, -4.5, 2.0, 3.6, 5.6)
    corridor = [cube_1, cube_2, cube_3, cube_4, cube_5, cube_6, cube_7, cube_8, cube_9, cube_10]

    ref_stamps = [0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 3.2, 3.6, 4.]

    # Fill data to optimizer
    b_spline_optimizer = BSplineOptimizer()
    b_spline_optimizer.load(start_constraints, end_constraints, corridor, ref_stamps)
    optimized_points3d = b_spline_optimizer.runOnce()

    # Generate trajectory
    trajectory_generator = BSplineTrajectory()
    trajectory = trajectory_generator.trajectoryGeneration(optimized_points3d)

    # Visualization
    # Visualization of cubes
    x_range, y_range, z_range = Tools.calculateRanges(corridor)
    fig = plt.figure(0)
    ax = Axes3D(fig)
    Visualization.visualizationCorridors(corridor, ax)

    # Visualization trajectory
    ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'gray')
    ax.scatter3D(optimized_points3d[:, 0], optimized_points3d[:, 1], optimized_points3d[:, 2], cmap='Blues')
    ax.set_zlabel('time')
    ax.set_ylabel('d')
    ax.set_xlabel('s')
    plt.show()

