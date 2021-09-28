# -- coding: utf-8 --
# @Time : 2021/9/24 下午8:34
# @Author : fujiawei0724
# @File : quinticBSplineOptimization.py
# @Software: PyCharm

"""
This code is used to simplify the optimization objective function, calculate Hessian matrix and optimize quintic B-spline.
"""

import copy
import numpy as np
np.set_printoptions(threshold=float('inf'))
# from scipy.optimize import minimize
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from verifyBSplineProjectionPrecision import QuinticBSpline

EPS = 1e-7

class OptimizationTools:
    # Initialize Hessian matrix and P matrix for single segment
    Hessian_matrix = np.array([[1.0 / 10.0, -1.0 / 12.0, -1.0 / 3.0, 1.0 / 2.0, -1.0 / 6.0, -1.0 / 60.0],
                                         [-1.0 / 12.0, 1.0 / 2.0, -5.0 / 6.0, 1.0 / 3.0, 1.0 / 4.0, -1.0 / 6.0],
                                         [-1.0 / 3.0, -5.0 / 6.0, 4.0, -11.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0],
                                         [1.0 / 2.0, 1.0 / 3.0, -11.0 / 3.0, 4.0, -5.0 / 6.0, -1.0 / 3.0],
                                         [-1.0 / 6.0, 1.0 / 4.0, 1.0 / 3.0, -5.0 / 6.0, 1.0 / 2.0, -1.0 / 12.0],
                                         [-1.0 / 60.0, -1.0 / 6.0, 1.0 / 2.0, -1.0 / 3.0, -1.0 / 12.0, 1.0 / 10.0]])
    single_P_matrix = 0.5 * Hessian_matrix


    # Solve P matrix for quadratic optimization
    @staticmethod
    def calcPMatrix(all_control_points):
        points_num = all_control_points.shape[0]

        # Initialize P matrix
        P = np.zeros((points_num, points_num))

        # Calculate segment number
        segment_num = points_num - 5

        # Construct item for P matrix
        for i in range(0, segment_num):
            segment_control_points = all_control_points[i:i + 6]

            # Calculate time span
            t_start = OptimizationTools.calcStartTime(segment_control_points)
            t_end = OptimizationTools.calcEndTime(segment_control_points)
            time_span = t_end - t_start

            # Construct P matrix
            # TODO: check "time_span ** (-3)" or "time_span ** (-5)"
            P[i:i + 6, i:i + 6] += OptimizationTools.single_P_matrix * (time_span ** (-3))

        return matrix(P)

    # Solve A matrix for quadratic optimization
    @staticmethod
    def calcAbMatrix(all_control_points):
        assert all_control_points.shape[0] > 6
        points_num = all_control_points.shape[0]

        # Store start point position and end point position
        start_point_value = all_control_points[2][1]
        end_point_value = all_control_points[points_num - 3][1]

        # Initialize A matrix
        A = np.zeros((6, points_num))
        b = np.zeros((6, ))

        # Added points constrain conditions
        A[0][0], A[0][2], A[0][4] = 1.0, -2.0, 1.0
        A[1][1], A[1][2], A[1][3] = 1.0, -2.0, 1.0
        A[2][points_num - 1], A[2][points_num - 3], A[2][points_num - 5] = 1.0, -2.0, 1.0
        A[3][points_num - 2], A[3][points_num - 3], A[3][points_num - 4] = 1.0, -2.0, 1.0

        # Start point and end point position constraint conditions
        A[4][2], A[5][points_num - 3] = 1.0, 1.0
        b[4], b[5] = start_point_value, end_point_value

        """
        Some problem happen when add the velocity and acceleration constraint condition, maybe the reason is that the 
        equality constraint conditions are redundant.
        In cpp, if nlopt does not care the problem, the optimization process will be feasible.
        """
        # # Start point velocity and acceleration constraint conditions (velocity and acceleration values are for test)
        # A[6][0], A[6][1], A[6][3], A[6][4] = -1.0 / 24.0, -5.0 / 12.0, 5.0 / 12.0, 1.0 / 24.0
        # b[6] = -5.0
        # A[7][0], A[7][1], A[7][2], A[7][3], A[7][4] = 1.0 / 6.0, 1.0 / 3.0, -1.0, 1.0 / 3.0, 1.0 / 6.0
        # b[7] = -1.0

        # # End point velocity and acceleration constraint conditions (velocity and acceleration values are for test)
        # A[8][points_num - 5], A[8][points_num - 4], A[8][points_num - 2], A[8][points_num - 1] = -1.0 / 24.0, -5.0 / 12.0, 5.0 / 12.0, 1.0 / 24.0
        # b[8] = 5.0
        # A[9][points_num - 5], A[9][points_num - 4], A[9][points_num - 3], A[9][points_num - 2], A[9][points_num - 1] = 1.0 / 6.0, 1.0 / 3.0, -1.0, 1.0 / 3.0, 1.0 / 6.0
        # b[9] = 1.0


        return matrix(A), matrix(b)

    # Calculate start time of a specified segment
    @staticmethod
    def calcStartTime(segment_control_points):
        assert segment_control_points.shape == (6, 2)
        return (1.0 / 120.0) * segment_control_points[0][0] + (26.0 / 120.0) * segment_control_points[1][0] + (33.0 / 60.0) * segment_control_points[2][0] + (13.0 / 60.0) * segment_control_points[3][0] + (1.0 / 120.0) * segment_control_points[4][0]

    # Calculate end time of a specified segment
    @staticmethod
    def calcEndTime(segment_control_points):
        assert segment_control_points.shape == (6, 2)
        return (1.0 / 120.0) * segment_control_points[1][0] + (13.0 / 60.0) * segment_control_points[2][0] + (33.0 / 60.0) * segment_control_points[3][0] + (26.0 / 120.0) * segment_control_points[4][0] + (1.0 / 120.0) * segment_control_points[5][0]

    # Optimization function
    @staticmethod
    def quinticBSplineOptimization(path_scatters):
        # Construct all control points
        initial_quintic_b_spline = QuinticBSpline(path_scatters)
        all_control_points = initial_quintic_b_spline.points_

        # Generate optimization objective function
        P = OptimizationTools.calcPMatrix(all_control_points)
        q = matrix(np.zeros((all_control_points.shape[0],)))

        # Determine unequal constrain conditions
        # G = matrix(np.zeros((1, all_control_points.shape[0])))
        # h = matrix(np.zeros((1, )))

        # Determine equal constrain condition
        A, b = OptimizationTools.calcAbMatrix(all_control_points)

        # Solve quadratic optimization problem
        res = solvers.qp(P, q, None, None, A, b)

        # print(res['x'])

        # Construct optimized control points
        optimized_y = np.array(res['x'][2:-2]).reshape((1, -1))
        time_scatter_points = path_scatters[:, 0]
        optimized_control_points = np.vstack((time_scatter_points, optimized_y)).T

        return optimized_control_points


if __name__ == '__main__':
    path_scatters = np.array([[0.0, 0.0],
                              [1.0, 5.0],
                              [2.0, 6.0],
                              # [3.0, 15.0],
                              # [4.0, 10.0],
                              # [5.0, 15.0],
                              # [6.0, 8.0],
                              # [7.0, 10.0],
                              [8.0, 12.0],
                              [9.0, 15.0],
                              [10.0, 16.0]])

    optimized_control_points = OptimizationTools.quinticBSplineOptimization(path_scatters)

    # Construct initial and optimized quintic B-spline
    initial_quintic_b_spline = QuinticBSpline(path_scatters)
    initial_quintic_b_spline_path = initial_quintic_b_spline.generateInterpolatedPath(0.01)
    optimized_quintic_B_spline = QuinticBSpline(optimized_control_points)
    optimized_quintic_B_spline_path = optimized_quintic_B_spline.generateInterpolatedPath(0.01)


    # # Output
    # print("Initial control points: {}".format(path_scatters))
    # print("Optimized control points: {}".format(optimized_control_points))

    # Visualization
    plt.figure(0, (12, 5))
    plt.title("Quintic B-spline optimization")
    plt.plot(initial_quintic_b_spline_path[:, 0], initial_quintic_b_spline_path[:, 1], c='r', linewidth=1.0, label='Initial path')
    plt.scatter(path_scatters[:, 0], path_scatters[:, 1], c='r', s=5.0, label='Initial control points')
    plt.plot(optimized_quintic_B_spline_path[:, 0], optimized_quintic_B_spline_path[:, 1], c='g', linewidth=1.0, label='Optimized path')
    plt.scatter(optimized_control_points[:, 0], optimized_control_points[:, 1], c='g', s=5.0, label='Optimized control points')
    plt.legend()
    plt.show()




