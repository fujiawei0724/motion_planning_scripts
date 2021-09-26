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
import scipy
import matplotlib.pyplot as plt
from verifyBSplineProjectionPrecision import QuinticBSpline


class QuinticBSplineOptimizer:
    def __init__(self, initial_control_points):
        assert initial_control_points.shape[1] == 2
        self.initial_control_points_ = copy.deepcopy(initial_control_points)

    # Generate quintic B-spline using control points
    def calcQuinticBspline(self, control_points):
        assert control_points.shape[1] == 2 and control_points.shape[0] >= 2
        quintic_b_spline = QuinticBSpline(control_points)
        quintic_interpolated_path = quintic_b_spline.generateInterpolatedPath(0.01)
        return quintic_interpolated_path

    # Calculate optimization objective function of quintic B-spline
    def calcObjectiveFunction(self, interpolated_path):
        """
        t means trajectory's time dimension,
        m denotes the coefficients of quintic B-spline,
        ordinate means the s or l value.
        """
        # Firstly, calculate the cut-off points of t and m
        

    # Optimize process
    def optimize(self):
        pass



if __name__ == '__main__':
    path_scatters = np.array([[0.0, 0.0],
                              [1.0, 5.0],
                              [2.0, 6.0],
                              [3.0, 15.0],
                              [4.0, 10.0],
                              [6.0, 8.0],
                              [7.0, 10.0],
                              [8.0, 12.0],
                              [9.0, 15.0],
                              [10.0, 16.0]])

    # Generate quintic B-spline
    quintic_b_spline = QuinticBSpline(path_scatters)
    quintic_interpolated_path = quintic_b_spline.generateInterpolatedPath(0.01)

    # Visualization quintic B-spline
    plt.figure(0, (12, 5))
    plt.scatter(path_scatters[:, 0], path_scatters[:, 1], c='r', s=10.0)
    plt.plot(quintic_interpolated_path[:, 0], quintic_interpolated_path[:, 1], c='b', linewidth=1.0)
    plt.show()
