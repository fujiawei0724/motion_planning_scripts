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
import scipy
import matplotlib.pyplot as plt
from verifyBSplineProjectionPrecision import QuinticBSpline

EPS = 1e-7

class QuinticBSplineOptimizer:
    def __init__(self, initial_control_points):
        assert initial_control_points.shape[1] == 2
        self.initial_control_points_ = copy.deepcopy(initial_control_points)
        self.segment_num_ = initial_control_points.shape[0] - 1

        """
        t means trajectory's time dimension,
        m denotes the coefficients of quintic B-spline,
        ordinate means the s or l value.
        """
        self.t_start_ = initial_control_points[0][1]
        self.t_end_ = initial_control_points[-1][1]
        self.m_start_ = 0.0
        self.m_end_ = self.segment_num_

        # Generate initial quintic B-spline
        self.initial_quintic_b_spline_ = QuinticBSpline(initial_control_points)
        self.initial_quintic_interpolated_path_ = self.initial_quintic_b_spline_.generateInterpolatedPath(0.01)

        # Get Hessian matrix
        self.Hessian_matrix_ = np.array([[1.0 / 10.0, -1.0 / 12.0, -1.0 / 3.0, 1.0 / 2.0, -1.0 / 6.0, -1.0 / 60.0],
                                         [-1.0 / 12.0, 1.0 / 2.0, -5.0 / 6.0, 1.0 / 3.0, 1.0 / 4.0, -1.0 / 6.0],
                                         [-1.0 / 3.0, -5.0 / 6.0, 4.0, -11.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0],
                                         [1.0 / 2.0, 1.0 / 3.0, -11.0 / 3.0, 4.0, -5.0 / 6.0, -1.0 / 3.0],
                                         [-1.0 / 6.0, 1.0 / 4.0, 1.0 / 3.0, -5.0 / 6.0, 1.0 / 2.0, -1.0 / 12.0],
                                         [-1.0 / 60.0, -1.0 / 6.0, 1.0 / 2.0, -1.0 / 3.0, -1.0 / 12.0, 1.0 / 10.0]])


    # Calculate optimization objective function of the whole quintic B-spline
    def calcObjectiveFunction(self):
        # Firstly, calculate the cut-off points of t and m
        m_cut_off_points = np.arange(self.m_start_, self.m_end_ + EPS, 1.0)
        t_cut_off_points = []
        for sample_m in m_cut_off_points:
            t_cut_off_points.append(self.initial_quintic_b_spline_.generateScatterPoint(sample_m)[0])
        t_cut_off_points = np.array(t_cut_off_points)

        # print("m cut-off points: {}".format(m_cut_off_points))
        # print("t cut-off points: {}".format(t_cut_off_points))
        # print("m cut-off points: {}".format(np.diff(m_cut_off_points)))
        # print("t cut-off points: {}".format(np.diff(t_cut_off_points)))

        # Secondly, calculate objective cost for each segment

    # Calculate optimization objective function of a given segment
    def calcSegmentObjectiveFunction(self, segment_index):
        # Generate segment control points
        all_control_points = self.initial_quintic_b_spline_.points_
        segment_control_points = all_control_points[segment_index:segment_index + 6]

        # Calculate time span (normalization coefficient)
        segment_t_start = self.calcStartTime(segment_control_points)
        segment_t_end = self.calcEndTime(segment_control_points)
        time_span = segment_t_end - segment_t_start

        # Determine variables
        varaibles = segment_control_points[:, 1]
        varaibles = varaibles.reshape(1, 6)
        varaibles_T = varaibles.T

        return np.linalg.multi_dot([varaibles, self.Hessian_matrix_, varaibles_T])


    # Calculate start time of a specified segment
    def calcStartTime(self, segment_control_points):
        assert segment_control_points.shape == (6, 2)
        return (1.0 / 120.0) * segment_control_points[0][0] + (26.0 / 120.0) * segment_control_points[1][0] + (33.0 / 60.0) * segment_control_points[2][0] + (13.0 / 60.0) * segment_control_points[3][0] + (1.0 / 120.0) * segment_control_points[4][0]

    # Calculate end time of a specified segment
    def calcEndTime(self, segment_control_points):
        assert segment_control_points.shape == (6, 2)
        return (1.0 / 120.0) * segment_control_points[1][0] + (13.0 / 60.0) * segment_control_points[2][0] + (33.0 / 60.0) * segment_control_points[3][0] + (26.0 / 120.0) * segment_control_points[4][0] + (1.0 / 120.0) * segment_control_points[5][0]





    # Optimize process
    def optimize(self):
        pass



if __name__ == '__main__':
    path_scatters = np.array([[0.0, 0.0],
                              [1.0, 5.0],
                              [2.0, 6.0],
                              [3.0, 15.0],
                              [4.0, 10.0],
                              [5.0, 15.0],
                              [6.0, 8.0],
                              [7.0, 10.0],
                              [8.0, 12.0],
                              [9.0, 15.0],
                              [10.0, 16.0]])

    quintic_b_spline_optimizier = QuinticBSplineOptimizer(path_scatters)
    quintic_b_spline_optimizier.calcObjectiveFunction()
