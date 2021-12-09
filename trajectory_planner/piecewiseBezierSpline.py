#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 下午5:38
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : piecewiseBezierSpline.py
# @Software: PyCharm

"""
Generation of the piecewise bezier spline.
"""

import numpy as np
import matplotlib.pyplot as plt

class Point2f:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y

class Point3f:
    def __init__(self, s, d, t):
        self.s_ = s
        self.d_ = d
        self.t_ = t

# 3d trajectory generation
class BezierPiecewiseCurve:
    def __init__(self, s, d, ref_stamps):
        assert(len(s) == len(d))
        assert((len(ref_stamps) - 1) * 5 + 1 == len(s))
        self.ref_stamps_ = ref_stamps
        self.segment_num_ = len(ref_stamps) - 1
        self.s_coefficients_ = np.zeros((self.segment_num_, 6))
        self.d_coefficients_ = np.zeros((self.segment_num_, 6))
        for i in range(0, self.segment_num_):
            start_influenced_index = i * 5
            self.s_coefficients_[i][0] = s[start_influenced_index]
            self.s_coefficients_[i][1] = s[start_influenced_index + 1]
            self.s_coefficients_[i][2] = s[start_influenced_index + 2]
            self.s_coefficients_[i][3] = s[start_influenced_index + 3]
            self.s_coefficients_[i][4] = s[start_influenced_index + 4]
            self.s_coefficients_[i][5] = s[start_influenced_index + 5]
            self.d_coefficients_[i][0] = d[start_influenced_index]
            self.d_coefficients_[i][1] = d[start_influenced_index + 1]
            self.d_coefficients_[i][2] = d[start_influenced_index + 2]
            self.d_coefficients_[i][3] = d[start_influenced_index + 3]
            self.d_coefficients_[i][4] = d[start_influenced_index + 4]
            self.d_coefficients_[i][5] = d[start_influenced_index + 5]

    def generateTraj(self, sample_gap=0.01):
        traj = []
        for segment_index in range(self.segment_num_):
            time_span = self.ref_stamps_[segment_index + 1] - self.ref_stamps_[segment_index]
            sample_num = int(time_span / sample_gap)
            if segment_index == self.segment_num_ - 1:
                segment_seeds = np.linspace(0.0, 1.0, sample_num, endpoint=True)
            else:
                segment_seeds = np.linspace(0.0, 1.0, sample_num, endpoint=False)

            for current_seed in segment_seeds:
                time_stamp = self.ref_stamps_[segment_index] + (time_span * current_seed)
                traj.append(self.generatePoint(segment_index, current_seed, time_stamp))
        return traj


    def generatePoint(self, segment_index, remain, time_stamp):
        s_value = self.s_coefficients_[segment_index][0] * pow(1.0 - remain, 5) + 5.0 * self.s_coefficients_[segment_index][1] * remain * pow(1.0 - remain, 4) + 10.0 * self.s_coefficients_[segment_index][2] * pow(remain, 2) * pow(1.0 - remain, 3) + 10.0 * self.s_coefficients_[segment_index][3] * pow(remain, 3) * pow(1.0 - remain, 2) + 5.0 * self.s_coefficients_[segment_index][4] * pow(remain, 4) * (1.0 - remain) + self.s_coefficients_[segment_index][5] * pow(remain, 5)
        d_value = self.d_coefficients_[segment_index][0] * pow(1.0 - remain, 5) + 5.0 * self.d_coefficients_[segment_index][1] * remain * pow(1.0 - remain, 4) + 10.0 * self.d_coefficients_[segment_index][2] * pow(remain, 2) * pow(1.0 - remain, 3) + 10.0 * self.d_coefficients_[segment_index][3] * pow(remain, 3) * pow(1.0 - remain, 2) + 5.0 * self.d_coefficients_[segment_index][4] * pow(remain, 4) * (1.0 - remain) + self.d_coefficients_[segment_index][5] * pow(remain, 5)

        return Point3f(s_value, d_value, time_stamp)

# 2d trajectory generation
class BezierPiecewiseCurve2d:
    def __init__(self, x, ref_stamps):
        assert((len(ref_stamps) - 1) * 6 == len(x))
        self.ref_stamps_ = ref_stamps
        self.segment_num_ = len(ref_stamps) - 1
        self.coefficients_ = np.zeros((self.segment_num_, 6))
        for i in range(0, self.segment_num_):
            start_influenced_index = i * 6
            self.coefficients_[i][0] = x[start_influenced_index]
            self.coefficients_[i][1] = x[start_influenced_index + 1]
            self.coefficients_[i][2] = x[start_influenced_index + 2]
            self.coefficients_[i][3] = x[start_influenced_index + 3]
            self.coefficients_[i][4] = x[start_influenced_index + 4]
            self.coefficients_[i][5] = x[start_influenced_index + 5]

    def generateTraj(self, sample_gap=0.01):
        traj = []
        for segment_index in range(self.segment_num_):
            time_span = self.ref_stamps_[segment_index + 1] - self.ref_stamps_[segment_index]
            sample_num = int(time_span / sample_gap)
            if segment_index == self.segment_num_ - 1:
                segment_seeds = np.linspace(0.0, 1.0, sample_num, endpoint=True)
            else:
                segment_seeds = np.linspace(0.0, 1.0, sample_num, endpoint=False)

            for current_seed in segment_seeds:
                time_stamp = self.ref_stamps_[segment_index] + (time_span * current_seed)
                traj.append(self.generatePoint(segment_index, current_seed, time_stamp))
        return traj

    def generatePoint(self, segment_index, remain, time_stamp):
        value = self.coefficients_[segment_index][0] * pow(1.0 - remain, 5) + 5.0 * self.coefficients_[segment_index][1] * remain * pow(1.0 - remain, 4) + 10.0 * self.coefficients_[segment_index][2] * pow(remain, 2) * pow(1.0 - remain, 3) + 10.0 * self.coefficients_[segment_index][3] * pow(remain, 3) * pow(1.0 - remain, 2) + 5.0 * self.coefficients_[segment_index][4] * pow(remain, 4) * (1.0 - remain) + self.coefficients_[segment_index][5] * pow(remain, 5)

        return Point2f(value, time_stamp)

if __name__ == '__main__':
    # s = [-0.000075, 2.397838, 4.795751, 10.168802, 5.729700, 13.999925, 21.639526, 30.999925, 30.999925, 15.799338, 30.999925, 42.849428, 55.147995, 35.175672, 37.570232, 39.964353]
    # d = [3.192407, 3.092339, 2.992272, -9.900000, -6.187500, 9.900000, 1.650000, -9.900000, -9.900000, -4.233759, 8.046125, 4.870806, 2.769850, 1.944588, 1.782969, 1.615077]
    # t = [0.000000, 1.200000, 2.800000, 4.000000]
    #
    # bezier_piecewise_curve = BezierPiecewiseCurve(s, d, t)
    # points = bezier_piecewise_curve.generateTraj()
    # interpolated_s, interpolated_d, interpolated_t = [], [], []
    # for point in points:
    #     interpolated_s.append(point.s_)
    #     interpolated_d.append(point.d_)
    #     interpolated_t.append(point.t_)

    s_gene = [-0.041166, 2.352154, 4.745475, 4.646715, 3.695750, 3.094776, 3.094776, 2.293478, 1.958834, 12.629113, 10.458834, 10.458834, 10.458834, 10.458834, 12.086543, 35.119937, 37.517110, 39.914047]
    s_test = [0, 2.39856, 4.79712, 7.19615, 9.59557, 11.9951, 11.9951, 15.1945, 18.3941, 21.5934, 24.7916, 27.9884, 27.9884, 30.3859, 32.7824, 35.1776, 37.5716, 39.9651]
    t = [0.0, 1.2, 2.8, 4.0]

    bezier_piecewise_curve_2d = BezierPiecewiseCurve2d(s_test, t)
    points = bezier_piecewise_curve_2d.generateTraj()
    interpolated_s, interpolated_t = [], []
    for point in points:
        interpolated_s.append(point.x_)
        interpolated_t.append(point.y_)
    plt.figure(0)
    plt.plot(interpolated_t, interpolated_s, c='r', linewidth=1.0)
    plt.show()

