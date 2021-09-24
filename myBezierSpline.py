#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/21 下午10:07
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : myBezierSpline.py
# @Software: PyCharm

"""
This code includes construction of bezier spline without external libraries.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy


class CubicBSpline:
    def __init__(self, path):
        assert path.shape[1] == 2
        points = np.zeros((path.shape[0] + 2, 2))
        for i in range(0, len(path)):
            points[i + 1] = path[i]
        point_num = points.shape[0]
        assert point_num > 3

        # Additional execution for ensure the position of start point and end point
        points[0][0] = 2.0 * path[0][0] - path[1][0]
        points[0][1] = 2.0 * path[0][1] - path[1][1]
        points[point_num - 1][0] = 2.0 * path[-1][0] - path[-2][0]
        points[point_num - 1][1] = 2.0 * path[-1][1] - path[-2][1]


        # Determine segments number
        self.segment_num_ = point_num - 3
        self.x_coefficients_ = np.zeros((self.segment_num_, 4))
        self.y_coefficients_ = np.zeros((self.segment_num_, 4))

        # Determine coefficients
        for i in range(0, point_num - 3):
            self.x_coefficients_[i][0] = 1.0 / 6.0 * (points[i][0] + 4.0 * points[i + 1][0] + points[i + 2][0])
            self.x_coefficients_[i][1] = -0.5 * (points[i][0] - points[i + 2][0])
            self.x_coefficients_[i][2] = 0.5 * (points[i][0] - 2.0 * points[i + 1][0] + points[i + 2][0])
            self.x_coefficients_[i][3] = -1.0 / 6.0 * (points[i][0] - 3.0 * points[i + 1][0] + 3.0 * points[i + 2][0] - points[i + 3][0])
            self.y_coefficients_[i][0] = 1.0 / 6.0 * (points[i][1] + 4.0 * points[i + 1][1] + points[i + 2][1])
            self.y_coefficients_[i][1] = -0.5 * (points[i][1] - points[i + 2][1])
            self.y_coefficients_[i][2] = 0.5 * (points[i][1] - 2.0 * points[i + 1][1] + points[i + 2][1])
            self.y_coefficients_[i][3] = -1.0 / 6.0 * (points[i][1] - 3.0 * points[i + 1][1] + 3.0 * points[i + 2][1] - points[i + 3][1])

    # Transform input
    def inputVerify(self, u):
        if u < 0.0:
            return 0.0
        elif u > self.segment_num_:
            return self.segment_num_
        else:
            return u

    # Generate input information
    def getSegmentInfo(self, u):
        u = self.inputVerify(u)
        for i in range(0, self.segment_num_):
            if u < i + 1:
                remain = u - i
                return i, remain
        return self.segment_num_ - 1, 1.0

    # Calculate x position
    def xValue(self, u):
        u = self.inputVerify(u)
        index, u = self.getSegmentInfo(u)
        return self.x_coefficients_[index][0] + self.x_coefficients_[index][1] * u + self.x_coefficients_[index][2] * u * u + self.x_coefficients_[index][3] * u * u * u

    # Calculate y positon
    def yValue(self, u):
        u = self.inputVerify(u)
        index, u = self.getSegmentInfo(u)
        return self.y_coefficients_[index][0] + self.y_coefficients_[index][1] * u + self.y_coefficients_[index][2] * u * u + self.y_coefficients_[index][3] * u * u * u

    # Generate interpolated path
    def generateInterpolatedPath(self, sample_gap):
        print(self.x_coefficients_)
        print(self.y_coefficients_)
        samples = np.linspace(0.0, self.segment_num_, int(self.segment_num_ / sample_gap))
        path = []
        for sample_value in samples:
            x_position = self.xValue(sample_value)
            y_position = self.yValue(sample_value)
            path.append([x_position, y_position])
        return np.array(path)

class QuinticBSpline:
    def __init__(self, path):
        assert path.shape[1] == 2
        
        points = copy.deepcopy(path)
        # TODO: add additional process to approximate start point and end point
        
        self.segment_num_ = points.shape[0] - 5
        self.x_coefficients_ = np.zeros((self.segment_num_, 6))
        self.y_coefficients_ = np.zeros((self.segment_num_, 6))
        
        # Calculate coefficients
        for i in range(0, self.segment_num_):
            self.x_coefficients_[i][0] = (1.0 / 120.0) * points[i][0] + (26.0 / 120.0) * points[i + 1][0] + (33.0 / 60.0) * points[i + 2][0] + (13.0 / 60.0) * points[i + 3][0] + (1.0 / 120.0) * points[i + 4][0]
            self.x_coefficients_[i][1] = (-5.0 / 120.0) * points[i][0] + (-50.0 / 120.0) * points[i + 1][0] + (25.0 / 60.0) * points[i + 3][0] + (5.0 / 120.0) * points[i + 4][0]
            self.x_coefficients_[i][2] = (10.0 / 120.0) * points[i][0] + (20.0 / 120.0) * points[i + 1][0] + (-30.0 / 60.0) * points[i + 2][0] + (10.0 / 60.0) * points[i + 3][0] + (10.0 / 120.0) * points[i + 4][0]
            self.x_coefficients_[i][3] = (-10.0 / 120.0) * points[i][0] + (20.0 / 120.0) * points[i + 1][0] + (-10.0 / 60.0) * points[i + 3][0] + (10.0 / 120.0) * points[i + 4][0]
            self.x_coefficients_[i][4] = (5.0 / 120.0) * points[i][0] + (-20.0 / 120.0) * points[i + 1][0] + (15.0 / 60.0) * points[i + 2][0] + (-10.0 / 60.0) * points[i + 3][0] + (5.0 / 120.0) * points[i + 4][0]
            self.x_coefficients_[i][5] = (-1.0 / 120.0) * points[i][0] + (5.0 / 120.0) * points[i + 1][0] + (-5.0 / 60.0) * points[i + 2][0] + (5.0 / 60.0) * points[i + 3][0] + (-5.0 / 120.0) * points[i + 4][0] + (1.0 / 120.0) * points[i + 5][0]
            self.y_coefficients_[i][0] = (1.0 / 120.0) * points[i][1] + (26.0 / 120.0) * points[i + 1][1] + (33.0 / 60.0) * points[i + 2][1] + (13.0 / 60.0) * points[i + 3][1] + (1.0 / 120.0) * points[i + 4][1]
            self.y_coefficients_[i][1] = (-5.0 / 120.0) * points[i][1] + (-50.0 / 120.0) * points[i + 1][1] + (25.0 / 60.0) * points[i + 3][1] + (5.0 / 120.0) * points[i + 4][1]
            self.y_coefficients_[i][2] = (10.0 / 120.0) * points[i][1] + (20.0 / 120.0) * points[i + 1][1] + (-30.0 / 60.0) * points[i + 2][1] + (10.0 / 60.0) * points[i + 3][1] + (10.0 / 120.0) * points[i + 4][1]
            self.y_coefficients_[i][3] = (-10.0 / 120.0) * points[i][1] + (20.0 / 120.0) * points[i + 1][1] + (-10.0 / 60.0) * points[i + 3][1] + (10.0 / 120.0) * points[i + 4][1]
            self.y_coefficients_[i][4] = (5.0 / 120.0) * points[i][1] + (-20.0 / 120.0) * points[i + 1][1] + (15.0 / 60.0) * points[i + 2][1] + (-10.0 / 60.0) * points[i + 3][1] + (5.0 / 120.0) * points[i + 4][1]
            self.y_coefficients_[i][5] = (-1.0 / 120.0) * points[i][1] + (5.0 / 120.0) * points[i + 1][1] + (-5.0 / 60.0) * points[i + 2][1] + (5.0 / 60.0) * points[i + 3][1] + (-5.0 / 120.0) * points[i + 4][1] + (1.0 / 120.0) * points[i + 5][1]

    # Transform input
    def inputVerify(self, u):
        if u < 0.0:
            return 0.0
        elif u > self.segment_num_:
            return self.segment_num_
        else:
            return u

    # Generate input information
    def getSegmentInfo(self, u):
        u = self.inputVerify(u)
        for i in range(0, self.segment_num_):
            if u < i + 1:
                remain = u - i
                return i, remain
        return self.segment_num_ - 1, 1.0

    # Calculate x position
    def xValue(self, u):
        u = self.inputVerify(u)
        index, u = self.getSegmentInfo(u)
        return self.x_coefficients_[index][0] + self.x_coefficients_[index][1] * u + self.x_coefficients_[index][2] * u * u + self.x_coefficients_[index][3] * u * u * u + self.x_coefficients_[index][4] * u * u * u * u + self.x_coefficients_[index][5] * u * u * u * u * u

    # Calculate y positon
    def yValue(self, u):
        u = self.inputVerify(u)
        index, u = self.getSegmentInfo(u)
        return self.y_coefficients_[index][0] + self.y_coefficients_[index][1] * u + self.y_coefficients_[index][2] * u * u + self.y_coefficients_[index][3] * u * u * u + self.y_coefficients_[index][4] * u * u * u * u + self.y_coefficients_[index][5] * u * u * u * u * u

    # Generate interpolated path
    def generateInterpolatedPath(self, sample_gap):
        print(self.x_coefficients_)
        print(self.y_coefficients_)
        samples = np.linspace(0.0, self.segment_num_, int(self.segment_num_ / sample_gap))
        path = []
        for sample_value in samples:
            x_position = self.xValue(sample_value)
            y_position = self.yValue(sample_value)
            path.append([x_position, y_position])
        return np.array(path)
        
        
        





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
    cubic_b_spline = CubicBSpline(path_scatters)
    quintic_b_spline = QuinticBSpline(path_scatters)

    # Generate interpolated path
    cubic_interpolated_path = cubic_b_spline.generateInterpolatedPath(0.01)
    quintic_interpolated_path = quintic_b_spline.generateInterpolatedPath(0.01)


    # Visualization
    plt.figure(0, (12, 5))
    plt.scatter(path_scatters[:, 0], path_scatters[:, 1], c='g', s=5.0)
    plt.plot(cubic_interpolated_path[:, 0], cubic_interpolated_path[:, 1], c='r', linewidth=0.5, label='cubic b-spline')
    plt.plot(quintic_interpolated_path[:, 0], quintic_interpolated_path[:, 1], c='b', linewidth=0.5, label='quintic b-spline')
    plt.legend()
    plt.show()



