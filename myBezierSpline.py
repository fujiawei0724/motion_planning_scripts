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
        assert(path.shape == (-1, 2))
        points = np.zeros((path.shape[0] + 2, 2))
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
            self.x_coefficients_[i][0] = 1.0 / 6.0 * (points[i].x_ + 4.0 * points[i + 1].x_ + points[i + 2].x_)
            self.x_coefficients_[i][1] = -0.5 * (points[i].x_ - points[i + 2].x_)
            self.x_coefficients_[i][2] = 0.5 * (points[i].x_ - 2.0 * points[i + 1].x_ + points[i + 2].x_)
            self.x_coefficients_[i][3] = -1.0 / 6.0 * (points[i].x_ - 3.0 * points[i + 1].x_ + 3.0 * points[i + 2].x_ - points[i + 3].x_)
            self.y_coefficients_[i][0] = 1.0 / 6.0 * (points[i].y_ + 4.0 * points[i + 1].y_ + points[i + 2].y_)
            self.y_coefficients_[i][1] = -0.5 * (points[i].y_ - points[i + 2].y_)
            self.y_coefficients_[i][2] = 0.5 * (points[i].y_ - 2.0 * points[i + 1].y_ + points[i + 2].y_)
            self.y_coefficients_[i][3] = -1.0 / 6.0 * (points[i].y_ - 3.0 * points[i + 1].y_ + 3.0 * points[i + 2].y_ - points[i + 3].y_)

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
        return self.segment_num_, 1.0

    # Calculate x position
    def xValue(self, u):
        u = self.inputVerify(u)
        u, index = self.getSegmentInfo(u)
        return self.x_coefficients_[index][0] + self.x_coefficients_[index][1] * u + self.x_coefficients_[index][2] * u * u + self.x_coefficients_[index][3] * u * u * u

    # Calculate y positon
    def yValue(self, u):
        u = self.inputVerify(u)
        u, index = self.getSegmentInfo(u)
        return self.y_coefficients_[index][0] + self.y_coefficients_[index][1] * u + self.y_coefficients_[index][2] * u * u + self.y_coefficients_[index][3] * u * u * u



if __name__ == '__main__':
    





