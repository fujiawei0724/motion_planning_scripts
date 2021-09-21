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
        points = np.zeros([path.shape[0] + 2, 2])
        point_num = points.shape[0]
        assert point_num > 3

        # Additional execution for ensure the position of start point and end point
        points[0][0] = 2.0 * path[0][0] - path[1][0]
        points[0][1] = 2.0 * path[0][1] - path[1][1]
        points[point_num - 1][0] = 2.0 * path[-1][0] - path[-2][0]
        points[point_num - 1][1] = 2.0 * path[-1][1] - path[-2][1]

        # Determine segments number
        self.segment_num_ = point_num - 3
        self.

if __name__ == '__main__':






