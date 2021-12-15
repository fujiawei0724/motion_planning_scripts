#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 下午6:44
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : calculateNearestPoint.py
# @Software: PyCharm

"""
Calculate the analytical solution of nearest point.
"""

import numpy as np
import matplotlib.pyplot as plt

# Start point and end point parameters
class CurvePoint:
    def __init__(self, x, y, theta, curvature):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta
        self.curvature_ = curvature

# Quintic spline parameters
class QParameters:
    def __init__(self, ax, bx, cx, dx, ex, fx, ay, by, cy, dy, ey, fy):
        self.ax_ = ax
        self.bx_ = bx
        self.cx_ = cx
        self.dx_ = dx
        self.ex_ = ex
        self.fx_ = fx
        self.ay_ = ay
        self.by_ = by
        self.cy_ = cy
        self.dy_ = dy
        self.ey_ = ey
        self.fy_ = fy
    
    def print(self):
        print('ax: {}, bx: {}, cx: {}, dx: {}, ex: {}, fx: {}'.format(self.ax_, self.bx_, self.cx_, self.dx_, self.ex_, self.fx_))
        print('ay: {}, by: {}, cy: {}, dy: {}, ey: {}, fy: {}'.format(self.ay_, self.by_, self.cy_, self.dy_, self.ey_, self.fy_))


class QuinticSplineUtils:
    # Calculate parameters due to mileage constraint
    @staticmethod
    def getParametersByMileageConstraint(begin_state, end_state, mileage_constraint):
        l = mileage_constraint
        p0x = begin_state.x_
        p0y = begin_state.y_
        t0x = np.cos(begin_state.theta_)
        t0y = np.sin(begin_state.theta_)
        k0x = -begin_state.curvature_ * np.sin(begin_state.theta_)
        k0y = begin_state.curvature_ * np.cos(begin_state.theta_)
        p1x = end_state.x_
        p1y = end_state.y_
        t1x = np.cos(end_state.theta_)
        t1y = np.sin(end_state.theta_)
        k1x = -end_state.curvature_ * np.sin(end_state.theta_)
        k1y = end_state.curvature_ * np.cos(end_state.theta_)

        fx = p0x
        ex = t0x * l
        dx = k0y * l * l / 2.0
        cx = (10.0 * p1x - 10.0 * p0x) + (-4.0 * t1x - 6 * t0x) * l + (k1x - 3.0 * k0x) * l * l / 2.0
        bx = (-15.0 * p1x + 15.0 * p0x) + (7.0 * t1x + 8.0 * t0x) * l + (-2.0 * k1x + 3.0 * k0x) * l * l / 2.0
        ax = (6.0 * p1x - 6.0 * p0x) + (-3.0 * t1x - 3.0 * t0x) * l + (k1x - k0x) * l * l / 2.0
        
        fy = p0y
        ey = t0y * l
        dy = k0x * l * l / 2.0
        cy = (10.0 * p1y - 10.0 * p0y) + (-4.0 * t1y - 6 * t0y) * l + (k1y - 3.0 * k0y) * l * l / 2.0
        by = (-15.0 * p1y + 15.0 * p0y) + (7.0 * t1y + 8.0 * t0y) * l + (-2.0 * k1y + 3.0 * k0y) * l * l / 2.0
        ay = (6.0 * p1y - 6.0 * p0y) + (-3.0 * t1y - 3.0 * t0y) * l + (k1y - k0y) * l * l / 2.0

        return QParameters(ax, bx, cx, dx, ex, fx, ay, by, cy, dy, ey, fy)

# Generate path from parameters
class QuinticSplineGenerator:
    def __init__(self, points_num):
        self.points_num_ = points_num
        self.ets_ = np.zeros((points_num, 5))
        for i in range(0, points_num):
            for j in range(0, 5):
                self.ets_[i][j] = (i / (points_num - 1)) ** (j + 1.0)

    def getPath(self, params):
        x, y = [], []
        for i in range(0, self.points_num_):
            x.append(params.ax_ * self.ets_[i][4] + params.bx_ * self.ets_[i][3] + params.cx_ * self.ets_[i][2] + params.dx_ * self.ets_[i][1] + params.ex_ * self.ets_[i][0] + params.fx_)
            y.append(params.ay_ * self.ets_[i][4] + params.by_ * self.ets_[i][3] + params.cy_ * self.ets_[i][2] + params.dy_ * self.ets_[i][1] + params.ey_ * self.ets_[i][0] + params.fy_)
        return x, y

if __name__ == '__main__':
    start_state = CurvePoint(0.0, 5.0, 0.0, 0.0)
    end_state = CurvePoint(5.0, 10.0, -1.0, 0.0)
    dis = np.sqrt((start_state.x_ - end_state.x_) ** 2.0 + (start_state.y_ - end_state.y_) ** 2.0)
    params = QuinticSplineUtils.getParametersByMileageConstraint(start_state, end_state, dis)
    params.print()
    points_num = 10
    quintic_spline_generator = QuinticSplineGenerator(points_num)
    x, y = quintic_spline_generator.getPath(params)

    plt.figure(0)
    plt.title('test')
    plt.plot(x, y, c='g', linewidth=1.0)
    plt.scatter(x, y, c='r', s=1.0)
    plt.show()


