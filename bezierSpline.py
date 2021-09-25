#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 下午10:07
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : bezierSpline.py
# @Software: PyCharm

"""
This code generates b spline trajectory in 3D dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from mpl_toolkits.mplot3d import Axes3D
from myBezierSpline import QuinticBSpline

class B_Spline:
    def __init__(self):
        pass

    # Generate 3D trajectory
    def trajectoryGeneration(self, scatter_points):
        # point[0] means s dimension, point[1] means d dimension, point[2] means t dimension
        assert len(scatter_points[0]) == 3
        s_projection = scatter_points[:, 0]
        d_projection = scatter_points[:, 1]
        t_projection = scatter_points[:, 2]

        s_t_projection_points = np.vstack((s_projection, t_projection)).T
        d_t_projection_points = np.vstack((d_projection, t_projection)).T

        # 2D interpolation
        # s_t_interpolation = self.BSplineInterpolation(s_t_projection_points)
        # d_t_interpolation = self.BSplineInterpolation(d_t_projection_points)

        s_t_interpolation = self.myQuinticBSpline(s_t_projection_points)
        d_t_interpolation = self.myQuinticBSpline(d_t_projection_points)
        assert(s_t_interpolation[:, 0].all() == d_t_interpolation[:, 0].all())

        # Construct 3D trajectory
        s_dimension = s_t_interpolation[:, 0]
        d_dimension = d_t_interpolation[:, 0]
        t_dimension = s_t_interpolation[:, 1]
        trajectory_points = np.vstack((s_dimension, d_dimension, t_dimension)).T

        return trajectory_points



    # 2D b-spline interpolation use scipy
    def BSplineInterpolation(self, project_points):
        assert len(project_points[0]) == 2
        x = project_points[:, 1]
        y = project_points[:, 0]
        t, c, k = interpolate.splrep(x, y, s=0, k=5)
        b_spline = interpolate.BSpline(t, c, k, extrapolate=False)
        x_min, x_max = min(x), max(x)
        xx = np.linspace(x_min, x_max, 100)
        return np.vstack((xx, b_spline(xx))).T

    # Quintic B-spline without external libraries
    def myQuinticBSpline(self, project_points):
        assert len(project_points[0]) == 2
        quintic_b_spline = QuinticBSpline(project_points)
        return quintic_b_spline.generateInterpolatedPath(0.01)




if __name__ == '__main__':
    point_1 = np.array([0.0, 0.0, 0.0])
    point_2 = np.array([4.0, 0.5, 1.0])
    point_3 = np.array([10.0, 0.7, 2.0])
    point_4 = np.array([15.0, 0.8, 3.0])
    point_5 = np.array([22.0, 0.9, 4.0])
    point_6 = np.array([30.0, 1.0, 5.0])


    # Construct scatter points
    trajectory_scatter_points = np.vstack((point_1, point_2, point_3, point_4, point_5, point_6))

    # Trajectory generation
    trajectory_generator = B_Spline()
    trajectory = trajectory_generator.trajectoryGeneration(trajectory_scatter_points)

    trajectory_scatter_points = trajectory_scatter_points.T
    # Visualization
    fig = plt.figure()
    ax_1 = Axes3D(fig)
    ax_1.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'gray')
    ax_1.scatter3D(trajectory_scatter_points[0, :], trajectory_scatter_points[1, :], trajectory_scatter_points[2, :], cmap='Blues')
    ax_1.set_zlabel('time')
    ax_1.set_ylabel('d')
    ax_1.set_xlabel('s')
    plt.show()


