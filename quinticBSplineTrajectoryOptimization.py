# -- coding: utf-8 --
# @Time : 2021/9/27 下午2:14
# @Author : fujiawei0724
# @File : quinticBSplineTrajectoryOptimization.py
# @Software: PyCharm

"""
This code is used for optimizing a quintic B-spline based trajectory in 3D dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bezierSplineTrajectory import BSplineTrajectory
from quinticBSplineOptimization import OptimizationTools as optimizer


class QuinticBSplineOptimizer:
    def __init__(self, initial_trajectory_scatter_points):
        # Trajectory point segmentation
        s_projection = initial_trajectory_scatter_points[:, 0]
        d_projection = initial_trajectory_scatter_points[:, 1]
        t_projection = initial_trajectory_scatter_points[:, 2]

        # Construct projection in s-t and d-t surface
        s_t_projection = np.vstack((t_projection, s_projection)).T
        d_t_projection = np.vstack((t_projection, d_projection)).T

        # Optimization in two dimensions
        self.s_t_optimized_control_points_ = self.projectionPointsOptimization(s_t_projection)
        self.d_t_optimized_control_points_ = self.projectionPointsOptimization(d_t_projection)

    # Quintic B-spline optimization for projection control points
    def projectionPointsOptimization(self, projection_control_point):
        optimized_control_points = optimizer.quinticBSplineOptimization(projection_control_point)
        print(optimized_control_points)
        return optimized_control_points

    # Construct 3D trajectory from optimized projection points
    def generateOptimizedControlPoints(self):
        # Get values in specified dimension
        optimized_s_projection = self.s_t_optimized_control_points_[:, 1]
        optimized_d_projection = self.d_t_optimized_control_points_[:, 1]
        assert self.s_t_optimized_control_points_[:, 0].all() == self.d_t_optimized_control_points_[:, 0].all()
        optimized_t_projection = self.s_t_optimized_control_points_[:, 0]

        # Construct optimized control points
        optimized_control_points = np.vstack((optimized_s_projection, optimized_d_projection, optimized_t_projection)).T

        return optimized_control_points

    # Construct optimized interpolated trajectory in 3D dimensions
    def generateOptimizedInterpolatedTrajectory(self):
        optimized_control_points = self.generateOptimizedControlPoints()

        # Optimized trajectory generation
        optimized_trajectory_generator = BSplineTrajectory()
        optimized_trajectory = optimized_trajectory_generator.trajectoryGeneration(optimized_control_points)

        return optimized_trajectory


if __name__ == '__main__':
    point_1 = np.array([0.0, 0.0, 0.0])
    point_2 = np.array([4.0, 5.5, 1.0])
    point_3 = np.array([10.0, 0.7, 2.0])
    point_4 = np.array([15.0, 0.8, 3.0])
    point_5 = np.array([22.0, 4.0, 4.0])
    point_6 = np.array([30.0, 1.0, 5.0])
    point_7 = np.array([35.0, 1.5, 6.0])
    point_8 = np.array([37.0, 2.5, 7.0])
    point_9 = np.array([38.0, 3.0, 8.0])

    # Construct scatter points
    initial_trajectory_scatter_points = np.vstack((point_1, point_2, point_3, point_4, point_5, point_6, point_7, point_8, point_9))

    # Initial trajectory generation
    initial_trajectory_generator = BSplineTrajectory()
    initial_trajectory = initial_trajectory_generator.trajectoryGeneration(initial_trajectory_scatter_points)

    # Optimized quintic B-spline
    quintic_b_spline_optimizer = QuinticBSplineOptimizer(initial_trajectory_scatter_points)
    optimized_trajectory_scatter_points = quintic_b_spline_optimizer.generateOptimizedControlPoints()
    optimized_trajectory = quintic_b_spline_optimizer.generateOptimizedInterpolatedTrajectory()

    # Visualization
    fig = plt.figure()
    ax_1 = Axes3D(fig)
    ax_1.plot3D(initial_trajectory[:, 0], initial_trajectory[:, 1], initial_trajectory[:, 2], 'gray', label='Initial trajectory')
    ax_1.scatter3D(initial_trajectory_scatter_points[:, 0], initial_trajectory_scatter_points[:, 1], initial_trajectory_scatter_points[:, 2], cmap='Blues', label='Initial control points')
    ax_1.plot3D(optimized_trajectory[:, 0], optimized_trajectory[:, 1], optimized_trajectory[:, 2], 'green', label='Optimized trajectory')
    ax_1.scatter3D(optimized_trajectory_scatter_points[:, 0], optimized_trajectory_scatter_points[:, 1], optimized_trajectory_scatter_points[:, 2], cmap='rainbow', label='Initial control points')
    ax_1.set_zlabel('time')
    ax_1.set_ylabel('d')
    ax_1.set_xlabel('s')
    plt.legend()
    plt.show()