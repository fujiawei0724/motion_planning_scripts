# -- coding: utf-8 --
# @Time : 2021/11/4 下午5:31
# @Author : fujiawei0724
# @File : drivingCorridor.py
# @Software: PyCharm

"""
This code is responsible for generating the driving corridors (several cubes).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon

class Cube:
    def __init__(self, x_start, x_end, y_start, y_end, z_start, z_end):
        assert x_end > x_start and y_end > y_start and z_end > z_start
        self.s_start_ = x_start
        self.s_end_ = x_end
        self.d_start_ = y_start
        self.d_end_ = y_end
        self.t_start_ = z_start
        self.t_end_ = z_end

class Tools:
    @staticmethod
    def calculateRanges(cubes):
        x_range = [float('inf'), -float('inf')]
        y_range = [float('inf'), -float('inf')]
        z_range = [float('inf'), -float('inf')]
        for cube in cubes:
            assert isinstance(cube, Cube)
            x_range[0] = min(cube.s_start_, x_range[0])
            x_range[1] = max(cube.s_end_, x_range[1])
            y_range[0] = min(cube.d_start_, y_range[0])
            y_range[1] = max(cube.d_end_, y_range[1])
            z_range[0] = min(cube.t_start_, z_range[0])
            z_range[1] = max(cube.t_end_, z_range[1])
        return x_range, y_range, z_range



class Visualization:
    @staticmethod
    def visualizationCube(cube: Cube, ax, color='red'):
        x, dx, y, dy, z, dz = cube.s_start_, cube.s_end_ - cube.s_start_, cube.d_start_, cube.d_end_ - cube.d_start_, cube.t_start_, cube.t_end_ - cube.t_start_
        assert isinstance(ax, Axes3D)
        xx = [x, x, x + dx, x + dx, x]
        yy = [y, y + dy, y + dy, y, y]
        kwargs = {'alpha': 1, 'color': color}
        ax.plot3D(xx, yy, [z] * 5, **kwargs)
        ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
        ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
        ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
        ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
        ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)

    @staticmethod
    def visualizationCorridors(cubes, ax, color='red'):
        for cube in cubes:
            Visualization.visualizationCube(cube, ax, color)



if __name__ == '__main__':
    # An example of lane change behavior
    cube_1 = Cube(0.0, 12.0, -2.0, 2.0, 0.0, 2.0)
    cube_2 = Cube(2.0, 14.0, -2.0, 2.0, 0.4, 2.4)
    cube_3 = Cube(4.0, 16.0, -2.0, 2.0, 0.8, 2.8)
    cube_4 = Cube(6.0, 18.0, -2.0, 2.0, 1.2, 3.2)
    cube_5 = Cube(8.0, 20.0, -4.5, 2.0, 1.6, 3.6)
    cube_6 = Cube(10.0, 22.0, -4.5, 2.0, 2.0, 4.0)
    cube_7 = Cube(12.0, 24.0, -4.5, 2.0, 2.4, 4.4)
    cube_8 = Cube(14.0, 26.0, -4.5, 2.0, 2.8, 4.8)
    cube_9 = Cube(16.0, 28.0, -4.5, 2.0, 3.2, 5.2)
    cube_10 = Cube(20.0, 30.0, -4.5, 2.0, 3.6, 5.6)
    corridor = [cube_1, cube_2, cube_3, cube_4, cube_5, cube_6, cube_7, cube_8, cube_9, cube_10]

    x_range, y_range, z_range = Tools.calculateRanges(corridor)
    fig = plt.figure(0)
    ax = Axes3D(fig)
    Visualization.visualizationCorridors(corridor, ax)
    ax.set_xlabel('s')
    ax.set_ylabel('d')
    ax.set_zlabel('t')
    ax.set_box_aspect(aspect=(x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0]))
    plt.show()