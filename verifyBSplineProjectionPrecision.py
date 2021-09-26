# -- coding: utf-8 --
# @Time : 2021/9/25 下午6:11
# @Author : fujiawei0724
# @File : verifyBSplineProjectionPrecision.py
# @Software: PyCharm

"""
This code is used for verify the relation of the abscissa and the ordinate in quintic B-spline.
"""
import copy

import numpy as np
import matplotlib.pyplot as plt


class QuinticBSpline:

    def __init__(self, path):
        assert path.shape[1] == 2

        # points = copy.deepcopy(path)
        points = np.zeros((path.shape[0] + 4, 2))
        for i in range(0, len(path)):
            points[i + 2] = path[i]
        points[0] = 2.0 * points[2] - points[4]
        points[1] = 2.0 * points[2] - points[3]
        points[-2] = 2.0 * points[-3] - points[-4]
        points[-1] = 2.0 * points[-3] - points[-5]

        self.segment_num_ = points.shape[0] - 5
        self.x_coefficients_ = np.zeros((self.segment_num_, 6))
        self.y_coefficients_ = np.zeros((self.segment_num_, 6))

        # Calculate coefficients
        for i in range(0, self.segment_num_):
            self.x_coefficients_[i][0] = (1.0 / 120.0) * points[i][0] + (26.0 / 120.0) * points[i + 1][0] + (
                        33.0 / 60.0) * points[i + 2][0] + (13.0 / 60.0) * points[i + 3][0] + (1.0 / 120.0) * \
                                         points[i + 4][0]
            self.x_coefficients_[i][1] = (-5.0 / 120.0) * points[i][0] + (-50.0 / 120.0) * points[i + 1][0] + (
                        25.0 / 60.0) * points[i + 3][0] + (5.0 / 120.0) * points[i + 4][0]
            self.x_coefficients_[i][2] = (10.0 / 120.0) * points[i][0] + (20.0 / 120.0) * points[i + 1][0] + (
                        -30.0 / 60.0) * points[i + 2][0] + (10.0 / 60.0) * points[i + 3][0] + (10.0 / 120.0) * \
                                         points[i + 4][0]
            self.x_coefficients_[i][3] = (-10.0 / 120.0) * points[i][0] + (20.0 / 120.0) * points[i + 1][0] + (
                        -10.0 / 60.0) * points[i + 3][0] + (10.0 / 120.0) * points[i + 4][0]
            self.x_coefficients_[i][4] = (5.0 / 120.0) * points[i][0] + (-20.0 / 120.0) * points[i + 1][0] + (
                        15.0 / 60.0) * points[i + 2][0] + (-10.0 / 60.0) * points[i + 3][0] + (5.0 / 120.0) * \
                                         points[i + 4][0]
            self.x_coefficients_[i][5] = (-1.0 / 120.0) * points[i][0] + (5.0 / 120.0) * points[i + 1][0] + (
                        -5.0 / 60.0) * points[i + 2][0] + (5.0 / 60.0) * points[i + 3][0] + (-5.0 / 120.0) * \
                                         points[i + 4][0] + (1.0 / 120.0) * points[i + 5][0]
            self.y_coefficients_[i][0] = (1.0 / 120.0) * points[i][1] + (26.0 / 120.0) * points[i + 1][1] + (
                        33.0 / 60.0) * points[i + 2][1] + (13.0 / 60.0) * points[i + 3][1] + (1.0 / 120.0) * \
                                         points[i + 4][1]
            self.y_coefficients_[i][1] = (-5.0 / 120.0) * points[i][1] + (-50.0 / 120.0) * points[i + 1][1] + (
                        25.0 / 60.0) * points[i + 3][1] + (5.0 / 120.0) * points[i + 4][1]
            self.y_coefficients_[i][2] = (10.0 / 120.0) * points[i][1] + (20.0 / 120.0) * points[i + 1][1] + (
                        -30.0 / 60.0) * points[i + 2][1] + (10.0 / 60.0) * points[i + 3][1] + (10.0 / 120.0) * \
                                         points[i + 4][1]
            self.y_coefficients_[i][3] = (-10.0 / 120.0) * points[i][1] + (20.0 / 120.0) * points[i + 1][1] + (
                        -10.0 / 60.0) * points[i + 3][1] + (10.0 / 120.0) * points[i + 4][1]
            self.y_coefficients_[i][4] = (5.0 / 120.0) * points[i][1] + (-20.0 / 120.0) * points[i + 1][1] + (
                        15.0 / 60.0) * points[i + 2][1] + (-10.0 / 60.0) * points[i + 3][1] + (5.0 / 120.0) * \
                                         points[i + 4][1]
            self.y_coefficients_[i][5] = (-1.0 / 120.0) * points[i][1] + (5.0 / 120.0) * points[i + 1][1] + (
                        -5.0 / 60.0) * points[i + 2][1] + (5.0 / 60.0) * points[i + 3][1] + (-5.0 / 120.0) * \
                                         points[i + 4][1] + (1.0 / 120.0) * points[i + 5][1]

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
        return self.x_coefficients_[index][0] + self.x_coefficients_[index][1] * u + self.x_coefficients_[index][
            2] * u * u + self.x_coefficients_[index][3] * u * u * u + self.x_coefficients_[index][4] * u * u * u * u + \
               self.x_coefficients_[index][5] * u * u * u * u * u

    # Calculate y positon
    def yValue(self, u):
        u = self.inputVerify(u)
        index, u = self.getSegmentInfo(u)
        return self.y_coefficients_[index][0] + self.y_coefficients_[index][1] * u + self.y_coefficients_[index][
            2] * u * u + self.y_coefficients_[index][3] * u * u * u + self.y_coefficients_[index][4] * u * u * u * u + \
               self.y_coefficients_[index][5] * u * u * u * u * u

    # Generate interpolated path
    def generateInterpolatedPath(self, sample_gap):
        # print(self.x_coefficients_)
        # print(self.y_coefficients_)
        samples = np.linspace(0.0, self.segment_num_, int(self.segment_num_ / sample_gap))
        path = []
        for sample_value in samples:
            x_position = self.xValue(sample_value)
            y_position = self.yValue(sample_value)
            path.append([x_position, y_position, sample_value // 1.0])
        return np.array(path)

    # Generate scatter point
    def generateScatterPoint(self, m):
        assert 0.0 <= m <= self.segment_num_
        x_position = self.xValue(m)
        y_position = self.yValue(m)
        return np.array([x_position, y_position])

class Tools:

    # Transform time dimension to coefficient m
    @staticmethod
    def t_to_m(t_start, t_end, m_start, m_end, t_cur):
        assert t_end >= t_cur >= t_start
        rate = (t_cur - t_start) / (t_end - t_start)
        return m_start + rate * (m_end - m_start)

    # Transform time list to the list of coefficient n
    @staticmethod
    def transformProjections(t_start, t_end, m_start, m_end, sample_num):
        t_samples = np.linspace(t_start, t_end, sample_num)
        m_samples = []
        for t_sample in t_samples:
            m_samples.append(Tools.t_to_m(t_start, t_end, m_start, m_end, t_sample))
        return np.array(m_samples)

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
    # print(quintic_interpolated_path)

    # Test projection precision
    t_start, t_end = path_scatters[0][0], path_scatters[-1][0]
    m_start, m_end = 0.0, path_scatters.shape[0] - 1
    m_samples = Tools.transformProjections(t_start, t_end, m_start, m_end, 1000)

    # Visualization scatters
    plt.figure(0, (12, 5))
    plt.scatter(path_scatters[:, 0], path_scatters[:, 1], c='r', s=10.0, label='control point')

    # Visualization segments using different colors
    index = 0
    segments_num = int(max(quintic_interpolated_path[:, 2]))
    cut_off_points = []
    for i in range(0, segments_num + 1):
        start_index = index
        end_index = index
        for j in range(start_index, len(quintic_interpolated_path)):
            end_index = j
            if int(quintic_interpolated_path[j][2]) != i:
                if i < segments_num - 1:
                    cut_off_points.append(quintic_interpolated_path[j])
                break
        plt.plot(quintic_interpolated_path[start_index:end_index, 0], quintic_interpolated_path[start_index:end_index, 1], c=(0.5, i / segments_num, 0.5), linewidth=1.0, label='segment: {}'.format(i))
        index = end_index

    # Visualization cut-off points
    cut_off_points = np.array(cut_off_points)
    plt.scatter(cut_off_points[:, 0], cut_off_points[:, 1], c='k', s=10.0, label='cut-off point')
    plt.legend()
    plt.title('Quintic B-spline projection relation')
    plt.grid()


    # Output corresponding relation
    t_cut_off_samples = copy.deepcopy(cut_off_points[:, 0])
    t_cut_off_samples = np.insert(t_cut_off_samples, 0, path_scatters[0][0])
    t_cut_off_samples = np.append(t_cut_off_samples, path_scatters[-1][0])

    m_cut_off_samples = copy.deepcopy(cut_off_points[:, 2])
    m_cut_off_samples = np.insert(m_cut_off_samples, 0, 0.0)
    m_cut_off_samples = np.append(m_cut_off_samples, max(quintic_interpolated_path[:, 2]))

    print("t cut-off points: {}".format(t_cut_off_samples))
    print("m cut-off points: {}".format(m_cut_off_samples))
    print("t cut-off points diff: {}".format(np.diff(t_cut_off_samples)))
    print("m cut-off points diff: {}".format(np.diff(m_cut_off_samples)))

    plt.show()



