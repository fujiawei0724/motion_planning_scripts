# -- coding: utf-8 --
# @Time : 2021/9/25 上午11:18
# @Author : fujiawei0724
# @File : bezierCurve.py
# @Software: PyCharm

"""
This code generates bezier curve using specified control points.
"""

import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-7

class QuinticBezierCurve:
    def __init__(self, path_scatters):
        assert path_scatters.shape == (6, 2)
        self.path_scatters_ = path_scatters

    def getPoints(self, t):
        return self.path_scatters_[0] * (1 - t) ** 5.0 + self.path_scatters_[1] * 5.0 * t * (1 - t) ** 4.0 + self.path_scatters_[2] * 10.0 * t ** 2.0 * (1 - t) ** 3.0 + self.path_scatters_[3] * 10.0 * t ** 3.0 * (1 - t) ** 2.0 + self.path_scatters_[4] * 5.0 * t ** 4.0 * (1 - t) + self.path_scatters_[5] * t ** 5.0

    def generateInterpolatedPath(self, sample_num):
        samples = np.linspace(0, 1, sample_num)
        path = []
        for sample in samples:
            path.append(self.getPoints(sample))
        return np.array(path)

if __name__ == '__main__':
    path_scatters = np.array([[0.0, 0.0],
                              [1.0, 5.0],
                              [2.0, 6.0],
                              [3.0, 15.0],
                              [4.0, 10.0],
                              # [6.0, 8.0],
                              # [7.0, 10.0],
                              # [8.0, 12.0],
                              # [9.0, 15.0],
                              [5.0, 16.0]])
    # Generate interpolated path
    quintic_bezier_curve = QuinticBezierCurve(path_scatters[:])
    interpolated_path = quintic_bezier_curve.generateInterpolatedPath(100)

    # Visualization
    plt.figure(0, (12, 5))
    plt.scatter(path_scatters[:, 0], path_scatters[:, 1], c='r', s=2.0)
    plt.plot(interpolated_path[:, 0], interpolated_path[:, 1], c='g', linewidth=0.5)

    # Test projection gap
    m_samples = np.arange(0.0, 1.0 + EPS, 0.01)
    t_samples = []
    for m_sample in m_samples:
        t_samples.append(quintic_bezier_curve.getPoints(m_sample)[0])
    t_samples = np.array(t_samples)

    multiple_rate = path_scatters[-1][0] - path_scatters[0][0]
    print('m samples: {}'.format(m_samples * multiple_rate))
    print('t samples: {}'.format(t_samples))

    plt.show()

