# -- coding: utf-8 --
# @Time : 2021/10/2 下午8:43
# @Author : fujiawei0724
# @File : testCVXOPT.py
# @Software: PyCharm

"""
This code is used for testing the capability of CVXOPT, which is a optimization library in python.
"""

from cvxopt import solvers, matrix
import numpy as np

if __name__ == '__main__':
    P = matrix(2.0 * np.array([[1., 0.5, 0.],
                  [0.5, 4., 0.5],
                  [0., 0.5, 5]]))
    q = matrix(np.zeros(3, ))
    G = matrix(np.array([[1., 1., 1.],
                         [-1., 2., 0.],
                         [-1., 0., 0.],
                         [0., -1., 0.],
                         [0., 4., 0.]]))
    h = matrix(np.array([7., 4., 0., 0., 4.]))

    res = solvers.qp(P, q, G, h, None, None)

    print(res['x'])


