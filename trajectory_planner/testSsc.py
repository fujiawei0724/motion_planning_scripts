#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 下午2:37
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : testSsc.py
# @Software: PyCharm

"""
Test piecewise curves generation using ssc.
"""

import numpy as np
import matplotlib.pyplot as plt 
from cvxopt import matrix, solvers

if __name__ == '__main__':
    hessian = np.array([720.0, -1800.0, 1200.0, 0.0, 0.0, -120.0, -1800.0, 4800.0, -3600.0, 0.0, 600.0, 0.0, 1200.0, -3600.0, 3600.0, -1200.0, 0.0, 0.0, 0.0, 0.0, -1200.0, 3600.0, -3600.0, 1200.0, 0.0, 600.0, 0.0, -3600.0, 4800.0, -1800.0, -120.0, 0.0, 0.0, 1200.0, -1800.0, 720.0]).reshape((6, 6))
    time_span = 0.8
    start_constraint = [0.0, 5.0, 0.0]
    end_constraint = [5.0, 6.0, 0.0]

    a_array = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                  [-5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, -5.0, 5.0],
                  [20.0, -40.0, 20.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 20.0, -40.0, 20.0]])
    
    b_array = np.array([start_constraint[0], end_constraint[0], start_constraint[1] * time_span, end_constraint[1] * time_span, start_constraint[2] * time_span, end_constraint[2] * time_span]).reshape((-1, 1))

    # Supple data
    P = matrix(hessian * time_span ** -3.0)
    q = matrix(np.zeros(6, ))
    A = matrix(a_array)
    b = matrix(b_array)

    res = solvers.qp(P, q, None, None, A, b)

    optimized_val = np.array(res['x']).reshape((1, -1))

    print(optimized_val)



