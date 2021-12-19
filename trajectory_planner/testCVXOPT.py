# -- coding: utf-8 --
# @Time : 2021/10/2 下午8:43
# @Author : fujiawei0724
# @File : testCVXOPT.py
# @Software: PyCharm

"""
This code is used for testing the capability of CVXOPT, which is a optimization library in python. These result are applied to compared with those alculated by CGAL to show their correctness,
"""

from cvxopt import solvers, matrix
import numpy as np
import copy

if __name__ == '__main__':
    # P = matrix(2.0 * np.array([[1., 0.5, 0.],
    #               [0.5, 4., 0.5],
    #               [0., 0.5, 5]]))
    # q = matrix(np.zeros(3, ))
    # G = matrix(np.array([[-1., -1., -1.],
    #                      [-1., 2., 0.],
    #                      [-1., 0., 0.],
    #                      [0., -1., 0.],
    #                      [0., 4., 0.]]))
    # h = matrix(np.array([-7., 4., 0., 0., 4.]))
    
    p_array = np.array([416.66666666667, -1041.6666666667, 694.44444444444, 0, 0, -69.444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -1041.6666666667, 2777.7777777778, -2083.3333333333, 0, 347.22222222222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            694.44444444444, -2083.3333333333, 2083.3333333333, -694.44444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, -694.44444444444, 2083.3333333333, -2083.3333333333, 694.44444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 347.22222222222, 0, -2083.3333333333, 2777.7777777778, -1041.6666666667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -69.444444444444, 0, 0, 694.44444444444, -1041.6666666667, 416.66666666667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 175.78125, -439.453125, 292.96875, 0, 0, -29.296875, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, -439.453125, 1171.875, -878.90625, 0, 146.484375, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 292.96875, -878.90625, 878.90625, -292.96875, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, -292.96875, 878.90625, -878.90625, 292.96875, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 146.484375, 0, -878.90625, 1171.875, -439.453125, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, -29.296875, 0, 0, 292.96875, -439.453125, 175.78125, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 416.66666666667, -1041.6666666667, 694.44444444444, 0, 0, -69.444444444444,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1041.6666666667, 2777.7777777778, -2083.3333333333, 0, 347.22222222222, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 694.44444444444, -2083.3333333333, 2083.3333333333, -694.44444444444, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -694.44444444444, 2083.3333333333, -2083.3333333333, 694.44444444444,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 347.22222222222, 0, -2083.3333333333, 2777.7777777778, -1041.6666666667,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -69.444444444444, 0, 0, 694.44444444444, -1041.6666666667, 416.66666666667]).reshape((18, 18))

    # print('p_array: {}'.format(p_array))

    P = matrix(p_array)

    q = matrix(np.zeros(18, ))
    
    a_array = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 5, 20, -40, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, -40, 20, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.1666666666667, 4.1666666666667, 3.125, -3.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16.666666666667, -33.333333333333, 16.666666666667, -12.5, 25, -12.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.125, 3.125, 4.1666666666667, -4.1666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12.5, -25, 12.5, -16.666666666667, 33.333333333333, -16.666666666667, 0, 0, 0]).reshape((12, 18))

    # # print('a_array: {}'.format(a_array))
    #
    A = matrix(a_array)
    #
    b_array = np.array([30.044475461375, 70.041325479264, 11.99887958196, 11.999541848998, -0.00092109849358377, -2.0176724178793e-05, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    #
    # # print('b_array: {}'.format(b_array))
    #
    # b = matrix(b_array)


    b = matrix(b_array)


    res = solvers.qp(P, q, None, None, A, b)

    optimized_val = np.array(res['x']).reshape((1, -1))

    print(optimized_val)

    # # Optimized result
    # objective_function_value = optimized_val @ p_array @ optimized_val.T
    # print('Objective value: {}'.format(objective_function_value))
    #
    # # Linspace result
    # test_val = np.linspace(0.0, 39.964489, 16).reshape(1, 16)
    # test_objective_function_value = test_val @ p_array @ test_val.T
    # print('Test objective value: {}'.format(test_objective_function_value))

    # # First segment
    # p_array_first_segment = copy.deepcopy(p_array[0:6, 0:6])
    # p_array_first_segment[5, 5] = p_array_first_segment[0, 0]
    # print('p_array first segment: {}'.format(p_array_first_segment))
    # optimized_val_first_segment = optimized_val[:, 0:6]
    # objective_function_value_first_segment = optimized_val_first_segment @ p_array_first_segment @ optimized_val_first_segment.T
    # print('Objective function first segment: {}'.format(objective_function_value_first_segment))
    #
    # # Third segment
    # p_array_third_segment = copy.deepcopy(p_array[10:16, 10:16])
    # p_array_third_segment[0, 0] = p_array_third_segment[-1, -1]
    # print('p_array third segment: {}'.format(p_array_third_segment))
    # optimized_val_third_segment = optimized_val[:, 10:16]
    # objective_function_value_third_segment = optimized_val_third_segment @ p_array_third_segment @ optimized_val_third_segment.T
    # print('Objective function third segment: {}'.format(objective_function_value_third_segment))
    #
    # # Second segment
    # p_array_second_segment = copy.deepcopy(p_array[5:11, 5:11])
    # p_array_second_segment[0, 0] -= p_array[0, 0]
    # p_array_second_segment[-1, -1] -= p_array[-1, -1]
    # print('p_array second segment: {}'.format(p_array_second_segment))
    # optimized_val_second_segment = optimized_val[:, 5:11]
    # objective_function_value_second_segment = optimized_val_second_segment @ p_array_second_segment @ optimized_val_second_segment.T
    # print('Objective function second segment: {}'.format(objective_function_value_second_segment))



