#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 下午5:38
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : piecewiseBezierSpline.py
# @Software: PyCharm

"""
Generation of the piecewise bezier spline.
"""

import numpy as np
import matplotlib.pyplot as plt
from corridorTrajectoryPlanning import Utils

class Point2f:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y

class Point3f:
    def __init__(self, s, d, t):
        self.s_ = s
        self.d_ = d
        self.t_ = t

# 3d trajectory generation
class BezierPiecewiseCurve:
    def __init__(self, s, d, ref_stamps):
        assert(len(s) == len(d))
        assert((len(ref_stamps) - 1) * 6 == len(s))
        self.ref_stamps_ = ref_stamps
        self.segment_num_ = len(ref_stamps) - 1
        self.s_coefficients_ = np.zeros((self.segment_num_, 6))
        self.d_coefficients_ = np.zeros((self.segment_num_, 6))
        for i in range(0, self.segment_num_):
            start_influenced_index = i * 6
            self.s_coefficients_[i][0] = s[start_influenced_index]
            self.s_coefficients_[i][1] = s[start_influenced_index + 1]
            self.s_coefficients_[i][2] = s[start_influenced_index + 2]
            self.s_coefficients_[i][3] = s[start_influenced_index + 3]
            self.s_coefficients_[i][4] = s[start_influenced_index + 4]
            self.s_coefficients_[i][5] = s[start_influenced_index + 5]
            self.d_coefficients_[i][0] = d[start_influenced_index]
            self.d_coefficients_[i][1] = d[start_influenced_index + 1]
            self.d_coefficients_[i][2] = d[start_influenced_index + 2]
            self.d_coefficients_[i][3] = d[start_influenced_index + 3]
            self.d_coefficients_[i][4] = d[start_influenced_index + 4]
            self.d_coefficients_[i][5] = d[start_influenced_index + 5]

    def generateTraj(self, sample_gap=0.01):
        traj = []
        for segment_index in range(self.segment_num_):
            time_span = self.ref_stamps_[segment_index + 1] - self.ref_stamps_[segment_index]
            sample_num = int(time_span / sample_gap)
            if segment_index == self.segment_num_ - 1:
                segment_seeds = np.linspace(0.0, 1.0, sample_num, endpoint=True)
            else:
                segment_seeds = np.linspace(0.0, 1.0, sample_num, endpoint=False)

            for current_seed in segment_seeds:
                time_stamp = self.ref_stamps_[segment_index] + (time_span * current_seed)
                traj.append(self.generatePoint(segment_index, current_seed, time_stamp))
        return traj


    def generatePoint(self, segment_index, remain, time_stamp):
        s_value = self.s_coefficients_[segment_index][0] * pow(1.0 - remain, 5) + 5.0 * self.s_coefficients_[segment_index][1] * remain * pow(1.0 - remain, 4) + 10.0 * self.s_coefficients_[segment_index][2] * pow(remain, 2) * pow(1.0 - remain, 3) + 10.0 * self.s_coefficients_[segment_index][3] * pow(remain, 3) * pow(1.0 - remain, 2) + 5.0 * self.s_coefficients_[segment_index][4] * pow(remain, 4) * (1.0 - remain) + self.s_coefficients_[segment_index][5] * pow(remain, 5)
        d_value = self.d_coefficients_[segment_index][0] * pow(1.0 - remain, 5) + 5.0 * self.d_coefficients_[segment_index][1] * remain * pow(1.0 - remain, 4) + 10.0 * self.d_coefficients_[segment_index][2] * pow(remain, 2) * pow(1.0 - remain, 3) + 10.0 * self.d_coefficients_[segment_index][3] * pow(remain, 3) * pow(1.0 - remain, 2) + 5.0 * self.d_coefficients_[segment_index][4] * pow(remain, 4) * (1.0 - remain) + self.d_coefficients_[segment_index][5] * pow(remain, 5)

        return Point3f(s_value, d_value, time_stamp)

# 2d trajectory generation
class BezierPiecewiseCurve2d:
    def __init__(self, x, ref_stamps):
        assert((len(ref_stamps) - 1) * 6 == len(x))
        self.ref_stamps_ = ref_stamps
        self.segment_num_ = len(ref_stamps) - 1
        self.coefficients_ = np.zeros((self.segment_num_, 6))
        for i in range(0, self.segment_num_):
            start_influenced_index = i * 6
            self.coefficients_[i][0] = x[start_influenced_index]
            self.coefficients_[i][1] = x[start_influenced_index + 1]
            self.coefficients_[i][2] = x[start_influenced_index + 2]
            self.coefficients_[i][3] = x[start_influenced_index + 3]
            self.coefficients_[i][4] = x[start_influenced_index + 4]
            self.coefficients_[i][5] = x[start_influenced_index + 5]

    def generateTraj(self, sample_gap=0.01):
        traj = []
        for segment_index in range(self.segment_num_):
            time_span = self.ref_stamps_[segment_index + 1] - self.ref_stamps_[segment_index]
            sample_num = int(time_span / sample_gap)
            if segment_index == self.segment_num_ - 1:
                segment_seeds = np.linspace(0.0, 1.0, sample_num, endpoint=True)
            else:
                segment_seeds = np.linspace(0.0, 1.0, sample_num, endpoint=False)

            for current_seed in segment_seeds:
                time_stamp = self.ref_stamps_[segment_index] + (time_span * current_seed)
                traj.append(self.generatePoint(segment_index, current_seed, time_stamp))
        return traj

    def generatePoint(self, segment_index, remain, time_stamp):
        value = self.coefficients_[segment_index][0] * pow(1.0 - remain, 5) + 5.0 * self.coefficients_[segment_index][1] * remain * pow(1.0 - remain, 4) + 10.0 * self.coefficients_[segment_index][2] * pow(remain, 2) * pow(1.0 - remain, 3) + 10.0 * self.coefficients_[segment_index][3] * pow(remain, 3) * pow(1.0 - remain, 2) + 5.0 * self.coefficients_[segment_index][4] * pow(remain, 4) * (1.0 - remain) + self.coefficients_[segment_index][5] * pow(remain, 5)

        return Point2f(value, time_stamp)

if __name__ == '__main__':
    # s = [29.987067, 32.386557, 34.786095, 37.269394, 39.866382, 42.579385, 42.579385, 45.292389, 48.121410, 51.068775, 54.109211, 57.216034, 57.216034, 60.322857, 63.496065, 66.708973, 69.933480, 73.166257, 73.166257, 74.243849, 75.324198, 76.406934, 77.494439, 78.592818]
    # d = [0.425765, 0.397481, 0.371210, 0.346578, 0.323214, 0.300804, 0.300804, 0.278394, 0.256937, 0.236117, 0.215671, 0.195438, 0.195438, 0.175204, 0.155184, 0.135216, 0.115240, 0.095346, 0.095346, 0.088715, 0.082111, 0.075545, 0.069043, 0.062651]
    # t = [0.000000, 1.200000, 2.400000, 3.600000, 4.000000]
    #
    # bezier_piecewise_curve = BezierPiecewiseCurve(s, d, t)
    # points = bezier_piecewise_curve.generateTraj()
    # interpolated_s, interpolated_d, interpolated_t = [], [], []
    # for point in points:
    #     interpolated_s.append(point.s_)
    #     interpolated_d.append(point.d_)
    #     interpolated_t.append(point.t_)

    interpolated_s = [-11269.403359, -11269.218520, -11269.033700, -11268.848835, -11268.663904, -11268.478886, -11268.293761, -11268.108510, -11267.923113, -11267.737551, -11267.551806, -11267.365860, -11267.179697, -11266.993298, -11266.806648, -11266.619729, -11266.432527, -11266.245027, -11266.057212, -11265.869069, -11265.680584, -11265.491655, -11265.302444, -11265.112850, -11264.922862, -11264.732467, -11264.541654, -11264.350412, -11264.158730, -11263.966597, -11263.774002, -11263.580852, -11263.387306, -11263.193270, -11262.998736, -11262.803696, -11262.608140, -11262.412062, -11262.215392, -11262.018250, -11261.820567, -11261.622335, -11261.423548, -11261.224127, -11261.024208, -11260.823718, -11260.622652, -11260.421005, -11260.218684, -11260.015857, -11259.812437, -11259.608422, -11259.403710, -11259.198486, -11258.992659, -11258.786228, -11258.579084, -11258.371427, -11258.163162, -11257.954181, -11257.744684, -11257.534575, -11257.323855, -11257.112409, -11256.900455, -11256.687891, -11256.474602, -11256.260808, -11256.046301, -11255.831282, -11255.615658, -11255.399326, -11255.182486, -11254.965049, -11254.746907, -11254.528269, -11254.308941, -11254.089113, -11253.868701, -11253.647605, -11253.426025, -11253.203774, -11252.981038, -11252.757649, -11252.533775, -11252.309263, -11252.084268, -11251.858651, -11251.632555, -11251.405852, -11251.178675, -11250.950907, -11250.722670, -11250.493857, -11250.264582, -11250.034748, -11249.804458, -11249.573625, -11249.342344, -11249.110537, -11248.878290, -11248.645534, -11248.412346, -11248.178667, -11247.944540, -11247.709989, -11247.474979, -11247.239554, -11247.003688, -11246.767417, -11246.530723, -11246.293629, -11246.056141, -11245.818262, -11245.579998, -11245.341363, -11245.102363, -11244.862991, -11244.623279, -11244.379081, -11244.134511, -11243.889622, -11243.644399, -11243.398822, -11243.152955, -11242.906778, -11242.660298, -11242.413487, -11242.166425, -11241.919083, -11241.671427, -11241.423550, -11241.175414, -11240.927029, -11240.678356, -11240.429495, -11240.180406, -11239.931046, -11239.681525, -11239.431798, -11239.181870, -11238.931700, -11238.681397, -11238.430914, -11238.180259, -11237.929391, -11237.678412, -11237.427281, -11237.176001, -11236.924540, -11236.672988, -11236.421305, -11236.169498, -11235.917572, -11235.665509, -11235.413366, -11235.161119, -11234.908774, -11234.656326, -11234.403803, -11234.151196, -11233.898505, -11233.645758, -11233.392922, -11233.140013, -11232.887036, -11232.633996, -11232.380952, -11232.127798, -11231.874589, -11231.621328, -11231.368018, -11231.114742, -11230.861342, -11230.607901, -11230.354419, -11230.100989, -11229.828618, -11229.556106, -11229.283450, -11229.010649, -11228.737694, -11228.464578, -11228.191286, -11227.917713, -11227.644022, -11227.370100, -11227.095920, -11226.821453, -11226.546667, -11226.271526, -11225.995990, -11225.720016, -11225.443558, -11225.166564]
    interpolated_d = [-2594.764659, -2594.679675, -2594.594702, -2594.509712, -2594.424694, -2594.339640, -2594.254540, -2594.169385, -2594.084167, -2593.998877, -2593.913508, -2593.828051, -2593.742498, -2593.656843, -2593.571077, -2593.485194, -2593.399187, -2593.313048, -2593.226772, -2593.140352, -2593.053782, -2592.967014, -2592.880125, -2592.793070, -2592.705841, -2592.618435, -2592.530845, -2592.443068, -2592.355098, -2592.266932, -2592.178563, -2592.089950, -2592.001165, -2591.912168, -2591.822952, -2591.733516, -2591.643855, -2591.553966, -2591.463816, -2591.373463, -2591.282873, -2591.192046, -2591.100979, -2591.009635, -2590.918081, -2590.826281, -2590.734235, -2590.641943, -2590.549360, -2590.456566, -2590.363523, -2590.270229, -2590.176639, -2590.082839, -2589.988787, -2589.894483, -2589.799875, -2589.705059, -2589.609988, -2589.514614, -2589.419030, -2589.323190, -2589.227096, -2589.130693, -2589.034084, -2588.937221, -2588.840049, -2588.742673, -2588.644995, -2588.547108, -2588.448969, -2588.350530, -2588.251883, -2588.152989, -2588.053794, -2587.954398, -2587.854709, -2587.754816, -2587.654679, -2587.554251, -2587.453625, -2587.352715, -2587.251605, -2587.150217, -2587.048630, -2586.946771, -2586.844712, -2586.742389, -2586.639866, -2586.537085, -2586.434106, -2586.330874, -2586.227446, -2586.123772, -2586.019902, -2585.915793, -2585.811491, -2585.706955, -2585.602228, -2585.497274, -2585.392132, -2585.286769, -2585.181220, -2585.075456, -2584.969498, -2584.863356, -2584.757011, -2584.650485, -2584.543764, -2584.436864, -2584.329776, -2584.222510, -2584.115067, -2584.007448, -2583.899656, -2583.791695, -2583.683568, -2583.575270, -2583.466817, -2583.356329, -2583.245670, -2583.134860, -2583.023893, -2582.912759, -2582.801486, -2582.690065, -2582.578497, -2582.466769, -2582.354918, -2582.242927, -2582.130782, -2582.018524, -2581.906136, -2581.793621, -2581.680959, -2581.568197, -2581.455316, -2581.342294, -2581.229182, -2581.115959, -2581.002626, -2580.889164, -2580.775622, -2580.661979, -2580.548237, -2580.434377, -2580.320446, -2580.206423, -2580.092312, -2579.978096, -2579.863815, -2579.749453, -2579.635011, -2579.520493, -2579.405888, -2579.291224, -2579.176489, -2579.061686, -2578.946814, -2578.831884, -2578.716892, -2578.601841, -2578.486743, -2578.371583, -2578.256369, -2578.141105, -2578.025792, -2577.910460, -2577.795057, -2577.679611, -2577.564122, -2577.448593, -2577.333061, -2577.217453, -2577.101806, -2576.986122, -2576.870443, -2576.746092, -2576.621647, -2576.497106, -2576.372468, -2576.247728, -2576.122881, -2575.997920, -2575.872794, -2575.747580, -2575.622221, -2575.496705, -2575.371018, -2575.245141, -2575.119057, -2574.992746, -2574.866186, -2574.739353, -2574.612221]
    interpolated_t = [0.000000, 0.020339, 0.040678, 0.061017, 0.081356, 0.101695, 0.122034, 0.142373, 0.162712, 0.183051, 0.203390, 0.223729, 0.244068, 0.264407, 0.284746, 0.305085, 0.325424, 0.345763, 0.366102, 0.386441, 0.406780, 0.427119, 0.447458, 0.467797, 0.488136, 0.508475, 0.528814, 0.549153, 0.569492, 0.589831, 0.610169, 0.630508, 0.650847, 0.671186, 0.691525, 0.711864, 0.732203, 0.752542, 0.772881, 0.793220, 0.813559, 0.833898, 0.854237, 0.874576, 0.894915, 0.915254, 0.935593, 0.955932, 0.976271, 0.996610, 1.016949, 1.037288, 1.057627, 1.077966, 1.098305, 1.118644, 1.138983, 1.159322, 1.179661, 1.200000, 1.220339, 1.240678, 1.261017, 1.281356, 1.301695, 1.322034, 1.342373, 1.362712, 1.383051, 1.403390, 1.423729, 1.444068, 1.464407, 1.484746, 1.505085, 1.525424, 1.545763, 1.566102, 1.586441, 1.606780, 1.627119, 1.647458, 1.667797, 1.688136, 1.708475, 1.728814, 1.749153, 1.769492, 1.789831, 1.810169, 1.830508, 1.850847, 1.871186, 1.891525, 1.911864, 1.932203, 1.952542, 1.972881, 1.993220, 2.013559, 2.033898, 2.054237, 2.074576, 2.094915, 2.115254, 2.135593, 2.155932, 2.176271, 2.196610, 2.216949, 2.237288, 2.257627, 2.277966, 2.298305, 2.318644, 2.338983, 2.359322, 2.379661, 2.400000, 2.420690, 2.441379, 2.462069, 2.482759, 2.503448, 2.524138, 2.544828, 2.565517, 2.586207, 2.606897, 2.627586, 2.648276, 2.668966, 2.689655, 2.710345, 2.731034, 2.751724, 2.772414, 2.793103, 2.813793, 2.834483, 2.855172, 2.875862, 2.896552, 2.917241, 2.937931, 2.958621, 2.979310, 3.000000, 3.020690, 3.041379, 3.062069, 3.082759, 3.103448, 3.124138, 3.144828, 3.165517, 3.186207, 3.206897, 3.227586, 3.248276, 3.268966, 3.289655, 3.310345, 3.331034, 3.351724, 3.372414, 3.393103, 3.413793, 3.434483, 3.455172, 3.475862, 3.496552, 3.517241, 3.537931, 3.558621, 3.579310, 3.600000, 3.622222, 3.644444, 3.666667, 3.688889, 3.711111, 3.733333, 3.755556, 3.777778, 3.800000, 3.822222, 3.844444, 3.866667, 3.888889, 3.911111, 3.933333, 3.955556, 3.977778, 4.000000]

    traj_length = len(interpolated_s)
    print('Traj length: {}'.format(traj_length))
    dis = ((interpolated_s[-1] - interpolated_s[0]) ** 2.0 + (interpolated_d[-1] - interpolated_d[0]) ** 2.0) ** 0.5
    print('Distance: {}'.format(dis))
    thetas, curvatures, velocities, accelerations = [0] * traj_length, [0] * traj_length, [0] * traj_length, [0] * traj_length
    for i in range(0, traj_length):
        if i != traj_length - 1:
            thetas[i] = np.arctan2(interpolated_d[i + 1] - interpolated_d[i], interpolated_s[i + 1] - interpolated_s[i])
            velocities[i] = np.sqrt((interpolated_d[i + 1] - interpolated_d[i]) ** 2.0 + (interpolated_s[i + 1] - interpolated_s[i]) ** 2.0) / (interpolated_t[i + 1] - interpolated_t[i])
        elif i == traj_length - 1:
            thetas[i] = thetas[i - 1]
            velocities[i] = velocities[i - 1]
    for i in range(0, traj_length):
        if i != traj_length - 1:
            curvatures[i] = (thetas[i + 1] - thetas[i]) / np.sqrt((interpolated_d[i + 1] - interpolated_d[i]) ** 2.0 + (interpolated_s[i + 1] - interpolated_s[i]) ** 2.0)
            accelerations[i] = (velocities[i + 1] - velocities[i]) / (interpolated_t[i + 1] - interpolated_t[i])
        elif i == traj_length - 1:
            curvatures[i] = curvatures[i - 1]
            accelerations[i] = accelerations[i - 1]

    info_fig = plt.figure(0, (12, 12))
    ax_0 = info_fig.add_subplot(221)
    ax_1 = info_fig.add_subplot(222)
    ax_2 = info_fig.add_subplot(223)
    ax_3 = info_fig.add_subplot(224)

    ax_0.plot(np.arange(0, traj_length, 1), thetas, c='r', linewidth=1.0)
    ax_0.title.set_text('theta')

    ax_1.plot(np.arange(0, traj_length, 1), curvatures, c='r', linewidth=1.0)
    ax_1.title.set_text('curvature')

    ax_2.plot(np.arange(0, traj_length, 1), velocities, c='r', linewidth=1.0)
    ax_2.title.set_text('velocity')

    ax_3.plot(np.arange(0, traj_length, 1), np.clip(accelerations, -5, 5), c='r', linewidth=1.0)
    ax_3.title.set_text('acceleration')



    # plt.figure(0)
    # plt.title('s-t')
    # plt.plot(interpolated_t, interpolated_s, c='r', linewidth=1.0)
    # plt.scatter(interpolated_t, interpolated_s, c='g', s=1.0)
    #
    # v_s_t, v_s = Utils.calculateVelocity(interpolated_t, interpolated_s)
    # plt.figure(1)
    # plt.title('ds-t')
    # plt.plot(v_s_t, v_s, c='r', linewidth=1.0)
    #
    # a_s_t, a_s = Utils.calculateAcceleration(v_s_t, v_s)
    # plt.figure(2)
    # plt.title('dds-t')
    # plt.plot(a_s_t, a_s, c='r', linewidth=1.0)

    plt.show()

