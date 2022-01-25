#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/25 下午2:05
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : analyzeAcceleration.py
# @Software: PyCharm

"""
Analyze the trajectory acceleration information of different behavior planners.
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import glob
import scipy.signal

if __name__ == '__main__':
    # Merge csv
    directory = '/home/fjw/Desktop/HPDM_record/simulation_data/data/20220121/EUDM/'
    csv_list = glob.glob(directory + '*.csv')
    for index, i in enumerate(csv_list):
        fr = open(i, 'r')
        reader = csv.reader(fr)
        accelerations = []
        for item in reader:
            if len(item) == 2:
                continue
            else:
                accelerations.append(float(item[5]))
        # Because of the transformation error between frenet frame and world frame, the position information has some fluctuation, we use a filter to eliminate it
        accelerations = scipy.signal.savgol_filter(accelerations, 53, 3)
        if index >= 10:
            break

        plt.figure(index)
        plt.plot(np.arange(0, len(accelerations), 1), accelerations, c='r', linewidth=1.0)
    plt.show()