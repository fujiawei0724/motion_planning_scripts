#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 下午3:51
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : csv_data_analysis.py
# @Software: PyCharm

"""
Grab the data from simulation csv and analyze in multi-extent.
"""

import numpy as np
import glob
import sys
import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':


    # Merge csv
    csv_list = glob.glob('/home/fjw/Desktop/HPDM_record/simulation_data/data/20220117/EUDM/*.csv')
    print('file number: ', len(csv_list))
    if len(csv_list) == 0:
        print('error: no file')
        sys.exit()
    for i in csv_list:
        fr = open(i, 'rb').read()
        with open('/home/fjw/Desktop/HPDM_record/simulation_data/data/20220117/EUDM/result.csv', 'ab') as f:
            f.write(fr)

    # Read csv
    csv_file = open('/home/fjw/Desktop/HPDM_record/simulation_data/data/20220117/EUDM/result.csv', "r")
    reader = csv.reader(csv_file)
    position_x = []
    position_y = []
    thetas = []
    curvatures = []
    velocities = []
    accelerations = []
    time_consumption = []
    min_dis = []
    # distance_data = []
    # min_distance = float('inf')
    for item in reader:
        if len(item) == 2:
            min_dis.append(float(item[0]))
            time_consumption.append(float(item[1]))
        else:
            position_x.append(float(item[0]))
            position_y.append(float(item[1]))
            thetas.append(float(item[2]))
            curvatures.append(float(item[3]))
            velocities.append(float(item[4]))
            accelerations.append(float(item[5]))
    csv_file.close()

    print('Average time consumption: {}'.format(np.mean(time_consumption)))
    print('Min distance to obstacles: {}'.format(np.min(min_dis)))
    print('Average curvature: {}'.format(np.mean(np.abs(curvatures))))
    print('Average velocities: {}'.format(np.mean(velocities)))
    print('Average acceleration: {}'.format(np.mean(np.abs(accelerations))))
    print('Average jerk: {}'.format(np.mean(np.abs(np.diff(accelerations)))))

