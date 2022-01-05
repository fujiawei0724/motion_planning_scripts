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
    csv_list = glob.glob('/home/fjw/PioneerTest/catkin_ws/src/planning/motion_planning/trajectory_record/*.csv')
    print('file number: ', len(csv_list))
    if len(csv_list) == 0:
        print('error: no file')
        sys.exit()
    for i in csv_list:
        fr = open(i, 'rb').read()
        with open('/home/fjw/PioneerTest/catkin_ws/src/planning/motion_planning/trajectory_record/result.csv', 'ab') as f:
            f.write(fr)

    # Read csv
    csv_file = open('/home/fjw/PioneerTest/catkin_ws/src/planning/motion_planning/trajectory_record/result.csv', "r")
    reader = csv.reader(csv_file)
    position_x = []
    position_y = []
    thetas = []
    curvatures = []
    velocities = []
    accelerations = []
    # distance_data = []
    # min_distance = float('inf')
    for item in reader:
        position_x.append(float(item[0]))
        position_y.append(float(item[1]))
        thetas.append(float(item[2]))
        curvatures.append(float(item[3]))
        velocities.append(float(item[4]))
        accelerations.append(float(item[5]))

    csv_file.close()

    print('Average curvature: {}'.format(np.mean(np.abs(curvatures))))
    print('Average velocities: {}'.format(np.mean(velocities)))
    print('Average acceleration: {}'.format(np.mean(np.abs(accelerations))))

