# -- coding: utf-8 --
# @Time : 2021/10/12 下午7:35
# @Author : fujiawei0724
# @File : decisionMaking.py
# @Software: PyCharm

"""
This code includes a single simulation for behavior planner.
"""

import numpy as np
import common

# For test
if __name__ == '__main__':

    # 1. Generate lanes
    lanes = {}
    center_lane = common.Lane(common.PathPoint(0.0, 0.0, None), common.PathPoint(100.0, 0.0, None), common.LaneId.CenterLane.value)
    left_lane = common.Lane(common.PathPoint(0.0, 3.0, None), common.PathPoint(100.0, 3.0, None), common.LaneId.LeftLane.value)
    right_lane = common.Lane(common.PathPoint(0.0, -3.0, None), common.PathPoint(100.0, -3.0, None), common.LaneId.RightLane.value)
    lanes[common.LaneId.CenterLane] = center_lane
    lanes[common.LaneId.LeftLane] = left_lane
    lanes[common.LaneId.RightLane] = right_lane

    # 2. Initialize ego vehicle information
    ego_vehicle_start_position = common.PathPoint(20.0, 0.0, 0.0)
    ego_vehicle_start_velocity = 5.0
    ego_vehicle_start_acceleration = 1.0
    ego_vehicle_length = 5.0
    ego_vehicle_width = 2.0
    ego_vehicle = common.Vehicle(0, ego_vehicle_start_position, ego_vehicle_length, ego_vehicle_width, ego_vehicle_start_velocity, ego_vehicle_start_acceleration)

    # 3. Generate surround vehicles randomly
    agents_generator = common.AgentGenerator()
    surround_agent_num = 5
    agents = agents_generator.generateAgents(surround_agent_num)

    # 4. Generate



