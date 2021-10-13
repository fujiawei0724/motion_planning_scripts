# -- coding: utf-8 --
# @Time : 2021/10/12 下午5:25
# @Author : fujiawei0724
# @File : common.py
# @Software: PyCharm

"""
This code contains the components for the behavior planner core.
"""


import numpy as np
import random
from enum import Enum, unique


# Path point class
class PathPoint:
    def __init__(self, x, y, theta):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta

# Lane class
class Lane:
    def __init__(self, start_point, end_point, id):
        self.id_ = LaneId(id)
        samples = np.linspace(0.0, 1.0, 101, end_point=True)
        x_diff = end_point.x_ - start_point.x_
        y_diff = end_point.y_ - start_point.y_
        lane_theta = np.arctan2(end_point.y_ - start_point.y_, end_point.x_ - start_point.x_)
        lane_path_points = []
        for sample in samples:
            lane_path_points.append(PathPoint(start_point.x_ + sample * x_diff, start_point.y_ + sample * y_diff, lane_theta))
        self.path_points_ = lane_path_points

# Agent vehicle generator (without ego vehicle)
class AgentGenerator:
    def __init__(self):
        self.index_ = 1

    def generateSingleAgent(self):
        agent_length = random.uniform(4.0, 6.0)
        agent_width = random.uniform(1.8, 2.5)
        agent_velocity = random.uniform(3.0, 10.0)
        agent_acceleration = random.uniform(-1.0, 1.0)
        x_position = random.uniform(0.0, 80.0)
        y_position = random.uniform(-3.5, 3.5)
        theta = random.uniform(-0.2, 0.2)
        agent_position = PathPoint(x_position, y_position, theta)
        ego_vehicle = Vehicle(self.index_, agent_position, agent_length, agent_width, agent_velocity, agent_acceleration)
        self.index_ += 1
        return ego_vehicle

    def generateAgents(self, num):
        agents = {}
        for i in range(1, num + 1):
            ego_vehicle = self.generateSingleAgent()
            agents[ego_vehicle.id_] = ego_vehicle
        return agents

# Forward simulation
class ForwardExtender:
    def __init__(self, ego_vehicle, potential_behavior, surround_agents, lanes):
        pass

    def multiAgentForward(self):
        pass

    def openLoopForward(self):
        pass

# Policy evaluate
class PolicyEvaluater:
    def __init__(self):
        pass



# IDM model
class IDM:
    @staticmethod
    def calculateVelocity():
        pass

    @staticmethod
    def calculateAcceleration():
        pass

# Define lateral behavior
@unique
class LateralBehavior(Enum):
    LaneKeeping = 0
    LaneChangeLeft = 1
    LaneChangeRight = 2

# Define lane id
@unique
class LaneId(Enum):
    CenterLane = 0
    LeftLane = 1
    RightLane = 2

# Rectangle class, denotes the area occupied by the vehicle
class Rectangle:
    def __init__(self, center_point, theta, length, width):
        # Calculate four vertex of the rectangle
        pass

# Vehicle class
class Vehicle:
    def __init__(self, vehicle_id, position, length, width, velocity, acceleration, time_stamp=None):
        self.id_ = vehicle_id
        self.position_ = position
        self.length_ = length
        self.width_ = width
        self.velocity_ = velocity
        self.acceleration_ = acceleration
        self.time_stamp_ = time_stamp

    def getRectangle(self):
        pass

# Trajectory class, includes
class Trajectory:
    def __init__(self):
        pass

if __name__ == '__main__':
    pass









