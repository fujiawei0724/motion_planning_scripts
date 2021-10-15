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
import copy
import math
from enum import Enum, unique


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

class Config:
    look_ahead_min_distance = 3.0
    look_ahead_max_distance = 50.0
    steer_control_gain = 1.5
    lateral_velocity_threshold = 0.35
    lateral_distance_threshold = 0.4

class Tools:
    @staticmethod
    def normalizeAngle(theta):
        processed_theta = theta
        while processed_theta > math.pi:
            processed_theta -= 2.0 * math.pi
        while processed_theta <= -math.pi:
            processed_theta += 2.0 * math.pi
        return processed_theta



# Path point class
class PathPoint:
    def __init__(self, x, y, theta):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta

    # Calculate distance between two path point
    def calculateDistance(self, path_point):
        return np.linalg.norm(np.array([self.x_ - path_point.x_, self.y_ - path_point.y_]))


# Lane class
class Lane:
    def __init__(self, start_point, end_point, id):
        # Generate lane path points information
        self.id_ = LaneId(id)
        samples = np.linspace(0.0, 1.0, 101, end_point=True)
        x_diff = end_point.x_ - start_point.x_
        y_diff = end_point.y_ - start_point.y_
        lane_theta = np.arctan2(end_point.y_ - start_point.y_, end_point.x_ - start_point.x_)
        lane_path_points = []
        for sample in samples:
            lane_path_points.append(PathPoint(start_point.x_ + sample * x_diff, start_point.y_ + sample * y_diff, lane_theta))
        self.path_points_ = lane_path_points
        self.path_points_margin_ = lane_path_points[0].calculateDistance(lane_path_points[1])

    # Calculate the distance from a position to lane
    def calculatePositionToLaneDistance(self, position):
        min_distance = float('inf')
        for lane_path_point in self.path_points_:
            cur_dis = position.calculateDistance(lane_path_point)
            min_distance = min(min_distance, cur_dis)
        return min_distance

    # Calculate the nearest path point index in lane from a specified position
    def calculateNearestIndexInLane(self, position):
        min_distance = float('inf')
        index = -1
        for i, lane_path_point in enumerate(self.path_points_):
            cur_distance = position.calculateDistance(lane_path_point)
            if cur_distance < min_distance:
                min_distance = cur_distance
                index = i
        assert index != -1
        return index

    # Get target point from a specified point and distance
    def calculateTargetDistancePoint(self, position, distance):
        # Calculate current index
        cur_position_index = self.calculateNearestIndexInLane(position)

        target_position_index = cur_position_index
        for lane_point_index in range(cur_position_index, len(self.path_points_)):
            if self.path_points_[cur_position_index].calculateDistance(self.path_points_[lane_point_index]) >= distance:
                target_position_index = lane_point_index
                break
        return target_position_index




# Lane set
class LaneServer:
    def __init__(self, lanes, vehicles):
        self.lanes_ = copy.deepcopy(lanes)
        self.vehicles_ = vehicles
        self.semantic_vehicles_ = {}
        self.update()

    # Update lane vehicle information
    def update(self):
        for vehicle in self.vehicles_:
            if vehicle.id_ != 0:
                semantic_vehicle = self.calculateSurroundVehicleBehavior(vehicle)
                self.semantic_vehicles_[semantic_vehicle.vehicle_.id_] = semantic_vehicle
            else:
                semantic_vehicle = self.calculateEgoVehicleBehavior(vehicle)
                self.semantic_vehicles_[semantic_vehicle.vehicle_.id_] = semantic_vehicle


    # Find the nearest lane from a postion
    def findNearestLane(self, cur_position):
        if not self.lanes_:
            assert False
        dis_mp = {}
        for lane in self.lanes_:
            dis_mp[lane.id_] = lane.calculatePositionToLaneDistance(cur_position)
        sorted_dis_mp = sorted(dis_mp.items(), key=lambda o: (o[1], o[0]))
        return self.lanes_[sorted_dis_mp[0][0]]

    # Calculate potential behavior and reference lane for surround agents
    def calculateSurroundVehicleBehavior(self, vehicle):
        assert vehicle.id_ != 0

        # Find the nearest lane
        nearest_lane = self.findNearestLane(vehicle.position_)

        # Calculate lateral velocity and distance
        lateral_velocity = vehicle.velocity_ * np.sin(vehicle.position_.theta_)
        lateral_distance = nearest_lane.calculatePositionToLaneDistance(vehicle.position_)

        # Delete error agent and behavior
        if nearest_lane.id_ == LaneId.LeftLane and lateral_velocity >= Config.lateral_velocity_threshold or nearest_lane.id_ == LaneId.RightLane and lateral_velocity <= -Config.lateral_velocity_threshold:
            return None

        # Generate semantic vehicles
        if lateral_distance >= Config.lateral_distance_threshold and lateral_velocity >= Config.lateral_velocity_threshold:
            return SemanticVehicle(vehicle, LateralBehavior.LaneChangeLeft, nearest_lane, self.lanes_[LaneId.LeftLane])
        elif lateral_distance <= -Config.lateral_distance_threshold and lateral_velocity <= -Config.lateral_velocity_threshold:
            return SemanticVehicle(vehicle, LateralBehavior.LaneChangeRight, nearest_lane, self.lanes_[LaneId.RightLane])
        else:
            return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, self.lanes_[LaneId.CenterLane])

    # Calculate potential behavior for ego vehicle
    def calculateEgoVehicleBehavior(self, vehicle):
        assert vehicle.id_ == 0

        # Find the nearest lane
        nearest_lane = self.findNearestLane(vehicle.position_)

        if nearest_lane.id_ == LaneId.CenterLane:
            return SemanticVehicle(vehicle, {LateralBehavior.LaneKeeping, LateralBehavior.LaneChangeLeft, LateralBehavior.LaneChangeRight}, nearest_lane)
        elif nearest_lane.id_ == LaneId.LeftLane:
            return SemanticVehicle(Vehicle, {LateralBehavior.LaneKeeping, LateralBehavior.LaneChangeRight}, nearest_lane)
        elif nearest_lane.id_ == LaneId.RightLane:
            return SemanticVehicle(Vehicle, {LateralBehavior.LaneKeeping, LateralBehavior.LaneChangeLeft}, nearest_lane)

    # Calculate reference lane for ego vehicle with a specified behavior
    def calculateEgoVehicleReferenceLane(self, semantic_vehicle, lateral_behavior):
        assert semantic_vehicle.vehicle_.id_ == 0

        # Judge behavior to select reference lane
        if lateral_behavior == LateralBehavior.LaneKeeping:
            if semantic_vehicle.nearest_lane_.id_ == LaneId.CenterLane:
                semantic_vehicle.reference_lane_ = self.lanes_[LaneId.CenterLane]
            elif semantic_vehicle.nearest_lane_.id_ == LaneId.LeftLane:
                semantic_vehicle.reference_lane_ = self.lanes_[LaneId.LeftLane]
            elif semantic_vehicle.nearest_lane_.id_ == LaneId.RightLane:
                semantic_vehicle.reference_lane_.id_ = self.lanes_[LaneId.RightLane]
        elif lateral_behavior == LateralBehavior.LaneChangeLeft:
            if semantic_vehicle.nearest_lane_.id_ == LaneId.CenterLane:
                semantic_vehicle.reference_lane_ = self.lanes_[LaneId.LeftLane]
            elif semantic_vehicle.nearest_lane_.id_ == LaneId.LeftLane:
                assert False
            elif semantic_vehicle.nearest_lane_.id_ == LaneId.RightLane:
                semantic_vehicle.reference_lane_.id_ = self.lanes_[LaneId.CenterLane]
        elif lateral_behavior == LateralBehavior.LaneChangeRight:
            if semantic_vehicle.nearest_lane_.id_ == LaneId.CenterLane:
                semantic_vehicle.reference_lane_ = self.lanes_[LaneId.RightLane]
            elif semantic_vehicle.nearest_lane_.id_ == LaneId.LeftLane:
                semantic_vehicle.reference_lane_.id_ = self.lanes_[LaneId.CenterLane]
            elif semantic_vehicle.nearest_lane_.id_ == LaneId.RightLane:
                assert False
        else:
            assert False

        return semantic_vehicle

    # Calculate leading vehicle, not limited to ego vehicle
    def getLeadingVehicle(self, cur_semantic_vehicle):
        reference_lane = cur_semantic_vehicle.reference_lane_

        # Initialize
        min_diff = float('inf')
        leading_vehicle = None

        # Get ego vehicle index in reference
        ego_vehicle_index = reference_lane.calculateNearestIndexInLane(cur_semantic_vehicle.vehicle_.position_)

        # Traverse semantic vehicles
        for semantic_vehicle_id, other_semantic_vehicle in self.semantic_vehicles_.items():
            if semantic_vehicle_id == cur_semantic_vehicle.vehicle_.id_:
                continue

            # Determine identical lane
            if other_semantic_vehicle.nearest_lane_.id_ == reference_lane.id_:
                other_vehicle_lane_index = other_semantic_vehicle.nearest_lane_.calculateNearestIndexInLane(other_semantic_vehicle.vehicle_.position_)
                if other_vehicle_lane_index > ego_vehicle_index:
                    if other_vehicle_lane_index - ego_vehicle_index < min_diff:
                        min_diff = other_vehicle_lane_index - ego_vehicle_index
                        leading_vehicle = other_semantic_vehicle

        return leading_vehicle

# Agent vehicle generator (without ego vehicle)
class AgentGenerator:
    def __init__(self):
        self.index_ = 1

    # Generate surround agents information
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
        # Information cache
        self.ego_vehicle_ = copy.deepcopy(ego_vehicle)
        self.potential_behavior_ = copy.deepcopy(potential_behavior)
        self.surround_agents_ = copy.deepcopy(surround_agents)
        self.lanes_ = copy.deepcopy(lanes)

    # Forward extend with interaction
    def multiAgentForward(self):
        pass
    # Forward extend without interaction
    def openLoopForward(self):
        pass

    # Forward once from current state
    def forwardOnce(self):
        pass

    # Calculate steer
    def calculateSteer(self, semantic_vehicle):
        # Determine look ahead distance in reference lane
        look_ahead_distance = min(max(Config.look_ahead_min_distance, semantic_vehicle.vehicle_.velocity_ * Config.steer_control_gain), Config.look_ahead_max_distance)

        # Calculate nearest path point in reference lane
        nearest_path_point = semantic_vehicle.reference_lane_.path_points_[semantic_vehicle.reference_lane_.calculateNearestIndexInLane(semantic_vehicle.vehicle_.position_)]

        # Calculate target path point in reference lane
        target_path_point_in_reference_lane = semantic_vehicle.reference_lane_.calculateTargetDistancePoint(nearest_path_point, look_ahead_distance)

        # Calculate look ahead distance in world frame
        look_ahead_distance_world = target_path_point_in_reference_lane.calculateDistance(semantic_vehicle.vehicle_.position_)

        # Calculate target angle and diff angle
        target_angle = np.arctan2(target_path_point_in_reference_lane.y_ - semantic_vehicle.vehicle_.position_.y_, target_path_point_in_reference_lane.x_ - semantic_vehicle.vehicle_.position_.x_)
        diff_angle = Tools.normalizeAngle(target_angle - semantic_vehicle.vehicle_.position_.theta_)








    # Calculate velocity
    def calculateVelocity(self):
        pass

    # Calculate desired state
    def calculateDesiredState(self):
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

class SemanticVehicle:
    def __init__(self, vehicle, potential_behaviors, nearest_lane, reference_lane=None):
        self.vehicle_ = vehicle
        self.potential_behaviors_ = potential_behaviors
        self.nearest_lane_ = nearest_lane
        self.reference_lane_ = reference_lane







# Trajectory class, includes
class Trajectory:
    def __init__(self):
        pass


if __name__ == '__main__':
    pass









