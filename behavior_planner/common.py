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
from collections import namedtuple
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
    BigEPS = 1e-1
    look_ahead_min_distance = 3.0
    look_ahead_max_distance = 50.0
    steer_control_gain = 1.5
    lateral_velocity_threshold = 0.35
    lateral_distance_threshold = 0.4
    wheelbase_length = 2.8
    


class Tools:
    @staticmethod
    def normalizeAngle(theta):
        processed_theta = theta
        while processed_theta > math.pi:
            processed_theta -= 2.0 * math.pi
        while processed_theta <= -math.pi:
            processed_theta += 2.0 * math.pi
        return processed_theta

    @staticmethod
    def calculateSteer(wheelbase_length, angle_diff, look_ahead_distance):
        return np.arctan2(2.0 * wheelbase_length * np.sin(angle_diff), look_ahead_distance)

    @staticmethod
    def truncate(val_in, lower, upper):
        if lower > upper:
            assert False
        res = val_in
        res = max(res, lower)
        res = min(res, upper)
        return res



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

    # Calculate the nearest path point index in a lane from a specified position
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

    # Calculate the nearest path point in a lane from a specified position
    def calculateNearestPointInLane(self, position):
        return self.path_points_[self.calculateNearestIndexInLane(position)]

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



    def __init__(self, ego_vehicle, potential_behavior, surround_agents, lane_server):
        # Information cache
        self.ego_vehicle_ = copy.deepcopy(ego_vehicle)
        self.potential_behavior_ = copy.deepcopy(potential_behavior)
        self.surround_agents_ = copy.deepcopy(surround_agents)
        self.lane_server_ = copy.deepcopy(lane_server)

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

        # Calculate target steer
        target_steer = Tools.calculateSteer(Config.wheelbase_length, diff_angle, look_ahead_distance_world)

        return target_steer

    # Calculate velocity
    def calculateVelocity(self, ego_semantic_vehicle, dt):
        # Calculate leading vehicle
        leading_semantic_vehicle = self.lane_server_.getLeadingVehicle(ego_semantic_vehicle)

        # Judge leading vehicle state
        if leading_semantic_vehicle == None:
            # Don't exist leading vehicle, using virtual leading vehicle
            virtual_leading_vehicle_distance = 100.0 + 100.0 * ego_semantic_vehicle.vehicle_.velocity_
            target_velocity = IDM.calculateVelocity(0.0, virtual_leading_vehicle_distance, ego_semantic_vehicle.vehicle_.velocity_, ego_semantic_vehicle.vehicle_.velocity_, dt)
        else:
            # With leading vehicle
            # Calculate ego vehicle and leading vehicle's nearest position in corresponding lane respectively
            assert ego_semantic_vehicle.reference_lane_ == leading_semantic_vehicle.nearest_lane_
            corresponding_lane_ = ego_semantic_vehicle.reference_lane_
            ego_vehicle_position_in_corresponding_lane = corresponding_lane_.calculateNearestPointInLane(ego_semantic_vehicle.vehicle_.position_)
            leading_vehicle_position_in_corresponding_lane = corresponding_lane_.calculateNearestPointInLane(leading_semantic_vehicle.vehicle_.position_)

            # Calculate the distance between ego vehicle and ego vehicle
            ego_leading_vehicles_diatance = ego_vehicle_position_in_corresponding_lane.calculateDistance(leading_vehicle_position_in_corresponding_lane)
            target_velocity = IDM.calculateVelocity(0.0, ego_leading_vehicles_diatance, ego_semantic_vehicle.vehicle_.velocity_, leading_semantic_vehicle.vehicle_.velocity_, dt)

        return target_velocity

    # Calculate desired state
    def calculateDesiredDistance(self):
        pass

        








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

    # IDM config
    desired_velocity = 0.0
    vehicle_length = 5.0
    minimum_spacing = 2.0
    desired_headaway_time = 1.0
    acceleration = 2.0
    comfortable_braking_deceleration = 3.0
    hard_braking_deceleration = 5.0
    exponent = 4

    # Calculate velocity using IDM model with linear function
    @staticmethod
    def calculateVelocity(cur_s, leading_s, cur_velocity, leading_velocity, dt):

        # Linear predict function
        def linearPredict(dt):
            # Calculate responding acceleration
            acc = IDM.calculateAcceleration(cur_s, leading_s, cur_velocity, leading_velocity)
            acc = max(acc, -min(IDM.hard_braking_deceleration, cur_velocity / dt))
            next_cur_s = cur_s + cur_velocity * dt + 0.5 * acc * dt * dt
            next_leading_s = leading_s + leading_velocity * dt
            next_cur_velocity = cur_velocity + acc * dt
            next_leading_velocity = leading_velocity
            return next_cur_s, next_leading_s, next_cur_velocity, next_leading_velocity

        _, _, target_velocity, _ = linearPredict(dt)

        return target_velocity

    # Calculate acceleration using IDM model
    @staticmethod
    def calculateAcceleration(cur_s, leading_s, cur_velocity, leading_velocity):
        # Calculate parameters
        a_free = IDM.acceleration * (1 - pow(cur_velocity / IDM.desired_velocity, IDM.exponent)) if cur_velocity <= IDM.desired_velocity else -IDM.comfortable_braking_deceleration * (1 - pow(IDM.desired_velocity / cur_velocity, IDM.acceleration * IDM.exponent / IDM.comfortable_braking_deceleration))
        s_alpha = max(0.0, leading_s - cur_s - IDM.vehicle_length)
        z = (IDM.minimum_spacing + max(0.0, cur_velocity * IDM.desired_headaway_time + cur_velocity * (cur_velocity - leading_velocity) / (2.0 * np.sqrt(IDM.acceleration * IDM.comfortable_braking_deceleration)))) / s_alpha

        # Calculate output acceleration
        if cur_velocity <= IDM.desired_velocity:
            a_out = IDM.acceleration * (1 - pow(z, 2)) if z >= 1.0 else a_free * (1 - pow(z, 2.0 * IDM.acceleration / a_free))
        else:
            a_out = a_free + IDM.acceleration * (1 - pow(z, 2)) if z >= 1.0 else a_free
        a_out = max(min(IDM.acceleration, a_out), -IDM.hard_braking_deceleration)

        return a_out

# Ideal steer model
class IdealSteerModel:

    def __init__(self, wheelbase_len, max_lon_acc, max_lon_dec, max_lon_acc_jerk, max_lon_dec_jerk, max_lat_acc, max_lat_jerk, max_steering_angle, max_steer_rate, max_curvature):
        self.wheelbase_len_ = wheelbase_len
        self.max_lon_acc_ = max_lon_acc
        self.max_lon_dec_ = max_lon_dec
        self.max_lon_acc_jerk_ = max_lon_acc_jerk
        self.max_lon_dec_jerk = max_lon_dec_jerk
        self.max_lat_acc_ = max_lat_acc
        self.max_lat_jerk_ = max_lat_jerk
        self.max_steering_angle_ = max_steering_angle
        self.max_steer_rate_ = max_steer_rate
        self.max_curvature_ = max_curvature

        # Initialize shell
        self.control_ = []
        self.state_ = None
        # Internal_state[0] means x position, internal_state[1] means y position, internal_state[2] means angle, internal_state[3] means velocity, internal_state[4] means steer
        self.internal_state_ = None
        self.desired_lon_acc_ = 0.0
        self.desired_lat_acc_ = 0.0
        self.desired_steer_rate_ = 0.0

    # Set control information, control[0] means steer, control[1] means velocity
    def setControl(self, control):
        self.control_ = control

    # Set state information, use vehicle class reprsent vehicle state
    def setState(self, vehicle):
        self.state_ = vehicle

    # Truncate control
    def truncateControl(self, dt):
        self.desired_lon_acc_ = self.control_[0] - self.state_.velocity / dt
        desired_lon_jerk = (self.desired_lon_acc_ - self.state_.acceleration_) / dt
        desired_lon_jerk = Tools.truncate(desired_lon_jerk, -self.max_lon_dec_jerk, self.max_lon_acc_jerk_)
        self.desired_lon_acc_ = desired_lon_jerk * dt + self.state_.acceleration_
        self.desired_lon_acc_ = Tools.truncate(self.desired_lon_acc_, -self.max_lon_dec_, self.max_lon_acc_)
        self.control_[1] = max(self.state_.velocity_ + self.desired_lon_acc_ * dt, 0.0)
        self.desired_lat_acc_ = pow(self.control_[1], 2) * (np.tan(self.control_[0]) / self.wheelbase_len_)
        lat_acc_ori = pow(self.state_.velocity_, 2.0) * self.state_.curvature_
        lat_jerk_desired = (self.desired_lat_acc_ - lat_acc_ori) / dt
        lat_jerk_desired = Tools.truncate(lat_jerk_desired, -self.max_lat_jerk_, self.max_lat_jerk_)
        desired_lat_acc_ = lat_jerk_desired * dt + lat_acc_ori
        desired_lat_acc_ = Tools.truncate(desired_lat_acc_, -self.max_lat_acc_, self.max_lat_acc_)
        self.control_[0] = np.atan(desired_lat_acc_ * self.wheelbase_len_ / max(pow(self.control_[1], 2.0), 0.1 * Config.BigEPS))
        self.desired_steer_rate_ = Tools.normalizeAngle(self.control_[0] - self.state_.steer_) / dt
        self.desired_steer_rate_ = Tools.truncate(self.desired_steer_rate_, -self.max_steer_rate_, self.max_steer_rate_)
        self.control_[0] = Tools.normalizeAngle(self.state_.steer_ + self.desired_steer_rate_ * dt)

    # Forward once
    def step(self, dt):
        self.state_.steer_ = np.atan(self.state_.curvature_ * self.wheelbase_len_)
        self.updateInternalState()
        self.control_[1] = max(0.0, self.control_[1])
        self.control_[0] = Tools.truncate(self.control_[0], -self.max_steering_angle_, self.max_steering_angle_)
        self.truncateControl(dt)
        self.desired_lon_acc_ = (self.control_[1] - self.state_.velocity_) / dt
        self.desired_steer_rate_ = Tools.normalizeAngle(self.control_[0] - self.state_.steer_)

        # Linear predict function
        def linearPredict(internal_state):
            


    # Update internal state
    def updateInternalState(self):
        self.internal_state_[0] = self.state_.position_.x_
        self.internal_state_[1] = self.state_.position_.y_
        self.internal_state_[2] = self.state_.position_.theta_
        self.internal_state_[3] = self.state_.velocity_
        self.internal_state_[4] = self.state_.steer_


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
        # TODO: add curvature and steer information
        self.curvature_ = 0.0
        self.steer_ = 0.0

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









