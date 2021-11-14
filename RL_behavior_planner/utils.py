# -- coding: utf-8 --
# @Time : 2021/11/14 下午6:49
# @Author : fujiawei0724
# @File : utils.py
# @Software: PyCharm

"""
The components for RL behavior planner.
"""
import numpy as np
import copy
import math
import random
from enum import Enum, unique
from collections import defaultdict


# Define lateral behavior
@unique
class LateralBehavior(Enum):
    LaneKeeping = 0
    LaneChangeLeft = 1
    LaneChangeRight = 2


# Define longitudinal behavior
@unique
class LongitudinalBehavior(Enum):
    Conservative = 0
    Normal = 1
    Aggressive = 2

class Config:
    BigEPS = 1e-1
    EPS = 1e-7
    look_ahead_min_distance = 3.0
    look_ahead_max_distance = 50.0
    steer_control_gain = 1.5
    lateral_velocity_threshold = 0.35
    lateral_distance_threshold = 0.4
    wheelbase_length = 2.8

    # Control parameters
    max_lon_acc_jerk = 5.0
    max_lon_brake_jerk = 5.0
    max_lat_acceleration_abs = 1.5
    max_lat_jerk_abs = 3.0
    max_steer_angle_abs = 45.0 / 180.0 * math.pi
    max_steer_rate = 0.39
    max_curvature_abs = 0.33

    # User defined velocity
    user_desired_velocity = 8.0

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

    # Truncate control
    @staticmethod
    def truncate(val_in, lower, upper):
        if lower > upper:
            assert False
        res = val_in
        res = max(res, lower)
        res = min(res, upper)
        return res


    # For OBB rectangle collision judgement
    @staticmethod
    def getProjectionOnVertex(vertex, axis):
        min = float('inf')
        max = -float('inf')
        for vertice in vertex:
            projection = np.dot(vertice, axis)
            if projection < min:
                min = projection
            if projection > max:
                max = projection
        proj = np.array([min, max])
        return proj

    # For OBB rectangle collision judgement
    @staticmethod
    def getOverlapLength(proj_1, proj_2):
        if proj_1[0] > proj_2[1] or proj_2[0] > proj_1[1]:
            return 0
        else:
            return min(proj_1[1], proj_2[1]) - max(proj_1[0], proj_2[0])

# Visualization
class Visualization:
    @staticmethod
    def transformPathPointsToArray(lane_path_points):
        points_num = len(lane_path_points)
        path_points_array = np.zeros((points_num, 2))

        # Traverse and load data
        for index, path_point in enumerate(lane_path_points):
            path_points_array[index][0] = path_point.x_
            path_points_array[index][1] = path_point.y_

        return path_points_array

# Vehicle behavior contains both latitudinal and longitudinal behavior
class VehicleBehavior:
    def __init__(self, lat_beh, lon_beh):
        self.lat_beh_ = lat_beh
        self.lon_beh_ = lon_beh


# Define lane id
@unique
class LaneId(Enum):
    CenterLane = 0
    LeftLane = 1
    RightLane = 2


# Define behavior sequence for construct behavior space
class BehaviorSequence:
    def __init__(self, behavior_sequence):
        self.beh_seq_ = behavior_sequence

    # DEBUG: print information
    def print(self):
        for veh_beh_index, veh_beh in enumerate(self.beh_seq_):
            print('Single behavior index: {}, lateral behavior: {}, longitudinal behavior: {}'.format(veh_beh_index, veh_beh.lat_beh_, veh_beh.lon_beh_))

# Construct available behavior sequence
# TODO: add consideration of current lateral behavior
class BehaviorGenerator:
    def __init__(self, seq_length):
        self.seq_length_ = seq_length

    # Construct vehicle behavior set
    def generateBehaviors(self):
        veh_beh_set = []

        # Traverse longitudinal behaviors
        for lon_beh in LongitudinalBehavior:
            cur_behavior_sequence = []
            for beh_index in range(0, self.seq_length_):
                for lat_beh in LateralBehavior:
                    if lat_beh != LateralBehavior.LaneKeeping:
                        # Add lane change situations
                        veh_beh_set.append(self.addBehavior(cur_behavior_sequence, lon_beh, lat_beh, self.seq_length_ - beh_index))
                cur_behavior_sequence.append(VehicleBehavior(LateralBehavior.LaneKeeping, lon_beh))
            veh_beh_set.append(BehaviorSequence(cur_behavior_sequence))

        return veh_beh_set

    # Add lane change situation which start from intermediate time stamp
    @classmethod
    def addBehavior(cls, cur_beh_seq, lon_beh, lat_beh, num):

        # Initialize
        res_beh_seq = copy.deepcopy(cur_beh_seq)

        # Add lane change behavior
        for i in range(0, num):
            res_beh_seq.append(VehicleBehavior(lat_beh, lon_beh))

        return BehaviorSequence(res_beh_seq)

# Path point class
class PathPoint:
    def __init__(self, x, y, theta=None):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta

    # Calculate distance between two path point
    def calculateDistance(self, path_point):
        return np.linalg.norm(np.array([self.x_ - path_point.x_, self.y_ - path_point.y_]))

    # Transform to array (for DEBUG)
    def toArray(self):
        return np.array([self.x_, self.y_])


# Lane class
class Lane:
    def __init__(self, start_point, end_point, id):
        self.id_ = id

        # Generate lane path points and lane boundary points
        # Initialize information
        sample_num = np.linalg.norm(np.array([end_point.x_ - start_point.x_, end_point.y_ - start_point.y_])) / 0.1
        samples = np.linspace(0.0, 1.0, int(sample_num), endpoint=True)
        x_diff = end_point.x_ - start_point.x_
        y_diff = end_point.y_ - start_point.y_
        lane_theta = np.arctan2(end_point.y_ - start_point.y_, end_point.x_ - start_point.x_)
        lane_path_points = []
        lane_left_boundary_points = []
        lane_right_boundary_points = []

        # Sampling based generation
        lane_width = 3.5
        for sample in samples:
            lane_path_points.append(PathPoint(start_point.x_ + sample * x_diff, start_point.y_ + sample * y_diff, lane_theta))
            lane_left_boundary_points.append([start_point.x_ + sample * x_diff + np.cos(lane_theta + math.pi / 2.0) * lane_width / 2.0, start_point.y_ + sample * y_diff + np.sin(lane_theta + math.pi / 2.0) * lane_width / 2.0])
            lane_right_boundary_points.append([start_point.x_ + sample * x_diff + np.cos(lane_theta - math.pi / 2.0) * lane_width / 2.0, start_point.y_ + sample * y_diff + np.sin(lane_theta - math.pi / 2.0) * lane_width / 2.0])
        self.path_points_ = lane_path_points
        self.left_boundary_points_ = np.array(lane_left_boundary_points)
        self.right_boundary_points_ = np.array(lane_right_boundary_points)

        # Calculate lane points margin
        self.path_points_margin_ = lane_path_points[0].calculateDistance(lane_path_points[1])

    # Calculate the distance from a position to lane
    def calculatePositionToLaneDistance(self, position):
        min_distance = float('inf')
        for lane_path_point in self.path_points_:
            cur_dis = position.calculateDistance(lane_path_point)
            min_distance = min(min_distance, cur_dis)
        return min_distance if position.y_ >= self.path_points_[0].y_ else -min_distance

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
        return self.path_points_[target_position_index]


# Lane set
class LaneServer:
    def __init__(self, lanes):
        self.lanes_ = copy.deepcopy(lanes)

    # Initialize information
    def initialize(self, vehicles):
        semantic_vehicles = dict()
        for vehicle in vehicles:
            if vehicle.id_ != 0:
                # For surround vehicle
                semantic_vehicle = self.calculateSurroundVehicleBehavior(vehicle)
                semantic_vehicles[semantic_vehicle.vehicle_.id_] = semantic_vehicle
            else:
                # For ego vehicle
                semantic_vehicle = self.calculateEgoVehicleBehavior(vehicle)
                semantic_vehicles[semantic_vehicle.vehicle_.id_] = semantic_vehicle

        return semantic_vehicles


    # Find the nearest lane from a position
    def findNearestLane(self, cur_position):
        if not self.lanes_:
            assert False
        dis_mp = {}
        for lane_id, lane in self.lanes_.items():
            dis_mp[lane_id] = abs(lane.calculatePositionToLaneDistance(cur_position))
        sorted_dis_mp = sorted(dis_mp.items(), key=lambda o: o[1])
        return self.lanes_[sorted_dis_mp[0][0]]

    # Calculate potential behavior and reference lane for surround agents
    def calculateSurroundVehicleBehavior(self, vehicle):
        assert vehicle.id_ != 0

        # Find the nearest lane
        nearest_lane = self.findNearestLane(vehicle.position_)

        # Calculate lateral velocity and distance
        lateral_velocity = vehicle.velocity_ * np.sin(vehicle.position_.theta_)
        lateral_distance = nearest_lane.calculatePositionToLaneDistance(vehicle.position_)

        # Generate semantic vehicles
        if lateral_distance >= Config.lateral_distance_threshold and lateral_velocity >= Config.lateral_velocity_threshold:
            # For change left
            if nearest_lane.id_ == LaneId.CenterLane:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeLeft, nearest_lane, self.lanes_[LaneId.LeftLane])
            elif nearest_lane.id_ == LaneId.LeftLane:
                # Error situation, set lane keeping
                return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, nearest_lane)
            elif nearest_lane.id_ == LaneId.RightLane:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeLeft, nearest_lane, self.lanes_[LaneId.CenterLane])
            else:
                assert False

        elif lateral_distance <= -Config.lateral_distance_threshold and lateral_velocity <= -Config.lateral_velocity_threshold:
            # For change right
            if nearest_lane.id_ == LaneId.CenterLane:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeRight, nearest_lane, self.lanes_[LaneId.RightLane])
            elif nearest_lane.id_ == LaneId.LeftLane:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeRight, nearest_lane, self.lanes_[LaneId.CenterLane])
            elif nearest_lane.id_ == LaneId.RightLane:
                # Error situation, set lane keeping
                return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, nearest_lane)
            else:
                assert False

        else:
            return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, nearest_lane)

    # Calculate potential behavior for ego vehicle
    def calculateEgoVehicleBehavior(self, vehicle):
        assert vehicle.id_ == 0

        # Find the nearest lane
        nearest_lane = self.findNearestLane(vehicle.position_)

        if nearest_lane.id_ == LaneId.CenterLane:
            return SemanticVehicle(vehicle, [LateralBehavior.LaneKeeping, LateralBehavior.LaneChangeLeft,
                                             LateralBehavior.LaneChangeRight], nearest_lane)
        elif nearest_lane.id_ == LaneId.LeftLane:
            return SemanticVehicle(vehicle, [LateralBehavior.LaneKeeping, LateralBehavior.LaneChangeRight],
                                   nearest_lane)
        elif nearest_lane.id_ == LaneId.RightLane:
            return SemanticVehicle(vehicle, [LateralBehavior.LaneKeeping, LateralBehavior.LaneChangeLeft], nearest_lane)

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

    # Calculate leading vehicle, not limited to ego vehicle
    def getLeadingVehicle(self, cur_semantic_vehicle, semantic_vehicles):
        reference_lane = cur_semantic_vehicle.reference_lane_

        # Initialize
        min_diff = float('inf')
        leading_vehicle = None

        # Get ego vehicle index in reference
        ego_vehicle_index = reference_lane.calculateNearestIndexInLane(cur_semantic_vehicle.vehicle_.position_)

        # Traverse semantic vehicles
        for semantic_vehicle_id, other_semantic_vehicle in semantic_vehicles.items():
            if semantic_vehicle_id == cur_semantic_vehicle.vehicle_.id_:
                continue

            # Determine identical lane
            if other_semantic_vehicle.nearest_lane_.id_ == reference_lane.id_:
                other_vehicle_lane_index = other_semantic_vehicle.nearest_lane_.calculateNearestIndexInLane(
                    other_semantic_vehicle.vehicle_.position_)
                if other_vehicle_lane_index > ego_vehicle_index:
                    if other_vehicle_lane_index - ego_vehicle_index < min_diff:
                        min_diff = other_vehicle_lane_index - ego_vehicle_index
                        leading_vehicle = other_semantic_vehicle

        return leading_vehicle


# Vehicle class
class Vehicle:
    def __init__(self, vehicle_id, position, length, width, velocity, acceleration, time_stamp=None, curvature=0.0, steer=0.0):
        self.id_ = vehicle_id
        self.position_ = position
        self.length_ = length
        self.width_ = width
        self.velocity_ = velocity
        self.acceleration_ = acceleration
        self.time_stamp_ = time_stamp
        self.curvature_ = curvature
        self.steer_ = steer

        # Construct occupied rectangle
        self.rectangle_ = Rectangle(position, length, width)


class SemanticVehicle:
    def __init__(self, vehicle, potential_behaviors, nearest_lane, reference_lane=None):
        self.vehicle_ = vehicle
        self.potential_behaviors_ = potential_behaviors
        self.nearest_lane_ = nearest_lane
        self.reference_lane_ = reference_lane


# Rectangle class, denotes the area occupied by the vehicle
class Rectangle:
    def __init__(self, center_point, length, width):
        self.center_point_ = center_point
        self.length_ = length
        self.width_ = width

        # Load four vertex and two axes of rectangle
        self.vertex_ = []
        self.generateVertex()
        self.axes_ = []
        self.generateAxes()


    # Generate vertex
    def generateVertex(self):

        # Calculate four vertex position respectively
        point_1_x = self.center_point_.x_ + self.length_ * 0.5 * np.cos(self.center_point_.theta_) - self.width_ * 0.5 * np.sin(self.center_point_.theta_)
        point_1_y = self.center_point_.y_ + self.length_ * 0.5 * np.sin(self.center_point_.theta_) + self.width_ * 0.5 * np.cos(self.center_point_.theta_)
        point_2_x = self.center_point_.x_ + self.length_ * 0.5 * np.cos(self.center_point_.theta_) + self.width_ * 0.5 * np.sin(self.center_point_.theta_)
        point_2_y = self.center_point_.y_ + self.length_ * 0.5 * np.sin(self.center_point_.theta_) - self.width_ * 0.5 * np.cos(self.center_point_.theta_)
        point_3_x = self.center_point_.x_ - self.length_ * 0.5 * np.cos(self.center_point_.theta_) + self.width_ * 0.5 * np.sin(self.center_point_.theta_)
        point_3_y = self.center_point_.y_ - self.length_ * 0.5 * np.sin(self.center_point_.theta_) - self.width_ * 0.5 * np.cos(self.center_point_.theta_)
        point_4_x = self.center_point_.x_ - self.length_ * 0.5 * np.cos(self.center_point_.theta_) - self.width_ * 0.5 * np.sin(self.center_point_.theta_)
        point_4_y = self.center_point_.y_ - self.length_ * 0.5 * np.sin(self.center_point_.theta_) + self.width_ * 0.5 * np.cos(self.center_point_.theta_)

        # Store
        self.vertex_.append([point_1_x, point_1_y])
        self.vertex_.append([point_2_x, point_2_y])
        self.vertex_.append([point_3_x, point_3_y])
        self.vertex_.append([point_4_x, point_4_y])

        self.vertex_ = np.array(self.vertex_)

    # Generate axes
    def generateAxes(self):
        # For the first two vertex
        vec_1 = np.array([self.vertex_[1][0] - self.vertex_[0][0], self.vertex_[1][1] - self.vertex_[0][1]])
        length_1 = np.linalg.norm(vec_1)
        normalized_vec_1 = vec_1 / length_1
        self.axes_.append([-normalized_vec_1[1], normalized_vec_1[0]])

        # For the second and third vertex
        vec_2 = np.array([self.vertex_[2][0] - self.vertex_[1][0], self.vertex_[2][1] - self.vertex_[1][1]])
        length_2 = np.linalg.norm(vec_2)
        normalized_vec_2 = vec_2 / length_2
        self.axes_.append([-normalized_vec_2[1], normalized_vec_2[0]])

        self.axes_ = np.array(self.axes_)



    # Judge collision
    @classmethod
    def isCollision(cls, rectangle_1, rectangle_2):

        # Get vertex of two rectangles
        rectangle_1_vertex = rectangle_1.vertex_
        rectangle_2_vertex = rectangle_2.vertex_

        # Pooling axes of two rectangle
        axes = np.vstack((rectangle_1.axes_, rectangle_2.axes_))

        # Traverse axis
        for axis in axes:
            # Get projection
            proj_1 = Tools.getProjectionOnVertex(rectangle_1_vertex, axis)
            proj_2 = Tools.getProjectionOnVertex(rectangle_2_vertex, axis)

            # Calculate overlap length
            overlap_length = Tools.getOverlapLength(proj_1, proj_2)

            if abs(overlap_length) < 1e-6:
                return False

        return True

# Ideal steer model
class IdealSteerModel:

    def __init__(self, wheelbase_len, max_lon_acc, max_lon_dec, max_lon_acc_jerk, max_lon_dec_jerk, max_lat_acc, max_lat_jerk, max_steering_angle, max_steer_rate, max_curvature):
        self.wheelbase_len_ = wheelbase_len
        self.max_lon_acc_ = max_lon_acc
        self.max_lon_dec_ = max_lon_dec
        self.max_lon_acc_jerk_ = max_lon_acc_jerk
        self.max_lon_dec_jerk_ = max_lon_dec_jerk
        self.max_lat_acc_ = max_lat_acc
        self.max_lat_jerk_ = max_lat_jerk
        self.max_steering_angle_ = max_steering_angle
        self.max_steer_rate_ = max_steer_rate
        self.max_curvature_ = max_curvature

        # Initialize shell
        self.control_ = []
        self.state_ = None
        # Internal_state[0] means x position, internal_state[1] means y position, internal_state[2] means angle, internal_state[3] means velocity, internal_state[4] means steer
        self.internal_state_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.desired_lon_acc_ = 0.0
        self.desired_lat_acc_ = 0.0
        self.desired_steer_rate_ = 0.0

    # Set control information, control[0] means steer, control[1] means velocity
    def setControl(self, control):
        self.control_ = control

    # Set state information, use vehicle class represent vehicle state
    def setState(self, vehicle):
        self.state_ = copy.deepcopy(vehicle)

    # Truncate control
    def truncateControl(self, dt):
        self.desired_lon_acc_ = (self.control_[1] - self.state_.velocity_) / dt
        desired_lon_jerk = (self.desired_lon_acc_ - self.state_.acceleration_) / dt
        desired_lon_jerk = Tools.truncate(desired_lon_jerk, -self.max_lon_dec_jerk_, self.max_lon_acc_jerk_)
        self.desired_lon_acc_ = desired_lon_jerk * dt + self.state_.acceleration_
        self.desired_lon_acc_ = Tools.truncate(self.desired_lon_acc_, -self.max_lon_dec_, self.max_lon_acc_)
        self.control_[1] = max(self.state_.velocity_ + self.desired_lon_acc_ * dt, 0.0)
        self.desired_lat_acc_ = pow(self.control_[1], 2) * (np.tan(self.control_[0]) / self.wheelbase_len_)
        lat_acc_ori = pow(self.state_.velocity_, 2.0) * self.state_.curvature_
        lat_jerk_desired = (self.desired_lat_acc_ - lat_acc_ori) / dt
        lat_jerk_desired = Tools.truncate(lat_jerk_desired, -self.max_lat_jerk_, self.max_lat_jerk_)
        desired_lat_acc_ = lat_jerk_desired * dt + lat_acc_ori
        desired_lat_acc_ = Tools.truncate(desired_lat_acc_, -self.max_lat_acc_, self.max_lat_acc_)
        self.control_[0] = np.arctan(
            desired_lat_acc_ * self.wheelbase_len_ / max(pow(self.control_[1], 2.0), 0.1 * Config.BigEPS))
        self.desired_steer_rate_ = Tools.normalizeAngle(self.control_[0] - self.state_.steer_) / dt
        self.desired_steer_rate_ = Tools.truncate(self.desired_steer_rate_, -self.max_steer_rate_, self.max_steer_rate_)
        self.control_[0] = Tools.normalizeAngle(self.state_.steer_ + self.desired_steer_rate_ * dt)

    # Forward once
    def step(self, dt):
        self.state_.steer_ = np.arctan(self.state_.curvature_ * self.wheelbase_len_)
        self.updateInternalState()
        self.control_[1] = max(0.0, self.control_[1])
        self.control_[0] = Tools.truncate(self.control_[0], -self.max_steering_angle_, self.max_steering_angle_)
        self.truncateControl(dt)
        self.desired_lon_acc_ = (self.control_[1] - self.state_.velocity_) / dt

        # For DEBUG
        # print('Final control velocity input: {}'.format(self.control_[1]))
        # print('Final state velocity input: {}'.format(self.state_.velocity_))
        # print('Final desired longitudinal acceleration: {}'.format(self.desired_lon_acc_))
        self.desired_steer_rate_ = Tools.normalizeAngle(self.control_[0] - self.state_.steer_)

        # Linear predict function
        # Probably need to fix bug
        def linearPredict(internal_state, dt):
            predict_state = [0.0 for _ in range(5)]
            predict_state[0] = internal_state[0] + dt * np.cos(internal_state[2]) * internal_state[3]
            predict_state[1] = internal_state[1] + dt * np.sin(internal_state[2]) * internal_state[3]
            predict_state[2] = np.tan(internal_state[4]) * internal_state[3] / self.wheelbase_len_
            predict_state[3] = internal_state[3] + dt * self.desired_lon_acc_
            predict_state[4] = internal_state[4] + dt * self.desired_steer_rate_
            return predict_state

        # Generate predict state
        predict_state = copy.deepcopy(self.internal_state_)
        iteration_num = 40
        for _ in range(0, iteration_num):
            predict_state = linearPredict(predict_state, dt / iteration_num)
        predict_state_position = PathPoint(predict_state[0], predict_state[1], Tools.normalizeAngle(predict_state[2]))

        # self.state_.position_.x_ = predict_state[0]
        # self.state_.position_.y_ = predict_state[1]
        # self.state_.position_.theta_ = Tools.normalizeAngle(predict_state[2])
        # self.state_.velocity_ = predict_state[3]
        # self.state_.steer_ = Tools.normalizeAngle(predict_state[4])
        # self.state_.curvature_ = np.tan(self.state_.steer_) * 1.0 / self.wheelbase_len_
        # self.state_.acceleration_ = self.desired_lon_acc_

        self.state_ = Vehicle(self.state_.id_, predict_state_position, self.state_.length_, self.state_.width_, predict_state[3], self.desired_lon_acc_, None, np.tan(predict_state[4]) * 1.0 / self.wheelbase_len_, Tools.normalizeAngle(predict_state[4]))

        self.updateInternalState()

    # Update internal state
    def updateInternalState(self):
        self.internal_state_[0] = self.state_.position_.x_
        self.internal_state_[1] = self.state_.position_.y_
        self.internal_state_[2] = self.state_.position_.theta_
        self.internal_state_[3] = self.state_.velocity_
        self.internal_state_[4] = self.state_.steer_