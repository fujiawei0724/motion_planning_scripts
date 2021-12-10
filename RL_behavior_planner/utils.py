# -- coding: utf-8 --
# @Time : 2021/11/14 下午6:49
# @Author : fujiawei0724
# @File : utils.py
# @Software: PyCharm

"""
The components for RL behavior planner.
"""
import numpy as np
import time
import copy
import math
import random
from enum import Enum, unique
from collections import defaultdict
from scipy.integrate import odeint


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
    steer_control_gain = 3.0
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

    # Refresh vehicles information in semantic vehicles
    @staticmethod
    def refreshSemanticVehicles(semantic_vehicles, vehicles):
        for veh_id, veh in vehicles.items():
            semantic_vehicles[veh_id].vehicle_ = veh

    # Judge initial situation available, delete the initial situation where there are collisions
    @staticmethod
    def checkInitSituation(ego_vehicle, surround_vehicles):
        # Get all vehicles
        all_vehicles = []
        all_vehicles.append(ego_vehicle)
        for sur_veh_id, sur_veh in surround_vehicles.items():
            all_vehicles.append(sur_veh)

        # Calculate collision between each two vehicles
        vehicles_num = len(all_vehicles)
        for i in range(0, vehicles_num):
            for j in range(i + 1, vehicles_num):
                if Rectangle.isCollision(all_vehicles[i].rectangle_, all_vehicles[j].rectangle_):
                    return False
        return True


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

# Vehicle behavior contains latitudinal behavior and longitudinal speed compensation
"""
The candidates for compensation is np.arange(-5.0, 5.0 + EPS, 1.0), probably need to adjust .
"""
class VehicleIntention:
    def __init__(self, lat_beh, velocity_compensation):
        self.lat_beh_ = lat_beh
        self.velocity_compensation_ = velocity_compensation

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

    # Print information for data transform
    def printInfo(self):
        info = []
        info.append(self.beh_seq_[0].lon_beh_.value)
        for _, veh_beh in enumerate(self.beh_seq_):
            info.append(veh_beh.lat_beh_.value)
        # print(info)
        return info


# Define intention sequence for construct intention space
class IntentionSequence:
    def __init__(self, intention_sequence):
        self.intention_seq_ = intention_sequence

    # DEBUG: print information
    def print(self):
        for veh_intention_index, veh_intention in enumerate(self.intention_seq_):
            print('Single intention index: {}, lateral behavior: {}, longitudinal speed compensation: {}'.format(veh_intention_index, veh_intention.lat_beh_, veh_intention.velocity_compensation_))

    # Print information for data transform
    def printInfo(self):
        info = []
        info.append(int(self.intention_seq_[0].velocity_compensation_))
        for _, veh_intention in enumerate(self.intention_seq_):
            info.append(veh_intention.lat_beh_.value)
        return info


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

    # Construct vehicle intention set
    def generateIntends(self):
        veh_intention_set = []
        lon_vel_comp_candidates = np.arange(-5.0, 5.0 + 1e-7, 1.0)

        # Traverse longitudinal velocity compensation
        for lon_vel_comp in lon_vel_comp_candidates:
            cur_intention_sequence = []
            for intention_index in range(0, self.seq_length_):
                for lat_beh in LateralBehavior:
                    if lat_beh != LateralBehavior.LaneKeeping:
                        # Add lane change situations
                        veh_intention_set.append(self.addIntention(cur_intention_sequence, lon_vel_comp, lat_beh, self.seq_length_ - intention_index))
                cur_intention_sequence.append(VehicleIntention(LateralBehavior.LaneKeeping, lon_vel_comp))
            veh_intention_set.append(IntentionSequence(cur_intention_sequence))

        return veh_intention_set

    # Add lane change situation which start from intermediate time stamp
    @classmethod
    def addBehavior(cls, cur_beh_seq, lon_beh, lat_beh, num):

        # Initialize
        res_beh_seq = copy.deepcopy(cur_beh_seq)

        # Add lane change behavior
        for i in range(0, num):
            res_beh_seq.append(VehicleBehavior(lat_beh, lon_beh))

        return BehaviorSequence(res_beh_seq)

    # Add lane change situation which start from intermediate time stamp
    @classmethod
    def addIntention(cls, cur_intention_seq, vel_comp, lat_beh, num):

        # Initialize
        res_intention_seq = copy.deepcopy(cur_intention_seq)

        # Add lane change behavior
        for i in range(0, num):
            res_intention_seq.append(VehicleIntention(lat_beh, vel_comp))

        return IntentionSequence(res_intention_seq)


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
            lane_path_points.append(
                PathPoint(start_point.x_ + sample * x_diff, start_point.y_ + sample * y_diff, lane_theta))
            lane_left_boundary_points.append(
                [start_point.x_ + sample * x_diff + np.cos(lane_theta + math.pi / 2.0) * lane_width / 2.0,
                 start_point.y_ + sample * y_diff + np.sin(lane_theta + math.pi / 2.0) * lane_width / 2.0])
            lane_right_boundary_points.append(
                [start_point.x_ + sample * x_diff + np.cos(lane_theta - math.pi / 2.0) * lane_width / 2.0,
                 start_point.y_ + sample * y_diff + np.sin(lane_theta - math.pi / 2.0) * lane_width / 2.0])
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

    # Get semantic vehicles
    def getSemanticVehicles(self, vehicles):
        semantic_vehicles = dict()
        for veh_id, vehicle in vehicles.items():
            if veh_id != 0:
                # For surround vehicle
                semantic_vehicle = self.calculateSurroundVehicleBehavior(vehicle)
                semantic_vehicles[semantic_vehicle.vehicle_.id_] = semantic_vehicle
            else:
                # For ego vehicle
                semantic_vehicle = self.calculateEgoVehicleBehavior(vehicle)
                semantic_vehicles[semantic_vehicle.vehicle_.id_] = semantic_vehicle

        return semantic_vehicles

    # Get semantic vehicle
    def getSingleSemanticVehicle(self, vehicle):
        if vehicle.id_ == 0:
            return self.calculateEgoVehicleBehavior(vehicle)
        else:
            return self.calculateSurroundVehicleBehavior(vehicle)

    # Reset ego semantic vehicle from the given potential behavior
    def resetEgoSemanticVehicle(self, vehicle, potential_behavior):
        assert vehicle.id_ == 0
        # Find the nearest lane
        nearest_lane = self.findNearestLane(vehicle.position_)

        if nearest_lane.id_ == LaneId.CenterLane:
            if potential_behavior == LateralBehavior.LaneChangeLeft and LaneId.LeftLane in self.lanes_:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeLeft, nearest_lane, self.lanes_[LaneId.LeftLane])
            elif potential_behavior == LateralBehavior.LaneChangeRight and LaneId.RightLane in self.lanes_:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeRight, nearest_lane, self.lanes_[LaneId.RightLane])
            else:
                return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, nearest_lane)

        elif nearest_lane.id_ == LaneId.LeftLane:
            if potential_behavior == LateralBehavior.LaneChangeRight and LaneId.CenterLane in self.lanes_:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeRight, nearest_lane, self.lanes_[LaneId.CenterLane])
            else:
                return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, nearest_lane)

        elif nearest_lane.id_ == LaneId.RightLane:
            if potential_behavior == LateralBehavior.LaneChangeLeft and LaneId.CenterLane in self.lanes_:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeLeft, nearest_lane, self.lanes_[LaneId.CenterLane])
            else:
                return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, nearest_lane)
        else:
            assert False

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
            if nearest_lane.id_ == LaneId.CenterLane and LaneId.LeftLane in self.lanes_:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeLeft, nearest_lane,
                                       self.lanes_[LaneId.LeftLane])
            elif nearest_lane.id_ == LaneId.LeftLane:
                # Error situation, set lane keeping
                return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, nearest_lane)
            else:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeLeft, nearest_lane,
                                       self.lanes_[LaneId.CenterLane])


        elif lateral_distance <= -Config.lateral_distance_threshold and lateral_velocity <= -Config.lateral_velocity_threshold:
            # For change right
            if nearest_lane.id_ == LaneId.CenterLane and LaneId.RightLane in self.lanes_:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeRight, nearest_lane,
                                       self.lanes_[LaneId.RightLane])
            elif nearest_lane.id_ == LaneId.LeftLane:
                return SemanticVehicle(vehicle, LateralBehavior.LaneChangeRight, nearest_lane,
                                       self.lanes_[LaneId.CenterLane])
            else:
                # Error situation, set lane keeping
                return SemanticVehicle(vehicle, LateralBehavior.LaneKeeping, nearest_lane, nearest_lane)

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
    def __init__(self, vehicle_id, position, length, width, velocity, acceleration, time_stamp=None, curvature=0.0,
                 steer=0.0):
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

    # DEBUG
    def print(self):
        print('Id: {}'.format(self.id_))
        print('Position x: {}'.format(self.position_.x_))
        print('Position y: {}'.format(self.position_.y_))
        print('Theta: {}'.format(self.position_.theta_))
        print('Length: {}'.format(self.length_))
        print('Width: {}'.format(self.width_))
        print('Velocity: {}'.format(self.velocity_))
        print('Acceleration: {}'.format(self.acceleration_))
        print('Time stamp: {}'.format(self.time_stamp_))
        print('Curvature: {}'.format(self.curvature_))
        print('Steer: {}'.format(self.steer_))


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
        point_1_x = self.center_point_.x_ + self.length_ * 0.5 * np.cos(
            self.center_point_.theta_) - self.width_ * 0.5 * np.sin(self.center_point_.theta_)
        point_1_y = self.center_point_.y_ + self.length_ * 0.5 * np.sin(
            self.center_point_.theta_) + self.width_ * 0.5 * np.cos(self.center_point_.theta_)
        point_2_x = self.center_point_.x_ + self.length_ * 0.5 * np.cos(
            self.center_point_.theta_) + self.width_ * 0.5 * np.sin(self.center_point_.theta_)
        point_2_y = self.center_point_.y_ + self.length_ * 0.5 * np.sin(
            self.center_point_.theta_) - self.width_ * 0.5 * np.cos(self.center_point_.theta_)
        point_3_x = self.center_point_.x_ - self.length_ * 0.5 * np.cos(
            self.center_point_.theta_) + self.width_ * 0.5 * np.sin(self.center_point_.theta_)
        point_3_y = self.center_point_.y_ - self.length_ * 0.5 * np.sin(
            self.center_point_.theta_) - self.width_ * 0.5 * np.cos(self.center_point_.theta_)
        point_4_x = self.center_point_.x_ - self.length_ * 0.5 * np.cos(
            self.center_point_.theta_) - self.width_ * 0.5 * np.sin(self.center_point_.theta_)
        point_4_y = self.center_point_.y_ - self.length_ * 0.5 * np.sin(
            self.center_point_.theta_) + self.width_ * 0.5 * np.cos(self.center_point_.theta_)

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

    def __init__(self, wheelbase_len, max_lon_acc, max_lon_dec, max_lon_acc_jerk, max_lon_dec_jerk, max_lat_acc,
                 max_lat_jerk, max_steering_angle, max_steer_rate, max_curvature):
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
    def step(self, dt, linear_prediction=False):
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

        predict_state = None
        if linear_prediction:
            # Linear predict function
            # Probably need to fix bug
            def linearPredict(internal_state, dt):
                predict_state = [0.0 for _ in range(5)]
                predict_state[0] = internal_state[0] + dt * np.cos(internal_state[2]) * internal_state[3]
                predict_state[1] = internal_state[1] + dt * np.sin(internal_state[2]) * internal_state[3]
                predict_state[2] = internal_state[2] + dt * np.tan(internal_state[4]) * internal_state[
                    3] / self.wheelbase_len_
                predict_state[3] = internal_state[3] + dt * self.desired_lon_acc_
                predict_state[4] = internal_state[4] + dt * self.desired_steer_rate_
                return predict_state

            # Generate predict state
            predict_state = copy.deepcopy(self.internal_state_)
            iteration_num = 40
            for _ in range(0, iteration_num):
                predict_state = linearPredict(predict_state, dt / iteration_num)

        else:
            # Integration based prediction
            # Define derivative
            def deriv(state, t):
                state_deriv = np.zeros((5,))
                state_deriv[0] = np.cos(state[2]) * state[3]
                state_deriv[1] = np.sin(state[2]) * state[3]
                state_deriv[2] = np.tan(state[4]) * state[3] / self.wheelbase_len_
                state_deriv[3] = self.desired_lon_acc_
                state_deriv[4] = self.desired_steer_rate_
                return state_deriv

            def predict(start_state, t):
                return odeint(deriv, start_state, t)

            t = np.array([0.0, dt])
            start_state = copy.deepcopy(self.internal_state_)
            predict_state_sequence = predict(start_state, t)
            predict_state = predict_state_sequence[1]

        assert predict_state is not None

        predict_state_position = PathPoint(predict_state[0], predict_state[1], Tools.normalizeAngle(predict_state[2]))

        self.state_ = Vehicle(self.state_.id_, predict_state_position, self.state_.length_, self.state_.width_,
                              predict_state[3], self.desired_lon_acc_, None,
                              np.tan(predict_state[4]) * 1.0 / self.wheelbase_len_,
                              Tools.normalizeAngle(predict_state[4]))

        self.updateInternalState()

    # Update internal state
    def updateInternalState(self):
        self.internal_state_[0] = self.state_.position_.x_
        self.internal_state_[1] = self.state_.position_.y_
        self.internal_state_[2] = self.state_.position_.theta_
        self.internal_state_[3] = self.state_.velocity_
        self.internal_state_[4] = self.state_.steer_


# Forward simulation
class ForwardExtender:

    def __init__(self, lane_server: LaneServer, dt, predict_time_span):
        # Information cache
        self.lane_server_ = copy.deepcopy(lane_server)
        self.dt_ = dt
        self.predict_time_span_ = predict_time_span

    # Forward extend with interaction among vehicles
    def multiAgentForward(self, ego_potential_behavior_sequence, vehicles, lane_speed_limit):

        # Determine ego vehicle id
        ego_vehicle_id = 0

        # Initialize current vehicle states and surround vehicle states in different time stamp
        ego_tra = []
        surround_tras = defaultdict(list)
        ego_tra.append(vehicles[0])
        for this_vehicle_id, this_vehicle in vehicles.items():
            if this_vehicle_id == ego_vehicle_id:
                continue
            else:
                surround_tras[this_vehicle_id].append(this_vehicle)

        # State cache
        cur_vehicles = copy.deepcopy(vehicles)

        # Determine number of forward update
        num_steps_forward = int(self.predict_time_span_ / self.dt_)
        assert len(ego_potential_behavior_sequence.beh_seq_) == num_steps_forward

        # Determine longitudinal behavior or longitudinal velocity compensation
        longitudinal_behavior = None
        longitudinal_velocity_compensation = None
        if isinstance(ego_potential_behavior_sequence, BehaviorSequence):
            longitudinal_behavior = ego_potential_behavior_sequence.beh_seq_[0].lon_beh_
        elif isinstance(ego_potential_behavior_sequence, IntentionSequence):
            longitudinal_velocity_compensation = ego_potential_behavior_sequence.intention_seq_[0].velocity_compensation_
        else:
            assert False

        # Get the initial velocities of all vehicles
        initial_velocities = dict()
        for veh_id, veh in vehicles.items():
            initial_velocities[veh_id] = veh.velocity_

        # Get initial semantic vehicles
        cur_semantic_vehicles = self.lane_server_.getSemanticVehicles(vehicles)

        # Start forward simulation
        for step_index in range(0, num_steps_forward):

            print('No. {} epoch forward calculating'.format(step_index + 1))

            # Initialize cache
            states_cache = {}

            for veh_id, veh in cur_vehicles.items():

                # print('Predicting vehicle id: {}'.format(veh_id))

                # Determine initial vehicles information
                desired_velocity = initial_velocities[veh_id]
                # init_time_stamp = veh.vehicle_.time_stamp_
                if veh_id == ego_vehicle_id:
                    if isinstance(ego_potential_behavior_sequence, IntentionSequence):
                        if longitudinal_behavior == LongitudinalBehavior.Conservative:
                            desired_velocity = max(0.0, desired_velocity - 5.0)
                        elif longitudinal_behavior == LongitudinalBehavior.Normal:
                            desired_velocity += 0.0
                        elif longitudinal_behavior == LongitudinalBehavior.Aggressive:
                            desired_velocity = min(lane_speed_limit, desired_velocity + 5.0)
                        else:
                            assert False
                    elif isinstance(ego_potential_behavior_sequence, IntentionSequence):
                        desired_velocity += longitudinal_velocity_compensation
                        desired_velocity = np.clip(desired_velocity, 0.0, lane_speed_limit)

                # TODO: set vehicles speed limits from reference lane speed limit
                desired_veh_state = self.forwardOnce(veh_id, ego_potential_behavior_sequence.beh_seq_[step_index].lat_beh_, cur_semantic_vehicles, desired_velocity)

                # Cache
                states_cache[desired_veh_state.id_] = desired_veh_state

            # Update current vehicle information
            cur_vehicles = copy.deepcopy(states_cache)
            Tools.refreshSemanticVehicles(cur_semantic_vehicles, cur_vehicles)

            # Store trajectories
            for vehicle_id, state in states_cache.items():
                if vehicle_id == ego_vehicle_id:
                    ego_tra.append(state)
                else:
                    surround_tras[vehicle_id].append(state)

        # Construct trajectories
        ego_trajectory = Trajectory(ego_tra)
        surround_trajectories = {}
        for vehicle_id, vehicle_tra in surround_tras.items():
            if vehicle_id == ego_vehicle_id:
                assert False
            surround_trajectories[vehicle_id] = Trajectory(vehicle_tra)

        return ego_trajectory, surround_trajectories

    # Forward extend without interaction among vehicles
    def openLoopForward(self):
        pass

    # Forward once from current state
    # Note that vehicles include ego vehicle
    def forwardOnce(self, cur_id, ego_potential_behavior, semantic_vehicles, desired_velocity):
        # Calculate all semantic vehicles and set ego potential behavior
        semantic_vehicles[0] = self.lane_server_.resetEgoSemanticVehicle(semantic_vehicles[0].vehicle_, ego_potential_behavior)

        # Get current semantic vehicle
        cur_semantic_vehicle = semantic_vehicles[cur_id]

        # Calculate steer
        steer = self.calculateSteer(cur_semantic_vehicle)

        # Calculate velocity
        velocity = self.calculateVelocity(cur_semantic_vehicle, semantic_vehicles, self.dt_, desired_velocity)

        # Calculate desired state
        desired_vehicle_state = self.calculateDesiredState(cur_semantic_vehicle, steer, velocity, self.dt_)

        return desired_vehicle_state

    # Calculate steer
    def calculateSteer(self, semantic_vehicle):
        # Determine look ahead distance in reference lane
        look_ahead_distance = min(
            max(Config.look_ahead_min_distance, semantic_vehicle.vehicle_.velocity_ * Config.steer_control_gain),
            Config.look_ahead_max_distance)

        # Calculate nearest path point in reference lane
        nearest_path_point = semantic_vehicle.reference_lane_.path_points_[
            semantic_vehicle.reference_lane_.calculateNearestIndexInLane(semantic_vehicle.vehicle_.position_)]

        # Calculate target path point in reference lane
        target_path_point_in_reference_lane = semantic_vehicle.reference_lane_.calculateTargetDistancePoint(
            nearest_path_point, look_ahead_distance)

        # Calculate look ahead distance in world frame
        look_ahead_distance_world = target_path_point_in_reference_lane.calculateDistance(
            semantic_vehicle.vehicle_.position_)

        # Calculate target angle and diff angle
        target_angle = np.arctan2(target_path_point_in_reference_lane.y_ - semantic_vehicle.vehicle_.position_.y_,
                                  target_path_point_in_reference_lane.x_ - semantic_vehicle.vehicle_.position_.x_)
        diff_angle = Tools.normalizeAngle(target_angle - semantic_vehicle.vehicle_.position_.theta_)

        # Calculate target steer
        target_steer = Tools.calculateSteer(Config.wheelbase_length, diff_angle, look_ahead_distance_world)

        return target_steer

    # Calculate velocity
    def calculateVelocity(self, ego_semantic_vehicle, semantic_vehicles, dt, desired_velocity):
        # Calculate leading vehicle
        leading_semantic_vehicle = self.lane_server_.getLeadingVehicle(ego_semantic_vehicle, semantic_vehicles)

        # Judge leading vehicle state
        if leading_semantic_vehicle is None:
            # Don't exist leading vehicle, using virtual leading vehicle
            virtual_leading_vehicle_distance = 100.0 + 100.0 * ego_semantic_vehicle.vehicle_.velocity_
            target_velocity = IDM.calculateVelocity(0.0, virtual_leading_vehicle_distance,
                                                    ego_semantic_vehicle.vehicle_.velocity_,
                                                    ego_semantic_vehicle.vehicle_.velocity_, dt, desired_velocity)
        else:
            # With leading vehicle
            # Calculate ego vehicle and leading vehicle's nearest position in corresponding lane respectively
            assert ego_semantic_vehicle.reference_lane_ == leading_semantic_vehicle.nearest_lane_
            corresponding_lane_ = ego_semantic_vehicle.reference_lane_
            ego_vehicle_position_in_corresponding_lane = corresponding_lane_.calculateNearestPointInLane(
                ego_semantic_vehicle.vehicle_.position_)
            leading_vehicle_position_in_corresponding_lane = corresponding_lane_.calculateNearestPointInLane(
                leading_semantic_vehicle.vehicle_.position_)

            # Calculate the distance between ego vehicle and ego vehicle
            ego_leading_vehicles_distance = ego_vehicle_position_in_corresponding_lane.calculateDistance(
                leading_vehicle_position_in_corresponding_lane)
            target_velocity = IDM.calculateVelocity(0.0, ego_leading_vehicles_distance,
                                                    ego_semantic_vehicle.vehicle_.velocity_,
                                                    leading_semantic_vehicle.vehicle_.velocity_, dt, desired_velocity)

        return target_velocity

    # Calculate desired state
    def calculateDesiredState(self, semantic_vehicle, steer, velocity, dt):

        # Load parameters for ideal steer model
        # Wheelbase len need to fix, for different vehicles, their wheelbase length are different
        ideal_steer_model = IdealSteerModel(Config.wheelbase_length, IDM.acceleration, IDM.hard_braking_deceleration,
                                            Config.max_lon_acc_jerk, Config.max_lon_brake_jerk,
                                            Config.max_lat_acceleration_abs, Config.max_lat_jerk_abs,
                                            Config.max_steer_angle_abs, Config.max_steer_rate, Config.max_curvature_abs)
        ideal_steer_model.setState(semantic_vehicle.vehicle_)
        ideal_steer_model.setControl([steer, velocity])
        ideal_steer_model.step(dt)

        # Calculate predicted vehicle state (the state of a vehicle belongs to vehicle class)
        predicted_state = ideal_steer_model.state_
        predicted_state.time_stamp_ = semantic_vehicle.vehicle_.time_stamp_ + dt

        return predicted_state


# Trajectory class, includes
class Trajectory:
    def __init__(self, vehicle_states):
        self.vehicle_states_ = vehicle_states

    # Calculate safety cost
    def calculateSafetyCost(self, judge_trajectory):
        assert len(self.vehicle_states_) == len(judge_trajectory.vehicle_states_)

        # Initialize safety cost
        safety_cost = 0.0
        final_collision = False

        # Judge collision
        for time_index in range(0, len(self.vehicle_states_)):
            assert self.vehicle_states_[time_index].time_stamp_ == judge_trajectory.vehicle_states_[
                time_index].time_stamp_

            # Judge whether collision
            is_collision = Rectangle.isCollision(self.vehicle_states_[time_index].rectangle_,
                                                 judge_trajectory.vehicle_states_[time_index].rectangle_)
            if is_collision:
                safety_cost += 0.01 * abs(
                    self.vehicle_states_[time_index].velocity_ - judge_trajectory.vehicle_states_[
                        time_index].velocity_) * 0.5
                final_collision = True

        return safety_cost, final_collision


# IDM model
# TODO: parameters need to adjust the situation
class IDM:
    # IDM config
    # desired_velocity = 10.0 # Update with the lane information, vehicle information and user designed
    vehicle_length = 5.0
    minimum_spacing = 2.0
    desired_headaway_time = 1.0
    acceleration = 2.0
    comfortable_braking_deceleration = 3.0
    hard_braking_deceleration = 5.0
    exponent = 4

    # Calculate velocity using IDM model with linear function
    @staticmethod
    def calculateVelocity(input_cur_s, input_leading_s, input_cur_velocity, input_leading_velocity, dt,
                          desired_velocity, linear_prediction=False):
        predicted_cur_velocity = None
        if linear_prediction:
            # Linear predict function
            def linearPredict(cur_s, leading_s, cur_velocity, leading_velocity, dt):
                # Calculate responding acceleration
                acc = IDM.calculateAcceleration(cur_s, leading_s, cur_velocity, leading_velocity, desired_velocity)
                acc = max(acc, -min(IDM.hard_braking_deceleration, cur_velocity / dt))
                next_cur_s = cur_s + cur_velocity * dt + 0.5 * acc * dt * dt
                next_leading_s = leading_s + leading_velocity * dt
                next_cur_velocity = cur_velocity + acc * dt
                next_leading_velocity = leading_velocity
                return next_cur_s, next_leading_s, next_cur_velocity, next_leading_velocity

            # State cache
            predicted_cur_s, predicted_leading_s, predicted_cur_velocity, predicted_leading_velocity = input_cur_s, input_leading_s, input_cur_velocity, input_leading_velocity

            # Predict 40 step with the time gap 0.01
            iteration_num = 40
            for _ in range(iteration_num):
                predicted_cur_s, predicted_leading_s, predicted_cur_velocity, predicted_leading_velocity = linearPredict(
                    predicted_cur_s, predicted_leading_s, predicted_cur_velocity, predicted_leading_velocity,
                    dt / iteration_num)
        else:
            # Define derivative
            def deriv(state, t):
                state_deriv = np.zeros((4, ))
                # Split state
                cur_s, leading_s, cur_velocity, leading_velocity = state[0], state[1], state[2], state[3]
                # Calculate responding acceleration
                acc = IDM.calculateAcceleration(cur_s, leading_s, cur_velocity, leading_velocity, desired_velocity)
                cur_s_deriv = cur_velocity + 0.5 * acc * t
                leading_s_deriv = leading_velocity
                cur_velocity_deriv = acc
                leading_velocity_deriv = 0.0
                state_deriv[0], state_deriv[1], state_deriv[2], state_deriv[3] = cur_s_deriv, leading_s_deriv, cur_velocity_deriv, leading_velocity_deriv
                return state_deriv

            def predict(start_state, t):
                return odeint(deriv, start_state, t)

            t = np.array([0.0, dt])
            start_state = np.array([input_cur_s, input_leading_s, input_cur_velocity, input_leading_velocity])
            predict_state_sequence = predict(start_state, t)
            predict_state = predict_state_sequence[1]
            predicted_cur_velocity = predict_state[2]

        assert predicted_cur_velocity is not None
        return predicted_cur_velocity

    # Calculate acceleration using IDM model
    @staticmethod
    def calculateAcceleration(cur_s, leading_s, cur_velocity, leading_velocity, desired_velocity):
        # Calculate parameters
        a_free = IDM.acceleration * (1 - pow(cur_velocity / (desired_velocity + Config.EPS),
                                             IDM.exponent)) if cur_velocity <= desired_velocity else -IDM.comfortable_braking_deceleration * (
                1 - pow(desired_velocity / (cur_velocity + Config.EPS),
                        IDM.acceleration * IDM.exponent / IDM.comfortable_braking_deceleration))
        s_alpha = max(0.0 + Config.EPS, leading_s - cur_s - IDM.vehicle_length)
        z = (IDM.minimum_spacing + max(0.0, cur_velocity * IDM.desired_headaway_time + cur_velocity * (
                cur_velocity - leading_velocity) / (2.0 * np.sqrt(
            IDM.acceleration * IDM.comfortable_braking_deceleration)))) / s_alpha

        # Calculate output acceleration
        if cur_velocity <= desired_velocity:
            a_out = IDM.acceleration * (1 - pow(z, 2)) if z >= 1.0 else a_free * (
                    1 - pow(z, 2.0 * IDM.acceleration / (a_free + Config.EPS)))
        else:
            a_out = a_free + IDM.acceleration * (1 - pow(z, 2)) if z >= 1.0 else a_free
        a_out = max(min(IDM.acceleration, a_out), -IDM.hard_braking_deceleration)

        return a_out


# Calculate a cost / reward for a policy
class PolicyEvaluator:
    @classmethod
    def praise(cls, ego_traj, sur_trajs, is_lane_changed, lane_speed_limit):
        safety_cost, is_collision = cls.calculateSafetyCost(ego_traj, sur_trajs, lane_speed_limit)
        lane_change_cost = cls.calculateLaneChangeCost(is_lane_changed)
        efficiency_cost = cls.calculateEfficiencyCost(ego_traj, lane_speed_limit)
        print('Safety cost: {}'.format(safety_cost))
        print('Lane change cost: {}'.format(lane_change_cost))
        print('Efficiency cost: {}'.format(efficiency_cost))
        return safety_cost + lane_change_cost + efficiency_cost, is_collision, safety_cost, lane_change_cost, efficiency_cost

    @classmethod
    def calculateLaneChangeCost(cls, is_lane_changed):
        return 0.3 if is_lane_changed else 0.0

    @classmethod
    def calculateSafetyCost(cls, ego_traj, sur_trajs, lane_speed_limit):
        safety_cost = 0.0
        is_collision = False
        for _, judge_sur_traj in sur_trajs.items():
            cur_safety_cost, is_cur_collision = ego_traj.calculateSafetyCost(judge_sur_traj)
            safety_cost += cur_safety_cost
            if is_cur_collision:
                is_collision = True
        if ego_traj.vehicle_states_[-1].velocity_ > lane_speed_limit:
            safety_cost += 100.0
        return safety_cost, is_collision

    # TODO: parameters need to change
    @classmethod
    def calculateEfficiencyCost(cls, ego_traj, lane_speed_limit):
        return (lane_speed_limit - ego_traj.vehicle_states_[-1].velocity_) / 10.0


# Agent vehicle generator (without ego vehicle)
class AgentGenerator:
    def __init__(self, left_lane_exist, right_lane_exist, center_left_distance, center_right_distance):
        self.y_boundary_up_ = 1.5 if not left_lane_exist else center_left_distance
        self.y_boundary_low_ = -1.5 if not right_lane_exist else -center_right_distance

    # Generate surround agents information
    def generateSingleAgent(self, index):
        agent_length = random.uniform(4.0, 6.0)
        agent_width = random.uniform(1.8, 2.5)
        agent_velocity = random.uniform(0.0, 25.0)
        agent_acceleration = random.uniform(-1.0, 1.0)

        x_position = random.uniform(0.0, 100.0)
        y_position = random.uniform(self.y_boundary_low_, self.y_boundary_up_)
        theta = random.uniform(-0.1, 0.1)
        agent_position = PathPoint(x_position, y_position, theta)
        this_vehicle = Vehicle(index, agent_position, agent_length, agent_width, agent_velocity,
                               agent_acceleration, 0.0)
        return this_vehicle

    def generateAgents(self, num):
        agents = {}
        for i in range(1, num + 1):
            this_vehicle = self.generateSingleAgent(i)
            agents[this_vehicle.id_] = this_vehicle
        return agents

# Ego vehicle train information generator
class EgoInfoGenerator:
    @staticmethod
    def generateOnce():
        # Define the constants
        length = 5.0
        width = 1.95
        curvature = 0.02
        steer = np.arctan(curvature * 2.8)
        return Vehicle(0, PathPoint(random.uniform(29.0, 31.0), random.uniform(-1.0, 1.0), random.uniform(-0.1, 0.1)), length, width, random.uniform(5.0, 7.0), random.uniform(-2.0, 1.5), 0.0, curvature, steer)

if __name__ == '__main__':
    beh_gene = BehaviorGenerator(10)
    intention_set = beh_gene.generateIntends()
    print('Intention set length: {}'.format(len(intention_set)))
    info = []
    for intention in intention_set:
        info.append(intention.printInfo())
    print(info)