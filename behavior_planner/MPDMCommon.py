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
# import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
# from collections import namedtuple
from collections import defaultdict
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
    def __init__(self):
        # Initialize shell
        self.lanes_ = None
        self.vehicles_ = None
        self.semantic_vehicles_ = None

    def refresh(self, lanes, vehicles):
        # Load data
        self.lanes_ = copy.deepcopy(lanes)
        self.vehicles_ = vehicles
        self.semantic_vehicles_ = {}
        self.initialize()

    # Initialize information
    def initialize(self):
        for vehicle in self.vehicles_:
            if vehicle.id_ != 0:
                # For surround vehicle
                semantic_vehicle = self.calculateSurroundVehicleBehavior(vehicle)
                self.semantic_vehicles_[semantic_vehicle.vehicle_.id_] = semantic_vehicle
            else:
                # For ego vehicle
                semantic_vehicle = self.calculateEgoVehicleBehavior(vehicle)
                self.semantic_vehicles_[semantic_vehicle.vehicle_.id_] = semantic_vehicle

    # Get ego vehicle's all potential behavior
    def getEgoVehicleBehaviors(self):
        return self.semantic_vehicles_[0].potential_behaviors_

    # Set ego vehicle potential behavior
    def setEgoVehicleBehavior(self, potential_behavior):
        self.semantic_vehicles_[0].potential_behaviors_ = potential_behavior
        self.calculateEgoVehicleReferenceLane(self.semantic_vehicles_[0], potential_behavior)

    # Update lane server information
    def update(self, vehicle_set, ego_potential_behavior):
        # Update vehicle information
        self.vehicles_.clear()
        self.vehicles_ = list(vehicle_set.values())

        # Update semantic vehicle information
        for sem_veh_id in self.semantic_vehicles_.keys():
            self.semantic_vehicles_[sem_veh_id].vehicle_ = vehicle_set[sem_veh_id]

        # Reset ego behavior for DCP tree
        self.setEgoVehicleBehavior(ego_potential_behavior)


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
                other_vehicle_lane_index = other_semantic_vehicle.nearest_lane_.calculateNearestIndexInLane(
                    other_semantic_vehicle.vehicle_.position_)
                if other_vehicle_lane_index > ego_vehicle_index:
                    if other_vehicle_lane_index - ego_vehicle_index < min_diff:
                        min_diff = other_vehicle_lane_index - ego_vehicle_index
                        leading_vehicle = other_semantic_vehicle

        return leading_vehicle


# Agent vehicle generator (without ego vehicle)
class AgentGenerator:
    def __init__(self, lanes=None):
        # TODO: for lane information inputting
        self.lanes_ = lanes

    # Generate surround agents information
    def generateSingleAgent(self, index):
        agent_length = random.uniform(4.0, 6.0)
        agent_width = random.uniform(1.8, 2.5)
        agent_velocity = random.uniform(3.0, 10.0)
        agent_acceleration = random.uniform(-1.0, 1.0)

        # TODO: calculate x, y position based on lanes information
        x_position = random.uniform(0.0, 100.0)
        y_position = random.uniform(-3.5, 3.5)
        theta = random.uniform(-0.2, 0.2)
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


# Forward simulation
class ForwardExtender:

    def __init__(self, lane_server, dt, predict_time_span):
        # Information cache
        self.lane_server_ = copy.deepcopy(lane_server)
        self.dt_ = dt
        self.predict_time_span_ = predict_time_span

    # Forward extend with interaction among vehicles
    def multiAgentForward(self, ego_potential_behavior):

        # Determine ego vehicle id
        ego_vehicle_id = 0
        # vehicle_set = copy.deepcopy(self.lane_server_.semantic_vehicles_)

        # Initialize current vehicle states and surround vehicle states in different time stamp
        ego_tra = []
        surround_tras = defaultdict(list)
        ego_tra.append(self.lane_server_.semantic_vehicles_[ego_vehicle_id].vehicle_)
        for this_vehicle_id, this_vehicle in self.lane_server_.semantic_vehicles_.items():
            if this_vehicle_id == ego_vehicle_id:
                continue
            else:
                surround_tras[this_vehicle_id].append(this_vehicle.vehicle_)

        # Determine number of forward update
        num_steps_forward = int(self.predict_time_span_ / self.dt_)

        # Get vehicles initial velocities
        initial_velocities = dict()
        for veh_id, sem_veh in self.lane_server_.semantic_vehicles_.items():
            initial_velocities[veh_id] = sem_veh.vehicle_.velocity_

        # Start forward simulation
        for step_index in range(0, num_steps_forward):

            print('Start No. {} prediction epoch'.format(step_index + 1))

            # Initialize cache
            states_cache = {}

            for veh_id, veh in self.lane_server_.semantic_vehicles_.items():

                print('Predicting vehicle id: {}'.format(veh_id))

                # Determine initial vehicles information
                desired_velocity = initial_velocities[veh_id]
                # init_time_stamp = veh.vehicle_.time_stamp_
                if veh_id == ego_vehicle_id:
                    # TODO: determine ego vehicle desired velocity
                    desired_velocity = Config.user_desired_velocity

                # # Determine other vehicle set
                # other_vehicle_set = {}
                # for veh_other_id, v_other in vehicle_set.items():
                #     if veh_other_id == veh_id:
                #         continue
                #     other_vehicle_set[veh_other_id] = v_other

                # TODO: set vehicles speed limits from reference lane speed limit
                desired_veh_state = self.forwardOnce(ego_potential_behavior, veh.vehicle_, desired_velocity)

                # Cache
                states_cache[desired_veh_state.id_] = desired_veh_state


            # Update information and lane server
            self.lane_server_.update(states_cache, ego_potential_behavior)


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
    def forwardOnce(self, ego_potential_behavior, cur_vehicle_state, desired_velocity):
        # Load ego potential behavior
        self.lane_server_.setEgoVehicleBehavior(ego_potential_behavior)

        # Load current semantic vehicle
        cur_semantic_vehicle = self.lane_server_.semantic_vehicles_[cur_vehicle_state.id_]

        # Calculate steer
        steer = self.calculateSteer(cur_semantic_vehicle)

        # Calculate velocity
        velocity = self.calculateVelocity(cur_semantic_vehicle, self.dt_, desired_velocity)

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
    def calculateVelocity(self, ego_semantic_vehicle, dt, desired_velocity):
        # Calculate leading vehicle
        leading_semantic_vehicle = self.lane_server_.getLeadingVehicle(ego_semantic_vehicle)

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
        ideal_steer_model = IdealSteerModel(Config.wheelbase_length, IDM.acceleration, IDM.hard_braking_deceleration, Config.max_lon_acc_jerk, Config.max_lon_brake_jerk, Config.max_lat_acceleration_abs, Config.max_lat_jerk_abs, Config.max_steer_angle_abs, Config.max_steer_rate, Config.max_curvature_abs)
        ideal_steer_model.setState(semantic_vehicle.vehicle_)
        ideal_steer_model.setControl([steer, velocity])
        ideal_steer_model.step(dt)

        # Calculate predicted vehicle state (the state of a vehicle belongs to vehicle class)
        predicted_state = ideal_steer_model.state_
        predicted_state.time_stamp_ = semantic_vehicle.vehicle_.time_stamp_ + dt

        return predicted_state


# Policy evaluate
class PolicyEvaluator:
    def __init__(self):
        # Initialize shell
        self.ego_potential_behavior_ = None
        self.ego_trajectory_ = None
        self.surround_trajectories_ = None

    # Load ego trajectory and surround trajectories
    def loadData(self, ego_potential_behavior, ego_trajectory, surround_trajectories):
        self.ego_potential_behavior_ = ego_potential_behavior
        self.ego_trajectory_ = ego_trajectory
        self.surround_trajectories_ = list(surround_trajectories.values())

    # Accumulate all the components of cost function
    def calculateCost(self):
        return self.calculateEfficiencyCost() + self.calculateSafetyCost() + self.calculateActionCost()

    # Efficiency cost
    def calculateEfficiencyCost(self):
        # Get the last state of ego vehicle
        ego_vehicle_last_state = self.ego_trajectory_.vehicle_states_[-1]
        last_velocity = ego_vehicle_last_state.velocity_
        return Config.user_desired_velocity - last_velocity


    # Safety cost
    def calculateSafetyCost(self):
        # Initialize
        safety_cost = 0.0

        # Traverse trajectory
        ego_vehicle_tra = self.ego_trajectory_
        for ego_vehicle_tra in self.surround_trajectories_:
            safety_cost += ego_vehicle_tra.calculateSafetyCost(ego_vehicle_tra)

        return safety_cost


    # Action cost
    def calculateActionCost(self):
        if self.ego_potential_behavior_ == LateralBehavior.LaneKeeping:
            return 0.0
        else:
            return 0.5

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
    def calculateVelocity(input_cur_s, input_leading_s, input_cur_velocity, input_leading_velocity, dt, desired_velocity):

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
            predicted_cur_s, predicted_leading_s, predicted_cur_velocity, predicted_leading_velocity = linearPredict(predicted_cur_s, predicted_leading_s, predicted_cur_velocity, predicted_leading_velocity, dt / iteration_num)

        # # Single step predict
        # _, _, predicted_cur_velocity, _ = linearPredict(input_cur_s, input_leading_s, input_cur_velocity, input_leading_velocity, 0.4)

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
            predict_state[2] = internal_state[2] + dt * np.tan(internal_state[4]) * internal_state[3] / self.wheelbase_len_
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


# Trajectory class, includes
class Trajectory:
    def __init__(self, vehicle_states):
        self.vehicle_states_ = vehicle_states

    # Calculate safety cost
    def calculateSafetyCost(self, judge_trajectory):
        assert len(self.vehicle_states_) == len(judge_trajectory.vehicle_states_)

        # Initialize safety cost
        safety_cost = 0.0

        # Judge collision
        for time_index in range(0, len(self.vehicle_states_)):
            assert self.vehicle_states_[time_index].time_stamp_ == judge_trajectory.vehicle_states_[time_index].time_stamp_

            # Judge whether collision
            is_collision = Rectangle.isCollision(self.vehicle_states_[time_index].rectangle_, judge_trajectory.vehicle_states_[time_index].rectangle_)
            if is_collision:
                safety_cost += 0.01 * abs(
                    self.vehicle_states_[time_index].velocity_ - judge_trajectory.vehicle_states_[time_index].velocity_) * 0.5

        return safety_cost

if __name__ == '__main__':

    # # Test rectangle and rectangle collision, visualization with plt
    # # Construct two rectangles and corresponding polygons
    # rectangle_1 = Rectangle(PathPoint(30.0, 20.0, 0.0), 5.0, 2.0)
    # rectangle_2 = Rectangle(PathPoint(25.0, 20.0, 0.8), 5.0, 2.0)
    # polygon_1 = Polygon(rectangle_1.vertex_)
    # polygon_2 = Polygon(rectangle_2.vertex_)
    #
    # # Judge collision
    # is_collison = Rectangle.isCollision(rectangle_1, rectangle_2)
    # print(is_collison)
    #
    # # Visualization
    # plt.figure(0, (12, 6))
    # plt.title('Test rectangle and collision')
    # plt.plot(*polygon_1.exterior.xy, c='r')
    # plt.plot(*polygon_2.exterior.xy, c='g')
    # plt.xlim(0.0, 100.0)
    # plt.ylim(0.0, 50.0)
    # plt.axis('equal')
    # plt.show()

    # Test lane and vehicle
    # Initialize lane
    center_lane_start_point = PathPoint(0.0, 0.0)
    center_lane_end_point = PathPoint(500.0, 0.0)
    center_lane = Lane(center_lane_start_point, center_lane_end_point, LaneId.CenterLane)
    center_lane_points_array = Visualization.transformPathPointsToArray(center_lane.path_points_)
    left_lane_start_point = PathPoint(0.0, 3.5)
    left_lane_end_point = PathPoint(500.0, 3.5)
    left_lane = Lane(left_lane_start_point, left_lane_end_point, LaneId.LeftLane)
    left_lane_points_array = Visualization.transformPathPointsToArray(left_lane.path_points_)
    right_lane_start_point = PathPoint(0.0, -3.5)
    right_lane_end_point = PathPoint(500.0, -3.5)
    right_lane = Lane(right_lane_start_point, right_lane_end_point, LaneId.RightLane)
    right_lane_points_array = Visualization.transformPathPointsToArray(right_lane.path_points_)

    # Initialize ego vehicle
    ego_vehicle = Vehicle(0, PathPoint(20.0, 0.0, 0.0), 5.0, 2.0, 5.0, 0.0, 0.0)
    ego_vehicle_polygon = Polygon(ego_vehicle.rectangle_.vertex_)

    # Generate surround agent vehicles
    # Set random seed
    random.seed(190000)
    agent_generator = AgentGenerator()
    surround_vehicle_set = agent_generator.generateAgents(0)

    # # Calculate vehicles' distance to corresponding lane
    # for sur_veh in surround_vehicle_set.values():
    #     dis_to_center_lane = center_lane.calculatePositionToLaneDistance(sur_veh.position_)
    #     dis_to_left_lane = left_lane.calculatePositionToLaneDistance(sur_veh.position_)
    #     dis_to_right_lane = right_lane.calculatePositionToLaneDistance(sur_veh.position_)
    #
    #     print('Vehicle id: {}, dis to center lane: {}, dis to left lane: {}, dis to right lane: {}'.format(sur_veh.id_, dis_to_center_lane, dis_to_left_lane, dis_to_right_lane))

    # # Calculate vehicles' nearest point index in corresponding lane
    # for sur_veh in surround_vehicle_set.values():
    #     nearest_point_index_in_center_lane = center_lane.calculateNearestIndexInLane(sur_veh.position_)
    #     nearest_point_index_in_left_lane = left_lane.calculateNearestIndexInLane(sur_veh.position_)
    #     nearest_point_index_in_right_lane = right_lane.calculateNearestIndexInLane(sur_veh.position_)
    #
    #     print('Vehicle id: {}, nearest point index in center lane: {}, nearest point index in left lane: {}, nearest point index in right lane: {}'.format(sur_veh.id_, nearest_point_index_in_center_lane, nearest_point_index_in_left_lane, nearest_point_index_in_right_lane))

    # # Calculate vehicle's nearest point in corresponding lane
    # for sur_veh in surround_vehicle_set.values():
    #     nearest_point_in_center_lane = center_lane.calculateNearestPointInLane(sur_veh.position_).toArray()
    #     nearest_point_in_left_lane = left_lane.calculateNearestPointInLane(sur_veh.position_).toArray()
    #     nearest_point_in_right_lane = right_lane.calculateNearestPointInLane(sur_veh.position_).toArray()
    #
    #     print('Vehicle id: {}, nearest point in center lane: {}, nearest point in left lane: {}, nearest point in right lane: {}'.format(sur_veh.id_, nearest_point_in_center_lane, nearest_point_in_left_lane, nearest_point_in_right_lane))

    # # Calculate vehicle's corresponding path point index from a position and distance
    # for sur_veh in surround_vehicle_set.values():
    #     target_distance = 50.0
    #     target_index_in_center_lane = center_lane.calculateTargetDistancePoint(sur_veh.position_, target_distance)
    #     target_index_in_left_lane = left_lane.calculateTargetDistancePoint(sur_veh.position_, target_distance)
    #     target_index_in_right_lane = right_lane.calculateTargetDistancePoint(sur_veh.position_, target_distance)
    #
    #     print('Vehicle id: {}, target index in center lane: {}, target index in left lane: {}, target index in right lane: {}'.format(sur_veh.id_, target_index_in_center_lane, target_index_in_left_lane, target_index_in_right_lane))


    # Test lane server and semantic vehicle
    # Construct lane server and semantic vehicles
    all_vehicle = [ego_vehicle] + list(surround_vehicle_set.values())
    lanes = {center_lane.id_: center_lane, left_lane.id_: left_lane, right_lane.id_: right_lane}
    lane_server = LaneServer()
    lane_server.refresh(lanes, all_vehicle)

    # # Check semantic vehicles' information in lane server
    # for seman_veh_id, seman_veh in lane_server.semantic_vehicles_.items():
    #     if seman_veh_id == 0:
    #         # For ego vehicle
    #         print('Vehicle id: {}, potential behavior: {}, nearest lane: {}'.format(seman_veh_id, seman_veh.potential_behaviors_, seman_veh.nearest_lane_.id_))
    #     else:
    #         # For surround vehicle
    #         print('Vehicle id: {}, potential behavior: {}, nearest lane: {}, reference lane: {}'.format(seman_veh_id, seman_veh.potential_behaviors_, seman_veh.nearest_lane_.id_, seman_veh.reference_lane_.id_))

    # Get ego vehicle's all potential behaviors and set ego vehicle's potential behavior randomly
    ego_vehicle_all_potential_behavior = lane_server.getEgoVehicleBehaviors()
    # lane_server.setEgoVehicleBehavior(LateralBehavior.LaneKeeping)

    # # Check vehicle's leading vehicle
    # for seman_veh_id, seman_veh in lane_server.semantic_vehicles_.items():
    #     leading_seman_veh = lane_server.getLeadingVehicle(seman_veh)
    #     if leading_seman_veh == None:
    #         # Without leading veh
    #         print('Vehicle id: {}, reference lane: {}, has no leading vehicle'.format(seman_veh_id, seman_veh.reference_lane_.id_))
    #     else:
    #         print('Vehicle id: {}, reference lane: {}, its leading vehicle id: {}'.format(seman_veh_id, seman_veh.reference_lane_.id_, leading_seman_veh.vehicle_.id_))

    # # Test IDM
    # # Set parameters
    # cur_s = 20.0
    # leading_s = 30.0
    # cur_velocity = 5.0
    # leading_velocity = 8.0
    #
    # # Calculate velocity and acceleration
    # idm_velocity = IDM.calculateVelocity(cur_s, leading_s, cur_velocity, leading_velocity, 0.4)
    # idm_acceleration = IDM.calculateAcceleration(cur_s, leading_s, cur_velocity, leading_velocity)
    #
    # print('IDM velocity: {}'.format(idm_velocity))
    # print('IDM acceleration: {}'.format(idm_acceleration))

    # # Test ideal steer model
    # # Load parameters for ideal steer model
    # ideal_steer_model = IdealSteerModel(Config.wheelbase_length, IDM.acceleration, IDM.hard_braking_deceleration, Config.max_lon_acc_jerk, Config.max_lon_brake_jerk, Config.max_lat_acceleration_abs, Config.max_lat_jerk_abs, Config.max_steer_angle_abs, Config.max_steer_rate, Config.max_curvature_abs)
    #
    # # Set current state and control information
    # ideal_steer_model.setState(ego_vehicle)
    # ideal_steer_model.setControl([0.05, 18.0])
    # ideal_steer_model.step(1.0)
    #
    # # Get first predicted state
    # predicted_state = ideal_steer_model.state_
    # predicted_state_polygon = Polygon(predicted_state.rectangle_.vertex_)
    #
    # # Get second predicted state
    # ideal_steer_model.setState(predicted_state)
    # ideal_steer_model.setControl([0.0, 10.0])
    # ideal_steer_model.step(1.0)
    # predicted_state_2 = ideal_steer_model.state_
    # predicted_state_polygon_2 = Polygon(predicted_state_2.rectangle_.vertex_)
    #
    # # Visualization ego vehicle and predicted state
    # plt.plot(*ego_vehicle_polygon.exterior.xy, c='r')
    # plt.text(ego_vehicle.position_.x_, ego_vehicle.position_.y_, 'id: {}, v: {}'.format(ego_vehicle.id_, ego_vehicle.velocity_), size=10.0)
    #
    # plt.plot(*predicted_state_polygon.exterior.xy, c='r', ls='--')
    # plt.text(predicted_state.position_.x_, predicted_state.position_.y_, 'id: {}, v: {}'.format(predicted_state.id_, predicted_state.velocity_), size=10.0)
    #
    # plt.plot(*predicted_state_polygon_2.exterior.xy, c='r', ls='--')
    # plt.text(predicted_state_2.position_.x_, predicted_state_2.position_.y_, 'id: {}, v: {}'.format(predicted_state_2.id_, predicted_state_2.velocity_), size=10.0)

    # # Test continuous ideal steer model
    # # Load parameters
    # ideal_steer_model = IdealSteerModel(Config.wheelbase_length, IDM.acceleration, IDM.hard_braking_deceleration, Config.max_lon_acc_jerk, Config.max_lon_brake_jerk, Config.max_lat_acceleration_abs, Config.max_lat_jerk_abs, Config.max_steer_angle_abs, Config.max_steer_rate, Config.max_curvature_abs)
    #
    # # Visualization ego vehicle
    # plt.plot(*ego_vehicle_polygon.exterior.xy, c='r')
    # plt.text(ego_vehicle.position_.x_, ego_vehicle.position_.y_, 'id: {}, v: {}'.format(ego_vehicle.id_, ego_vehicle.velocity_), size=10.0)
    #
    # # Initialize state cache
    # predicted_state = copy.deepcopy(ego_vehicle)
    #
    # # Predict 10 steps
    # for step_index in range(0, 10):
    #     ideal_steer_model.setState(predicted_state)
    #     ideal_steer_model.setControl([0.0, 10.0])
    #     ideal_steer_model.step(0.4)
    #     predicted_state = ideal_steer_model.state_
    #     predicted_state_polygon = Polygon(predicted_state.rectangle_.vertex_)
    #
    #     # Visualization predicted state
    #     plt.plot(*predicted_state_polygon.exterior.xy, c='r', ls='--')
    #     plt.text(predicted_state.position_.x_, predicted_state.position_.y_, 'id: {}, v: {}'.format(predicted_state.id_, predicted_state.velocity_), size=10.0)

    # Test forward extender
    # Construct forward extender
    forward_extender = ForwardExtender(lane_server, 0.4, 6.0)

    # Calculate ego trajectory and surround trajectory
    ego_trajectory, surround_trajectories = forward_extender.multiAgentForward(LateralBehavior.LaneChangeRight)


    # Visualization
    plt.figure(1, (12, 6))
    plt.title('Test lane, vehicle and semantic vehicle')

    # Visualization lane
    plt.plot(center_lane_points_array[:, 0], center_lane_points_array[:, 1], c='m', linewidth=1.0)
    plt.plot(center_lane.left_boundary_points_[:, 0], center_lane.left_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
    plt.plot(center_lane.right_boundary_points_[:, 0], center_lane.right_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
    plt.plot(left_lane_points_array[:, 0], left_lane_points_array[:, 1], c='m', linewidth=1.0)
    plt.plot(left_lane.left_boundary_points_[:, 0], left_lane.left_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
    plt.plot(left_lane.right_boundary_points_[:, 0], left_lane.right_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
    plt.plot(right_lane_points_array[:, 0], right_lane_points_array[:, 1], c='m', linewidth=1.0)
    plt.plot(right_lane.left_boundary_points_[:, 0], right_lane.left_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
    plt.plot(right_lane.right_boundary_points_[:, 0], right_lane.right_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)

    # # Visualization vehicle
    # plt.plot(*ego_vehicle_polygon.exterior.xy, c='r')
    # plt.text(ego_vehicle.position_.x_, ego_vehicle.position_.y_, 'id: {}, v: {}'.format(ego_vehicle.id_, ego_vehicle.velocity_), size=10.0)
    # for surround_vehicle in surround_vehicle_set.values():
    #     surround_vehicle_polygon = Polygon(surround_vehicle.rectangle_.vertex_)
    #     plt.plot(*surround_vehicle_polygon.exterior.xy, c='green')
    #     plt.text(surround_vehicle.position_.x_, surround_vehicle.position_.y_, 'id: {}, v: {}'.format(surround_vehicle.id_, surround_vehicle.velocity_), size=10.0)

    # Visualization vehicle and trajectories
    for i in range(0, len(ego_trajectory.vehicle_states_)):
        if i == 0:
            # For current position
            ego_vehicle_polygon = Polygon(ego_trajectory.vehicle_states_[i].rectangle_.vertex_)
            plt.plot(*ego_vehicle_polygon.exterior.xy, c='r')
            # plt.text(ego_vehicle.position_.x_, ego_vehicle.position_.y_, 'id: {}, v: {}'.format(ego_vehicle.id_, ego_vehicle.velocity_), size=10.0)
            # Traverse surround vehicle
            for sur_veh_id, sur_veh_tra in surround_trajectories.items():
                sur_vehicle_polygon = Polygon(sur_veh_tra.vehicle_states_[i].rectangle_.vertex_)
                plt.plot(*sur_vehicle_polygon.exterior.xy, c='green')
                # plt.text(sur_veh_tra.vehicle_states_[i].position_.x_, sur_veh_tra.vehicle_states_[i].position_.y_, 'id: {}, v: {}'.format(sur_veh_id, sur_veh_tra.vehicle_states_[i].velocity_), size=10.0)

        else:
            # For predicted position
            # For current position
            ego_vehicle_polygon = Polygon(ego_trajectory.vehicle_states_[i].rectangle_.vertex_)
            plt.plot(*ego_vehicle_polygon.exterior.xy, c='r', ls='--')
            # plt.text(ego_trajectory.vehicle_states_[i].position_.x_, ego_trajectory.vehicle_states_[i].position_.y_, 'id: {}, v: {}, time stamp: {}'.format(ego_vehicle.id_, ego_trajectory.vehicle_states_[i].velocity_, ego_trajectory.vehicle_states_[i].time_stamp_), size=10.0)
            # Traverse surround vehicle
            for sur_veh_id, sur_veh_tra in surround_trajectories.items():
                sur_vehicle_polygon = Polygon(sur_veh_tra.vehicle_states_[i].rectangle_.vertex_)
                plt.plot(*sur_vehicle_polygon.exterior.xy, c='green', ls='--')
                # plt.text(sur_veh_tra.vehicle_states_[i].position_.x_, sur_veh_tra.vehicle_states_[i].position_.y_, 'id: {}, v: {}, time stamp: {}'.format(sur_veh_id, sur_veh_tra.vehicle_states_[i].velocity_, sur_veh_tra.vehicle_states_[i].time_stamp_), size=10.0)

    plt.axis('equal')
    plt.show()






