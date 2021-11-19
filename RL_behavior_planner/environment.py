# -- coding: utf-8 --
# @Time : 2021/11/13 上午9:45
# @Author : fujiawei0724
# @File : environment.py
# @Software: PyCharm

"""
The description of environment.
"""

import random
import h5py
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from utils import *


# Transform the state between world and neural network data
class StateInterface:
    @staticmethod
    def worldToNetData(ego_veh, surround_veh):
        state_array = np.zeros((89,))
        # Supple ego vehicle information
        state_array[0], state_array[1], state_array[2], state_array[3], state_array[4], state_array[5], state_array[6], \
        state_array[7], state_array[
            8] = ego_veh.position_.x_, ego_veh.position_.y_, ego_veh.position_.theta_, ego_veh.length_, ego_veh.width_, ego_veh.velocity_, ego_veh.acceleration_, ego_veh.curvature_, ego_veh.steer_

        # Supple surround vehicles information
        for i in range(1, len(surround_veh) + 1):
            start_index = 9 + (i - 1) * 8
            cur_veh_state = surround_veh[i]
            state_array[start_index], state_array[start_index + 1], state_array[start_index + 2], state_array[
                start_index + 3], state_array[start_index + 4], state_array[start_index + 5], state_array[
                start_index + 6], state_array[
                start_index + 7] = 1, cur_veh_state.position_.x_, cur_veh_state.position_.y_, cur_veh_state.position_.theta_, cur_veh_state.length_, cur_veh_state.width_, cur_veh_state.velocity_, cur_veh_state.acceleration_

        return state_array

    @staticmethod
    def worldToNetDataAll(lane_info, ego_veh, surround_veh):
        all_state_array = np.zeros((94,))
        all_state_array[:5] = lane_info[:]
        vehicles_state_array = StateInterface.worldToNetData(ego_veh, surround_veh)
        all_state_array[5:] = vehicles_state_array[:]
        return all_state_array

    @staticmethod
    def netDataToWorld():
        pass

    # Calculate next state
    @staticmethod
    def calculateNextState(lane_info, ego_traj: Trajectory, surround_trajs):
        next_state = np.zeros((94,))
        # Supple lane information
        next_state[:5] = copy.deepcopy(lane_info)

        # Calculate ege vehicle state and surround vehicles states
        ego_veh_state = ego_traj.vehicle_states_[-1]
        sur_veh_states = {}
        for sur_veh_id, sur_veh_traj in surround_trajs.items():
            sur_veh_states[sur_veh_id] = sur_veh_traj.vehicle_states_[-1]

        # Calculate states array
        state_array = StateInterface.worldToNetData(ego_veh_state, sur_veh_states)

        next_state[5:] = state_array[:]

        return next_state


# Transform action between index and behavior sequence
class ActionInterface:
    action_index = np.arange(0, 63, 1)
    action_info = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
                            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
                            [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2],
                            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
                            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                            [2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2],
                            [2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                            [2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
                            [2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                            [2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                            [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                            [2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                            [2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                            [2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                            [2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    @classmethod
    def indexToBehSeq(cls, index, with_print=False):
        if torch.is_tensor(index):
            index = index.item()
        assert index in cls.action_index

        beh_seq_info = cls.action_info[index]
        if with_print:
            print('Longitudinal behavior: {}'.format(LongitudinalBehavior(beh_seq_info[0])))
            for i in range(1, len(beh_seq_info)):
                print('Latitudinal behavior: {}'.format(LateralBehavior(beh_seq_info[i])))
        return beh_seq_info

    @classmethod
    def behSeqToIndex(cls, beh_seq_info):
        assert beh_seq_info in cls.action_info
        for index, cur_beh_seq_info in enumerate(cls.action_info):
            if (cur_beh_seq_info == beh_seq_info).all():
                return index


# Construct the environment for reward calculation
# Environment includes the lane information and vehicle information (ego and surround)
class Environment:

    def __init__(self):
        # Initialize shell
        self.lane_info_ = None
        self.lane_server_ = None
        self.ego_vehicle_ = None
        self.surround_vehicle_ = None

    def load(self, all_state_array):
        # Check data
        self.lane_info_ = None
        self.lane_server_ = None
        self.ego_vehicle_ = None
        self.surround_vehicle_ = None
        assert len(all_state_array) == 94
        all_state_array = np.array(all_state_array)

        # Load lanes data
        left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit = all_state_array[0], all_state_array[1], all_state_array[2], all_state_array[3], all_state_array[4]
        self.loadLaneInfo(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit)

        # Load vehicles data
        ego_vehicle_state_array = all_state_array[5:14]
        sur_vehicles_states_array = all_state_array[14:].reshape(10, 8)
        self.loadVehicleInfo(ego_vehicle_state_array, sur_vehicles_states_array)

    def loadLaneInfo(self, left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit):
        # Store next state for next state calculation
        self.lane_info_ = np.array([left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit])

        center_lane = None
        left_lane = None
        right_lane = None

        # Initialize lane with the assumption that the lane has 500m to drive at least
        center_lane_start_point = PathPoint(0.0, 0.0)
        center_lane_end_point = PathPoint(500.0, 0.0)
        center_lane = Lane(center_lane_start_point, center_lane_end_point, LaneId.CenterLane)
        # center_lane_points_array = Visualization.transformPathPointsToArray(center_lane.path_points_)
        if left_lane_exist:
            left_lane_start_point = PathPoint(0.0, center_left_distance)
            left_lane_end_point = PathPoint(500.0, center_left_distance)
            left_lane = Lane(left_lane_start_point, left_lane_end_point, LaneId.LeftLane)
            # left_lane_points_array = Visualization.transformPathPointsToArray(left_lane.path_points_)
        if right_lane_exist:
            right_lane_start_point = PathPoint(0.0, -center_right_distance)
            right_lane_end_point = PathPoint(500.0, -center_right_distance)
            right_lane = Lane(right_lane_start_point, right_lane_end_point, LaneId.RightLane)
            # right_lane_points_array = Visualization.transformPathPointsToArray(right_lane.path_points_)

        # Construct lane server
        lanes = dict()
        lanes[center_lane.id_] = center_lane
        if left_lane_exist:
            lanes[left_lane.id_] = left_lane
        if right_lane_exist:
            lanes[right_lane.id_] = right_lane
        self.lane_server_ = LaneServer(lanes)
        self.lane_speed_limit_ = lane_speed_limit

    # Load vehicles information
    # The max number of vehicles considered is 10, if the real number is lower than this, all the data will be supple with 0
    def loadVehicleInfo(self, ego_info, sur_info):
        # Refresh
        self.ego_vehicle_ = None
        self.surround_vehicle_ = None

        # Load ego vehicle, ego vehicle's state could be represented by 9 values
        self.ego_vehicle_ = Vehicle(0, PathPoint(ego_info[0], ego_info[1], ego_info[2]), ego_info[3], ego_info[4],
                                    ego_info[5], ego_info[6], 0.0, ego_info[7], ego_info[8])

        # Load surround vehicles, for each surround vehicle, its state could by denoted by 8 values, compared with ego vehicle, a flag is added to denote whether this surround vehicle is exist, then the curvature and steer information are deleted because of the limits of perception
        self.surround_vehicle_ = dict()
        for index, single_sur_info in enumerate(sur_info):
            if single_sur_info[0] == 1:
                self.surround_vehicle_[index + 1] = Vehicle(index + 1, PathPoint(single_sur_info[1], single_sur_info[2],
                                                                                 single_sur_info[3]),
                                                            single_sur_info[4], single_sur_info[5], single_sur_info[6],
                                                            single_sur_info[7], 0.0, 0.0, 0.0)

    # Load behavior sequence
    # behavior sequence is a array has 11 elements, [0] denotes the longitudinal behavior, [1:11] denotes the corresponding latitudinal behavior in each time stamps respectively
    def simulateBehSeq(self, behavior_sequence_info, with_visualization, ax):
        # Record if done, if there is collision in trajectories or ego vehicle driving forward an nonexistent lane
        error_situation = False

        # ~Stage I: Transform the behavior sequence
        beh_seq = []
        for i in range(1, 11):
            beh_seq.append(VehicleBehavior(LateralBehavior(behavior_sequence_info[i]),
                                           LongitudinalBehavior(behavior_sequence_info[0])))
        behavior_sequence = BehaviorSequence(beh_seq)
        is_final_lane_changed = True if behavior_sequence.beh_seq_[
                                            -1].lat_beh_ != LateralBehavior.LaneKeeping else False
        if is_final_lane_changed:
            if behavior_sequence.beh_seq_[-1].lat_beh_ == LateralBehavior.LaneChangeLeft:
                if LaneId.LeftLane not in self.lane_server_.lanes_:
                    error_situation = True
            elif behavior_sequence.beh_seq_[-1].lat_beh_ == LateralBehavior.LaneChangeRight:
                if LaneId.RightLane not in self.lane_server_.lanes_:
                    error_situation = True

        # ~Stage II: Construct all vehicles
        vehicles = copy.deepcopy(self.surround_vehicle_)
        vehicles[0] = copy.deepcopy(self.ego_vehicle_)

        # ~Stage III: Construct forward extender and predict result trajectories for all vehicles (ego and surround)
        forward_extender = ForwardExtender(self.lane_server_, 0.4, 4.0)
        ego_traj, surround_trajs = forward_extender.multiAgentForward(behavior_sequence, vehicles, self.lane_speed_limit_)

        # ~Stage IV: calculate cost and transform to reward
        policy_cost, is_collision = PolicyEvaluator.praise(ego_traj, surround_trajs, is_final_lane_changed, self.lane_speed_limit_)
        reward = 1.0 / policy_cost
        if is_collision:
            error_situation = True

        # ~Stage V: calculate next state
        next_state = StateInterface.calculateNextState(self.lane_info_, ego_traj, surround_trajs)

        # ~Stage VI: visualization
        if with_visualization:
            self.visualizationTrajs(ax, ego_traj, surround_trajs)

        if error_situation:
            return -100.0, next_state, True

        return reward, next_state, True

    # Run with a action index
    def runOnce(self, action, with_visualization=False, ax=None):
        beh_seq = ActionInterface.indexToBehSeq(action)
        reward, next_state, done = self.simulateBehSeq(beh_seq, with_visualization, ax)
        return reward, next_state, done

    # DEBUG: visualization lanes
    def visualizationLanes(self, ax):
        # Visualization lanes
        if LaneId.CenterLane in self.lane_server_.lanes_:
            center_lane = self.lane_server_.lanes_[LaneId.CenterLane]
            center_lane_points_array = Visualization.transformPathPointsToArray(center_lane.path_points_)
            ax.plot(center_lane_points_array[:, 0], center_lane_points_array[:, 1], c='m', linewidth=1.0)
            ax.plot(center_lane.left_boundary_points_[:, 0], center_lane.left_boundary_points_[:, 1], c='black',
                    ls='--', linewidth=1.0)
            ax.plot(center_lane.right_boundary_points_[:, 0], center_lane.right_boundary_points_[:, 1], c='black',
                    ls='--', linewidth=1.0)
        if LaneId.LeftLane in self.lane_server_.lanes_:
            left_lane = self.lane_server_.lanes_[LaneId.LeftLane]
            left_lane_points_array = Visualization.transformPathPointsToArray(left_lane.path_points_)
            ax.plot(left_lane_points_array[:, 0], left_lane_points_array[:, 1], c='m', linewidth=1.0)
            ax.plot(left_lane.left_boundary_points_[:, 0], left_lane.left_boundary_points_[:, 1], c='black', ls='--',
                    linewidth=1.0)
            ax.plot(left_lane.right_boundary_points_[:, 0], left_lane.right_boundary_points_[:, 1], c='black', ls='--',
                    linewidth=1.0)
        if LaneId.RightLane in self.lane_server_.lanes_:
            right_lane = self.lane_server_.lanes_[LaneId.RightLane]
            right_lane_points_array = Visualization.transformPathPointsToArray(right_lane.path_points_)
            ax.plot(right_lane_points_array[:, 0], right_lane_points_array[:, 1], c='m', linewidth=1.0)
            ax.plot(right_lane.left_boundary_points_[:, 0], right_lane.left_boundary_points_[:, 1], c='black', ls='--',
                    linewidth=1.0)
            ax.plot(right_lane.right_boundary_points_[:, 0], right_lane.right_boundary_points_[:, 1], c='black',
                    ls='--', linewidth=1.0)

    # DEBUG: visualization states
    def visualization(self, ax):
        # Visualization lanes
        self.visualizationLanes(ax)

        # Visualization vehicles
        ego_vehilce = copy.deepcopy(self.ego_vehicle_)
        surround_vehicles = copy.deepcopy(self.surround_vehicle_)
        ego_vehicle_polygon = Polygon(ego_vehilce.rectangle_.vertex_)
        ax.plot(*ego_vehicle_polygon.exterior.xy, c='r')
        for _, sur_veh in surround_vehicles.items():
            cur_sur_vehicle_polygon = Polygon(sur_veh.rectangle_.vertex_)
            ax.plot(*cur_sur_vehicle_polygon.exterior.xy, c='g')

    # DEBUG: visualization all trajectories
    def visualizationTrajs(self, ax, ego_traj, sur_trajs):
        # Visualization lanes
        self.visualizationLanes(ax)

        # Visualization trajectories
        traj_length = len(ego_traj.vehicle_states_)
        for i in range(0, traj_length):
            if i == 0:
                # For current position
                ego_vehicle_polygon = Polygon(ego_traj.vehicle_states_[i].rectangle_.vertex_)
                ax.plot(*ego_vehicle_polygon.exterior.xy, c='r')
                # ax.text(ego_vehicle.position_.x_, ego_vehicle.position_.y_, 'id: {}, v: {}'.format(ego_vehicle.id_, ego_vehicle.velocity_), size=10.0)
                # Traverse surround vehicle
                for sur_veh_id, sur_veh_tra in sur_trajs.items():
                    sur_vehicle_polygon = Polygon(sur_veh_tra.vehicle_states_[i].rectangle_.vertex_)
                    ax.plot(*sur_vehicle_polygon.exterior.xy, c='green')
                    # ax.text(sur_veh_tra.vehicle_states_[i].position_.x_, sur_veh_tra.vehicle_states_[i].position_.y_, 'id: {}, v: {}'.format(sur_veh_id, sur_veh_tra.vehicle_states_[i].velocity_), size=10.0)

            else:
                # For predicted position
                # For current position
                ego_vehicle_polygon = Polygon(ego_traj.vehicle_states_[i].rectangle_.vertex_)
                ax.plot(*ego_vehicle_polygon.exterior.xy, c='r', ls='--')
                # ax.text(lane_keeping_ego_trajectory.vehicle_states_[i].position_.x_, lane_keeping_ego_trajectory.vehicle_states_[i].position_.y_, 'id: {}, v: {}, time stamp: {}'.format(ego_vehicle.id_, lane_keeping_ego_trajectory.vehicle_states_[i].velocity_, lane_keeping_ego_trajectory.vehicle_states_[i].time_stamp_), size=10.0)
                # Traverse surround vehicle
                for sur_veh_id, sur_veh_tra in sur_trajs.items():
                    sur_vehicle_polygon = Polygon(sur_veh_tra.vehicle_states_[i].rectangle_.vertex_)
                    ax.plot(*sur_vehicle_polygon.exterior.xy, c='green', ls='--')
                    # ax.text(sur_veh_tra.vehicle_states_[i].position_.x_, sur_veh_tra.vehicle_states_[i].position_.y_, 'id: {}, v: {}, time stamp: {}'.format(sur_veh_id, sur_veh_tra.vehicle_states_[i].velocity_, sur_veh_tra.vehicle_states_[i].time_stamp_), size=10.0)


if __name__ == '__main__':
    random.seed(1564255424361266166)
    # Load environment data randomly
    left_lane_exist = random.randint(0, 1)
    right_lane_exist = random.randint(0, 1)
    center_left_distance = random.uniform(3.0, 4.5)
    center_right_distance = random.uniform(3.0, 4.5)
    lane_limited_speed = random.uniform(10.0, 25.0)
    # Construct ego vehicle and surround vehicles randomly
    ego_vehicle = EgoInfoGenerator.generateOnce()
    surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
    surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))
    # Check initial situation
    if not Tools.checkInitSituation(ego_vehicle, surround_vehicles):
        assert False

    # Transform to state array
    current_state_array = StateInterface.worldToNetDataAll(
        [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_limited_speed], ego_vehicle,
        surround_vehicles)
    # Define action
    action = 35

    # Construct environment
    env = Environment()
    env.load(current_state_array)
    plt.figure(0)
    plt.title('Initial states')
    ax = plt.axes()
    env.visualization(ax)
    plt.axis('equal')
    action_info = ActionInterface.indexToBehSeq(action, True)

    plt.figure(1)
    plt.title('All trajectories')
    ax_1 = plt.axes()
    cur_reward, cur_next_state_array, cur_done = env.runOnce(action, True, ax_1)
    plt.axis('equal')

    plt.figure(2)
    plt.title('Stored final states')
    ax_2 = plt.axes()

    env.load(cur_next_state_array)
    env.visualization(ax_2)
    plt.axis('equal')

    print('Reward: {}'.format(cur_reward))

    plt.show()
