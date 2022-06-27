# -- coding: utf-8 --
# @Time : 2021/11/13 上午9:45
# @Author : fujiawei0724
# @File : environment.py
# @Software: PyCharm

"""
The description of environment.
"""

import sys
sys.path.append('..')
import os
os.chdir(os.path.dirname(__file__))
import logging
import random
import h5py
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from rl_behavior_planner.utils import *


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
    def netDataAllToWorld(all_state_array):
        # Supply lane information and ego vehicle state
        lane_info_with_speed = all_state_array[:5]
        ego_vehicle = Vehicle(0, PathPoint(all_state_array[5], all_state_array[6], all_state_array[7]), all_state_array[8], all_state_array[9], all_state_array[10], all_state_array[11], None, all_state_array[12], all_state_array[13])

        # Supple surround vehicles states
        surround_vehicles = {}
        cur_veh_index = 1
        for i in range(14, 94, 8):
            # Judge the existence of the current vehicle
            if all_state_array[i] != 1:
                break
            
            # Create single surround vehicle
            sur_veh = Vehicle(cur_veh_index, PathPoint(all_state_array[i + 1], all_state_array[i + 2], all_state_array[i + 3]), all_state_array[i + 4], all_state_array[i + 5], all_state_array[i + 6], all_state_array[i + 7])
            surround_vehicles[cur_veh_index] = sur_veh
            cur_veh_index += 1
        
        return lane_info_with_speed, ego_vehicle, surround_vehicles



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

    action_index_1 = np.arange(0, 63, 1)
    action_index_2 = np.arange(0, 231, 1)

    with h5py.File('./data/action_info.h5', 'r') as f:
        # print(f.keys())
        action_info_1 = f['action_info_1'][()]
        action_info_2 = f['action_info_2'][()]

    @classmethod
    def indexToBehSeq(cls, index, with_print=False):
        if torch.is_tensor(index):
            index = index.item()
        assert index in cls.action_index_1

        beh_seq_info = cls.action_info_1[index]
        if with_print:
            print('Longitudinal behavior: {}'.format(LongitudinalBehavior(beh_seq_info[0])))
            for i in range(1, len(beh_seq_info)):
                print('Latitudinal behavior: {}'.format(LateralBehavior(beh_seq_info[i])))
        return beh_seq_info

    @classmethod
    def behSeqToIndex(cls, beh_seq_info):
        assert beh_seq_info in cls.action_info_1
        for index, cur_beh_seq_info in enumerate(cls.action_info_1):
            if (cur_beh_seq_info == beh_seq_info).all():
                return index

    @classmethod
    def indexToIntentionSeq(cls, index, with_print=False):
        if torch.is_tensor(index):
            index = index.item()
        assert index in cls.action_index_2

        intention_seq_info = cls.action_info_2[index]
        if with_print:
            print('Longitudinal velocity compensation: {}'.format(intention_seq_info[0]))
            for i in range(1, len(intention_seq_info)):
                print('Latitudinal behavior: {}'.format(LateralBehavior(intention_seq_info[i])))
        return intention_seq_info

    @classmethod
    def intentionSeqToIndex(cls, intention_seq_info):
        assert intention_seq_info in cls.action_info_2
        for index, cur_intention_seq_info in enumerate(cls.action_info_2):
            if (cur_intention_seq_info == intention_seq_info).all():
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

    def load(self, arr_info, ego_vehicle=None, surround_vehicles=None):
        # Check data
        self.lane_info_ = None
        self.lane_server_ = None
        self.ego_vehicle_ = None
        self.surround_vehicle_ = None

        if ego_vehicle == None and surround_vehicles == None:
            # Load information from array
            assert len(arr_info) == 94
            all_state_array = np.array(arr_info)

            # Load lanes data
            left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit = all_state_array[0], all_state_array[1], all_state_array[2], all_state_array[3], all_state_array[4]
            self.loadLaneInfo(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit)

            # Load vehicles data
            ego_vehicle_state_array = all_state_array[5:14]
            sur_vehicles_states_array = all_state_array[14:].reshape(10, 8)
            self.loadVehicleInfo(ego_vehicle_state_array, sur_vehicles_states_array)

        else:
            # Load information from vehicles information and lane information
            # Load lane info
            assert len(arr_info) == 5
            left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit = arr_info[0], arr_info[1], arr_info[2], arr_info[3], arr_info[4]
            self.loadLaneInfo(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit)

            # Load vehicle information
            self.ego_vehicle_ = ego_vehicle
            self.surround_vehicle_ = surround_vehicles

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
                self.surround_vehicle_[index + 1] = Vehicle(index + 1, 
                                                            PathPoint(single_sur_info[1], single_sur_info[2],
                                                            single_sur_info[3]),
                                                            single_sur_info[4], 
                                                            single_sur_info[5], 
                                                            single_sur_info[6],
                                                            single_sur_info[7], 0.0, 0.0, 0.0)

    # Load behavior sequence
    # behavior sequence is a array has 11 elements, [0] denotes the longitudinal behavior, [1:11] denotes the corresponding latitudinal behavior in each time stamps respectively
    def simulateBehSeq(self, behavior_sequence_info, with_visualization, ax, with_intention=True):
        # Record if done, if there is collision in trajectories or ego vehicle driving forward an nonexistent lane
        error_situation = False

        # ~Stage I: Transform the behavior sequence / intention sequence
        behavior_sequence = None
        intention_sequence = None
        is_final_lane_changed = None
        if with_intention:
            intention_seq = []
            for i in range(1, 11):
                intention_seq.append(VehicleIntention(LateralBehavior(behavior_sequence_info[i]), float(behavior_sequence_info[0])))
            intention_sequence = IntentionSequence(intention_seq)
            is_final_lane_changed = True if intention_sequence.intention_seq_[-1].lat_beh_ != LateralBehavior.LaneKeeping else False
            if is_final_lane_changed:
                if intention_sequence.intention_seq_[-1].lat_beh_ == LateralBehavior.LaneChangeLeft:
                    if LaneId.LeftLane not in self.lane_server_.lanes_:
                        error_situation = True
                elif intention_sequence.intention_seq_[-1].lat_beh_ == LateralBehavior.LaneChangeRight:
                    if LaneId.RightLane not in self.lane_server_.lanes_:
                        error_situation = True
        else:
            beh_seq = []
            for i in range(1, 11):
                beh_seq.append(VehicleBehavior(LateralBehavior(behavior_sequence_info[i]),
                                               LongitudinalBehavior(behavior_sequence_info[0])))
            behavior_sequence = BehaviorSequence(beh_seq)
            is_final_lane_changed = True if behavior_sequence.beh_seq_[-1].lat_beh_ != LateralBehavior.LaneKeeping else False
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
        ego_traj, surround_trajs = forward_extender.multiAgentForward(intention_sequence if with_intention else behavior_sequence, vehicles, self.lane_speed_limit_)

        # ~Stage IV: calculate cost and transform to reward
        """
        Three components of cost is converted from here to DEBUG.
        """
        policy_cost, is_collision, safety_cost, lane_change_cost, efficiency_cost = PolicyEvaluator.praise(ego_traj, surround_trajs, is_final_lane_changed, self.lane_speed_limit_)
        reward = 1.0 / policy_cost
        if is_collision:
            error_situation = True

        # ~Stage V: calculate next state
        # next_state = StateInterface.calculateNextState(self.lane_info_, ego_traj, surround_trajs)
        next_ego_vehicle_state = ego_traj.vehicle_states_[-1]
        next_sur_vehicles_states = {}
        for sur_veh_id, sur_veh_traj in surround_trajs.items():
            next_sur_vehicles_states[sur_veh_id] = sur_veh_traj.vehicle_states_[-1]
        
        # Keeping the position of the ego vehicle
        # Its abscissa should be 30.0 (or other values)
        gap = next_ego_vehicle_state.position_.x_ - 30.0
        for next_sur_veh_state in next_sur_vehicles_states.values():
            next_sur_veh_state.position_.x_ -= gap

        next_state = (next_ego_vehicle_state, next_sur_vehicles_states)

        # ~Stage VI: visualization
        if with_visualization:
            self.visualizationTrajs(ax, ego_traj, surround_trajs)

        if error_situation:
            return -1.0, next_state, True, safety_cost, lane_change_cost, efficiency_cost

        # ~Stage VII: judge whether done, current logic is: if ego vehicle forward distance excesses 60, done will be set with true, which means the end of a series of behavior sequences
        done = False
        if ego_traj.vehicle_states_[-1].position_.x_ > 90.0:
            done = True

        return reward, next_state, done, safety_cost, lane_change_cost, efficiency_cost

    # Run with a action index
    def runOnce(self, action, with_visualization=False, ax=None):
        beh_seq = ActionInterface.indexToIntentionSeq(action)
        reward, next_state, done, safety_cost, lane_change_cost, efficiency_cost = self.simulateBehSeq(beh_seq, with_visualization, ax, True)
        return reward, next_state, done, safety_cost, lane_change_cost, efficiency_cost

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
    test_with_file_data = False
    if not test_with_file_data:
        # Test data from random generation
        random.seed()
        # Load environment data randomly
        left_lane_exist = random.randint(0, 1)
        right_lane_exist = random.randint(0, 1)
        center_left_distance = random.uniform(3.0, 4.5)
        center_right_distance = random.uniform(3.0, 4.5)
        lane_limited_speed = random.uniform(10.0, 25.0)
        lane_info_with_speed = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_limited_speed]

        # Construct ego vehicle and surround vehicles randomly
        ego_vehicle = EgoInfoGenerator.generateOnce()
        surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
        surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))
        # Check initial situation
        if not Tools.checkInitSituation(ego_vehicle, surround_vehicles):
            assert False

        # Define action
        action = 1

        # Construct environment
        env = Environment()
        env.load(lane_info_with_speed, ego_vehicle, surround_vehicles)
        plt.figure(0)
        plt.title('Initial states')
        ax = plt.axes()
        env.visualization(ax)
        plt.axis('equal')
        # action_info = ActionInterface.indexToBehSeq(action, True)

        plt.figure(1)
        plt.title('All trajectories')
        ax_1 = plt.axes()
        cur_reward, next_state, cur_done, _, _, _ = env.runOnce(action, True, ax_1)
        plt.axis('equal')

        plt.figure(2)
        plt.title('Stored final states')
        ax_2 = plt.axes()

        env.load(lane_info_with_speed, next_state[0], next_state[1])
        env.visualization(ax_2)
        plt.axis('equal')

        print('Reward: {}'.format(cur_reward))
        plt.show()

    else:
        # Test data from file
        # Read data
        with h5py.File('./data/data.h5', 'r') as f:
            print(f.keys())
            actions = f['actions'][()]
            current_states = f['current_states'][()]
            dones = f['dones'][()]
            next_states = f['next_states'][()]
            rewards = f['rewards'][()]

        # print(np.array(actions).shape)
        # print(np.array(current_states).shape)
        # print(np.array(dones).shape)
        # print(np.array(next_states).shape)
        # print(np.array(rewards).shape)

        # Initialize directory to store information
        if not os.path.exists('./figure/'):
            os.makedirs('./figure/')
        if not os.path.exists('./log/'):
            os.makedirs('./log/')
        log_format = '%(levelname)s %(asctime)s - %(message)s'
        logging.basicConfig(filename='./log/reword_info.log',
                            filemode='a',
                            format=log_format,
                            level=logging.INFO)
        logger = logging.getLogger()

        for i in range(0, len(actions)):
            # Initialize figure subdirectory
            if not os.path.exists('./figure/figure_{}'.format(i)):
                os.makedirs('./figure/figure_{}'.format(i))
            # Define data index and select data
            test_data_index = i
            print('Test data index: {}'.format(test_data_index))
            action = actions[test_data_index]
            current_state_array = current_states[test_data_index]
            done = dones[test_data_index]
            next_state_array = next_states[test_data_index]
            reward = rewards[test_data_index]

            env = Environment()
            # Visualization initial state
            env.load(current_state_array)
            plt.figure(0)
            plt.title('Initial states')
            ax = plt.axes()
            env.visualization(ax)
            plt.axis('equal')
            plt.savefig('./figure/figure_{}/initial_states.jpg'.format(i))
            # Print behavior sequence
            action_info = ActionInterface.indexToBehSeq(action)
            # Visualization all trajectories
            plt.figure(1)
            plt.title('All trajectories')
            ax_1 = plt.axes()
            _, _, _, safety_cost, lane_change_cost, efficiency_cost = env.runOnce(action, True, ax_1)
            plt.axis('equal')
            plt.savefig('./figure/figure_{}/all_trajectories.jpg'.format(i))
            plt.figure(2)
            plt.title('Stored final states')
            ax_2 = plt.axes()
            env.load(next_state_array)
            env.visualization(ax_2)
            plt.axis('equal')
            plt.savefig('./figure/figure_{}/stored_final_states.jpg'.format(i))

            logger.info('-----------------------Epoch: {}------------------------'.format(test_data_index))
            logger.info('Safety cost: {}, lane change cost: {}, efficiency cost: {}'.format(safety_cost, lane_change_cost, efficiency_cost))
            logger.info('Reward: {}'.format(reward))
            logger.info('Longitudinal behavior: {}'.format(LongitudinalBehavior(action_info[0])))
            for i in range(1, 11):
                logger.info('Latitudinal behavior: {}'.format(LateralBehavior(action_info[i])))
            # plt.show()






