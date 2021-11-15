# -- coding: utf-8 --
# @Time : 2021/11/13 上午9:45
# @Author : fujiawei0724
# @File : environment.py
# @Software: PyCharm

"""
The description of environment.
"""

import torch
import copy
import numpy as np

from utils import *

# Transform the state between world and neural network data
class StateInterface:
    @staticmethod
    def worldToNetData():
        pass

    @staticmethod
    def netDataToWorld():
        pass

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
    def indexToBehSeq(cls, index):
        assert index in cls.action_index
        return cls.action_info[index]

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
        pass

    def loadLaneInfo(self, left_lane_exist, right_lane_exist, center_left_distance, center_right_distance):
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

    # Load vehicles information
    # The max number of vehicles considered is 10, if the real number is lower than this, all the data will be supple with 0
    def loadVehicleInfo(self, ego_info, sur_info):
        # Refresh
        self.ego_vehicle_ = None
        self.surround_vehicle_ = None

        # Load ego vehicle, ego vehicle's state could be represented by 9 values
        self.ego_vehicle_ = Vehicle(0, PathPoint(ego_info[0], ego_info[1], ego_info[2]), ego_info[3], ego_info[4], ego_info[5], ego_info[6], 0.0, ego_info[7], ego_info[8])

        # Load surround vehicles, for each surround vehicle, its state could by denoted by 8 values, compared with ego vehicle, a flag is added to denote whether this surround vehicle is exist, then the curvature and steer information are deleted because of the limits of perception
        self.surround_vehicle_ = dict()
        for index, single_sur_info in enumerate(sur_info):
            if single_sur_info[0] == 1:
                self.surround_vehicle_[index + 1] = Vehicle(index + 1, PathPoint(single_sur_info[1], single_sur_info[2], single_sur_info[3]), single_sur_info[4], single_sur_info[5], single_sur_info[6], single_sur_info[7], 0.0, 0.0, 0.0)


    # Load behavior sequence
    # behavior sequence is a array has 11 elements, [0] denotes the longitudinal behavior, [1:11] denotes the corresponding latitudinal behavior in each time stamps respectively
    def simulateBehSeq(self, behavior_sequence_info):

        # ~Stage I: Transform the behavior sequence
        beh_seq = []
        for i in range(1, 11):
            beh_seq.append(VehicleBehavior(LateralBehavior(behavior_sequence_info[i]), LongitudinalBehavior(behavior_sequence_info[0])))
        behavior_sequence = BehaviorSequence(beh_seq)

        # ~Stage II: Construct all vehicles
        vehicles = copy.deepcopy(self.surround_vehicle_)
        vehicles[0] = copy.deepcopy(self.ego_vehicle_)

        # ~Stage III: Construct forward extender and predict result trajectories for all vehicles (ego and surround)
        forward_extender = ForwardExtender(self.lane_server_, 0.4, 4.0)
        ego_traj, surround_trajs = forward_extender.multiAgentForward(behavior_sequence, vehicles)
        is_final_lane_changed = True if behavior_sequence.beh_seq_[-1].lat_beh_ != LateralBehavior.LaneKeeping else False

        # ~Stage IV: calculate cost and transform to reward
        policy_cost = PolicyEvaluator.calculateCost(ego_traj, surround_trajs, is_final_lane_changed)
        reward = 1.0 / policy_cost

        return reward



if __name__ == '__main__':
    pass




