# -- coding: utf-8 --
# @Time : 2021/10/25 下午9:46
# @Author : fujiawei0724
# @File : EUDMCommon.py
# @Software: PyCharm

"""
This code contains the components for EUDM behavior planner.
"""

import numpy as np
import copy
from enum import Enum, unique


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
        print()

# Construct available behavior sequence
class BehaviorSet:
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
            veh_beh_set.append(cur_behavior_sequence)

        return veh_beh_set

    # Add lane change situation which start from intermediate time stamp
    @classmethod
    def addBehavior(cls, cur_beh_seq, lon_beh, lat_beh, num):

        # Initialize
        res_beh_seq = copy.deepcopy(cur_beh_seq)

        # Add lane change behavior
        for i in range(0, num):
            res_beh_seq.append(VehicleBehavior(lat_beh, lon_beh))

        return res_beh_seq




if __name__ == '__main__':
    behavior_set = BehaviorSet(10)

    print('Behavior number: {}'.format(len(behavior_set.generateBehaviors())))
