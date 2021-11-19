# -- coding: utf-8 --
# @Time : 2021/11/19 上午9:33
# @Author : fujiawei0724
# @File : dataCollector.py
# @Software: PyCharm

"""
Collect the data for training.
Note that these data are with the premise that after executing a behavior sequence, the playing process is finished. These data are more suitable for DDPG probably?
"""

import os
import random

import h5py
from collections import namedtuple
from environment import Environment, StateInterface
from utils import *


# Data in memory buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class DataCollector:
    @staticmethod
    def runOnce(data_size=1000, lane_info_reset_num=100000, vehicle_info_reset_num=100):
        current_states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        # Start iteration
        episode = 0
        for environment_reset_episode in range(0, lane_info_reset_num):
            # Construct training environment
            left_lane_exist = random.randint(0, 1)
            right_lane_exist = random.randint(0, 1)
            center_left_distance = random.uniform(3.0, 4.5)
            center_right_distance = random.uniform(3.0, 4.5)
            lane_speed_limit = random.uniform(10.0, 25.0)
            env = Environment()

            # Vehicles information reset iteration
            for vehicles_reset_episode in range(0, vehicle_info_reset_num):
                # Construct ego vehicle and surround vehicles randomly
                ego_vehicle = EgoInfoGenerator.generateOnce()
                surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
                surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))

                # Judge whether available
                if not Tools.checkInitSituation(ego_vehicle, surround_vehicles):
                    print('Initial situation error, reset vehicles information!!!')
                    break

                # Transform to state array
                current_state_array = StateInterface.worldToNetDataAll([left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit], ego_vehicle, surround_vehicles)

                # Load all information to env
                env.load(current_state_array)

                # Traverse action
                for action in range(0, 63):
                    episode += 1
                    print('Start episode: {}'.format(episode))
                    # Execute selected action
                    reward, next_state_array, done = env.runOnce(action)
                    # Store data
                    current_states.append(current_state_array)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state_array)
                    dones.append(done)

                    if episode >= data_size:
                        break

                if episode >= data_size:
                    break

            if episode >= data_size:
                break

        # Transform data
        current_states = np.array(current_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        # print('Current states: {}'.format(current_states))
        # print('Actions: {}'.format(actions))
        # print('Rewards: {}'.format(rewards))
        # print('Next states: {}'.format(next_states))
        # print('Dones: {}'.format(dones))

        # Initialize storage file
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        f = h5py.File('./data/data.h5', 'w')
        f['current_states'] = current_states
        f['actions'] = actions
        f['rewards'] = rewards
        f['next_states'] = next_states
        f['dones'] = dones

if __name__ == '__main__':
    DataCollector.runOnce(2)




