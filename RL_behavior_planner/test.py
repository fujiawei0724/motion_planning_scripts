# -- coding: utf-8 --
# @Time : 2021/11/16 下午4:04
# @Author : fujiawei0724
# @File : test.py
# @Software: PyCharm

"""
Load the trained network to generate behavior sequence and compare with branch calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from Double_DQN_net import DQN
from environment import Environment, StateInterface, ActionInterface
from utils import *

class DDQNTester:

    @staticmethod
    def test(test_eposide=100):
        # Load network
        state_size = 93
        action_size = 63
        policy_net = DQN(93, 63)
        policy_net.load_state_dict(torch.load('../weights/checkpoint0.pt', map_location='cpu'))
        policy_net.eval()

        # Initialize container
        best_actions = []
        best_rewards = []
        output_actions = []
        output_rewards = []

        for epoch in range(0, test_eposide):
            # Load environment data randomly
            env = Environment()
            # TODO: delete error situations, such as collision
            left_lane_exist = random.randint(0, 1)
            right_lane_exist = random.randint(0, 1)
            center_left_distance = random.uniform(3.0, 4.5)
            center_right_distance = random.uniform(3.0, 4.5)
            # Construct ego vehicle and surround vehicles randomly
            ego_vehicle = EgoInfoGenerator.generateOnce()
            surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
            surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))
            # Transform to state array
            current_state_array = StateInterface.worldToNetDataAll([left_lane_exist, right_lane_exist, center_left_distance, center_right_distance], ego_vehicle, surround_vehicles)
            env.load(current_state_array)

            # Calculate action predicted by network and its reward
            net_pred_action = policy_net.forward(current_state_array)
            net_pred_action_reward, _ = env.runOnce(net_pred_action)

            # Traverse calculate the best action with the largest reward
            real_rewards = np.array([0.0 for _ in range(63)])
            for action in range(0, 63):
                real_rewards[action], _ = env.runOnce(action)
            # Get best action and reward
            best_action = np.argmax(real_rewards)
            best_reward = real_rewards[best_action]

            output_actions.append(net_pred_action)
            output_rewards.append(net_pred_action_reward)
            best_actions.append(best_action)
            best_rewards.append(best_reward)

        print('Output actions: {}'.format(output_actions))
        print('Output rewards: {}'.format(output_rewards))
        print('Best actions: {}'.format(best_actions))
        print('Best rewards: {}'.format(best_rewards))

        



