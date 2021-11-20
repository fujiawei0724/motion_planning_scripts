# -- coding: utf-8 --
# @Time : 2021/11/16 下午4:04
# @Author : fujiawei0724
# @File : test.py
# @Software: PyCharm

"""
Load the trained network to generate behavior sequence and compare with branch calculation.
"""

import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from scipy.integrate import odeint
from Double_DQN_net import DQN
from DDPGTrainer import Actor, Critic
from environment import Environment, StateInterface, ActionInterface
from utils import *

class Tester:

    @staticmethod
    def testDDQN(test_eposide=100):
        # Load network
        state_size = 94
        action_size = 63
        policy_net = DQN(state_size, action_size)
        policy_net.load_state_dict(torch.load('./DDQN_weights/checkpoint0.pt', map_location='cpu'))
        policy_net.eval()

        # Initialize container
        best_actions = []
        best_rewards = []
        output_actions = []
        output_rewards = []

        epoch = 0
        while epoch < test_eposide:
            # Load environment data randomly
            env = Environment()
            left_lane_exist = random.randint(0, 1)
            right_lane_exist = random.randint(0, 1)
            center_left_distance = random.uniform(3.0, 4.5)
            center_right_distance = random.uniform(3.0, 4.5)
            lane_speed_limit = random.uniform(10.0, 25.0)
            # Construct ego vehicle and surround vehicles randomly
            ego_vehicle = EgoInfoGenerator.generateOnce()
            surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
            surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))
            # Judge whether available
            if not Tools.checkInitSituation(ego_vehicle, surround_vehicles):
                print('Initial situation error, reset vehicles information!!!')
                continue

            # Transform to state array
            current_state_array = StateInterface.worldToNetDataAll([left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit], ego_vehicle, surround_vehicles)
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

            epoch += 1

        print('Output actions: {}'.format(output_actions))
        print('Output rewards: {}'.format(output_rewards))
        print('Best actions: {}'.format(best_actions))
        print('Best rewards: {}'.format(best_rewards))

    # Test DDPG
    @staticmethod
    def testDDPG(test_episode=100):
        state_size = 94
        action_dim = 63
        high_action_scalar = 1.0
        low_action_scalar = -1.0

        # Load actor and critic
        actor = Actor(state_size, action_dim, high_action_scalar)
        critic = Critic(state_size, action_dim)
        actor.load_state_dict(torch.load('./DDPG_weights/actor_checkpoint.pt', map_location='cpu'))
        critic.load_state_dict(torch.load('./DDPG_weights/critic_checkpoint.pt', map_location='cpu'))
        actor.eval()
        critic.eval()

        # Initialize container
        best_actions = []
        best_rewards = []
        output_actions = []
        output_rewards = []

        epoch = 0
        while epoch < test_episode:
            # Load environment data randomly
            env = Environment()
            left_lane_exist = random.randint(0, 1)
            right_lane_exist = random.randint(0, 1)
            center_left_distance = random.uniform(3.0, 4.5)
            center_right_distance = random.uniform(3.0, 4.5)
            lane_speed_limit = random.uniform(10.0, 25.0)
            # Construct ego vehicle and surround vehicles randomly
            ego_vehicle = EgoInfoGenerator.generateOnce()
            surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
            surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))
            # Judge whether available
            if not Tools.checkInitSituation(ego_vehicle, surround_vehicles):
                print('Initial situation error, reset vehicles information!!!')
                continue

            # Transform to state array
            current_state_array = StateInterface.worldToNetDataAll(
                [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit], ego_vehicle, surround_vehicles)
            env.load(current_state_array)

            # Calculate action predicted by network and its reward
            net_pred_action = actor.forward(current_state_array)
            net_pred_action = net_pred_action.squeeze(0).cpu().numpy()
            net_pred_action = np.argmax(net_pred_action)
            net_pred_action_reward, _, _ = env.runOnce(net_pred_action)

            # Traverse calculate the best action with the largest reward
            real_rewards = np.array([0.0 for _ in range(63)])
            for action in range(0, 63):
                real_rewards[action], _, _ = env.runOnce(action)
            # Get best action and reward
            best_action = np.argmax(real_rewards)
            best_reward = real_rewards[best_action]

            output_actions.append(net_pred_action)
            output_rewards.append(net_pred_action_reward)
            best_actions.append(best_action)
            best_rewards.append(best_reward)

            epoch += 1

        print('Output actions: {}'.format(output_actions))
        print('Output rewards: {}'.format(output_rewards))
        print('Best actions: {}'.format(best_actions))
        print('Best rewards: {}'.format(best_rewards))



if __name__ == '__main__':
    pass


