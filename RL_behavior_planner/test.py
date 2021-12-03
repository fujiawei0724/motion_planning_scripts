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
from tensorboardX import SummaryWriter
from scipy.integrate import odeint
from Double_DQN_net import DQN
from DDPGTrainer import Actor, Critic
from PPOTrainer import ActorCritic
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
        match_num = 0
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
            current_state_array = torch.from_numpy(current_state_array).to(torch.float32)

            # Calculate action predicted by network and its reward
            net_pred_action = policy_net.forward(current_state_array)
            net_pred_action = net_pred_action.unsqueeze(0).max(1)[1].item()
            print('Net pred action: {}'.format(net_pred_action))
            net_pred_action_reward, _, _, _, _, _ = env.runOnce(net_pred_action)

            # Traverse calculate the best action with the largest reward
            real_rewards = np.array([0.0 for _ in range(63)])
            for action in range(0, 63):
                print('Episode: {}, calculating action: {}'.format(epoch, action))
                real_rewards[action], _, _, _, _, _ = env.runOnce(action)
            # Get best action and reward
            best_action = np.argmax(real_rewards)
            best_reward = real_rewards[best_action]

            output_actions.append(net_pred_action)
            output_rewards.append(net_pred_action_reward)
            best_actions.append(best_action)
            best_rewards.append(best_reward)

            epoch += 1
            if net_pred_action == best_action:
                match_num += 1
            print('Test episode: {}, match num: {}, success rate: {}'.format(epoch, match_num, match_num / epoch))
        print('Output actions: {}'.format(output_actions))
        print('Output rewards: {}'.format(output_rewards))
        print('Best actions: {}'.format(best_actions))
        print('Best rewards: {}'.format(best_rewards))
        print('Final success rate: {}'.format(match_num / epoch))

    # Test DDPG
    """
    Note that DDPG cannot converge for this situation.
    """
    @staticmethod
    def testDDPG(test_episode=100):
        # Load network
        state_size = 94
        action_size = 63
        policy_net = Actor(state_size, action_size, 1.0)
        policy_net.load_state_dict(torch.load('./DDPG_weights/actor_checkpoint0.pt', map_location='cpu'))
        policy_net.eval()

        # Initialize container
        best_actions = []
        best_rewards = []
        output_actions = []
        output_rewards = []

        epoch = 0
        match_num = 0
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
            current_state_array = StateInterface.worldToNetDataAll([left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit], ego_vehicle, surround_vehicles)
            env.load(current_state_array)
            current_state_array = torch.from_numpy(current_state_array).to(torch.float32)


            # Calculate action predicted by network and its reward
            net_pred_action = policy_net.forward(current_state_array)
            net_pred_action = net_pred_action.unsqueeze(0).max(1)[1].item()
            print('Net pred action: {}'.format(net_pred_action))
            net_pred_action_reward, _, _, _, _, _ = env.runOnce(net_pred_action)

            # Traverse calculate the best action with the largest reward
            real_rewards = np.array([0.0 for _ in range(63)])
            for action in range(0, 63):
                print('Episode: {}, calculating action: {}'.format(epoch, action))
                real_rewards[action], _, _, _, _, _ = env.runOnce(action)
            # Get best action and reward
            best_action = np.argmax(real_rewards)
            best_reward = real_rewards[best_action]

            output_actions.append(net_pred_action)
            output_rewards.append(net_pred_action_reward)
            best_actions.append(best_action)
            best_rewards.append(best_reward)

            epoch += 1
            if net_pred_action == best_action:
                match_num += 1
            print('Test episode: {}, match num: {}, success rate: {}'.format(epoch, match_num, match_num / epoch))
        print('Output actions: {}'.format(output_actions))
        print('Output rewards: {}'.format(output_rewards))
        print('Best actions: {}'.format(best_actions))
        print('Best rewards: {}'.format(best_rewards))
        print('Final success rate: {}'.format(match_num / epoch))


    @staticmethod
    def testPPO(test_episode=100):
        # Load network
        state_size = 94
        action_size = 63
        policy_net = ActorCritic(state_size, action_size)
        policy_net.load_state_dict(torch.load('./PPO_weights/checkpoint.pth', map_location='cpu'))
        policy_net.eval()

        # Initialize container
        best_actions = []
        best_rewards = []
        output_actions = []
        output_rewards = []

        epoch = 0
        match_num = 0
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
            current_state_array = StateInterface.worldToNetDataAll([left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit], ego_vehicle, surround_vehicles)
            env.load(current_state_array)
            current_state_array = torch.from_numpy(current_state_array).to(torch.float32)


            # Calculate action predicted by network and its reward
            net_pred_action = policy_net.act(current_state_array)
            net_pred_action = net_pred_action.unsqueeze(0).max(1)[1].item()
            print('Net pred action: {}'.format(net_pred_action))
            net_pred_action_reward, _, _, _, _, _ = env.runOnce(net_pred_action)

            # Traverse calculate the best action with the largest reward
            real_rewards = np.array([0.0 for _ in range(63)])
            for action in range(0, 63):
                print('Episode: {}, calculating action: {}'.format(epoch, action))
                real_rewards[action], _, _, _, _, _ = env.runOnce(action)
            # Get best action and reward
            best_action = np.argmax(real_rewards)
            best_reward = real_rewards[best_action]

            output_actions.append(net_pred_action)
            output_rewards.append(net_pred_action_reward)
            best_actions.append(best_action)
            best_rewards.append(best_reward)

            epoch += 1
            if net_pred_action == best_action:
                match_num += 1
            print('Test episode: {}, match num: {}, success rate: {}'.format(epoch, match_num, match_num / epoch))
        print('Output actions: {}'.format(output_actions))
        print('Output rewards: {}'.format(output_rewards))
        print('Best actions: {}'.format(best_actions))
        print('Best rewards: {}'.format(best_rewards))
        print('Final success rate: {}'.format(match_num / epoch))





if __name__ == '__main__':
    # Tester.testPPO()
    Tester.testDDPG()


