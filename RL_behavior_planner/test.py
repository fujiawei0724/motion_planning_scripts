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
import torch
from scipy.integrate import odeint
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

if __name__ == '__main__':

    print('Odeint based method.')
    start_time = time.time()
    start_state = np.array([2.0, 1.0, 0.2, 3.0, 0.04])
    def deriv(state, t):
        state_deriv = np.zeros((5, ))
        state_deriv[0] = np.cos(state[2]) * state[3]
        state_deriv[1] = np.sin(state[2]) * state[3]
        state_deriv[2] = np.tan(state[4]) * state[3] / 2.85
        state_deriv[3] = 0.5
        state_deriv[4] = 0.1
        return state_deriv

    def predict(start_state, t):
        return odeint(deriv, start_state, t)

    t = np.array([0.0, 0.4])
    predicted_state = predict(start_state, t)
    print(predicted_state)
    end_time = time.time()
    print('Odeint time consumption: {}'.format(end_time - start_time))

    print('Linear calculation based method.')
    start_state = time.time()
    start_state = np.array([2.0, 1.0, 0.2, 3.0, 0.04])

    def linearPredict(internal_state, dt):
        predict_state = [0.0 for _ in range(5)]
        predict_state[0] = internal_state[0] + dt * np.cos(internal_state[2]) * internal_state[3]
        predict_state[1] = internal_state[1] + dt * np.sin(internal_state[2]) * internal_state[3]
        predict_state[2] = internal_state[2] + dt * np.tan(internal_state[4]) * internal_state[3] / 2.85
        predict_state[3] = internal_state[3] + dt * 0.5
        predict_state[4] = internal_state[4] + dt * 0.1
        return predict_state

    current_state = copy.deepcopy(start_state)
    for _ in range(400):
        current_state = linearPredict(current_state, 0.001)

    print(start_state)
    print(current_state)
    end_time = time.time()
    print('Linear calculation time consumption: {}'.format(end_time - start_time))


