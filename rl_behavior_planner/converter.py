#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/11 下午9:50
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : converter.py
# @Software: PyCharm

"""
Convert the formation of the trained model to be deployed in C++.
"""

import torch
import torchvision
from ddqnNet import DQN, DQN_resi
from lstm import BackboneNetwork
from imageGenerator import ImageGenerator
from statesSimulator import StatesSimulator
from utils import *

SPEED_NORM = 25.0
ACC_NORM = 3.0

if __name__ == '__main__':
    # Official model
    # model = torchvision.models.resnet18()
    # traced_script_module = torch.jit.script(model)
    # traced_script_module.save('model.pt')

    # model = DQN(94, 231)
    # model.load_state_dict(torch.load('./model/checkpoint2.pt', map_location='cpu'))
    # model.eval()
    # dummy_example = torch.rand(1, 94)
    # traced_module = torch.jit.trace(model, dummy_example)
    # traced_module.save('./model/model2.pt')

    # test_input = torch.ones(1, 94)
    # test_res = model.forward(test_input)
    # print('Test res: {}'.format(test_res))

    model = BackboneNetwork(10, 512, 2, 231)
    model.load_state_dict(torch.load('/home/fjw/Desktop/checkpoint2.pt', map_location='cpu'))
    model.to(torch.device('cuda:0'))
    model.eval()
    # Load environment data randomly
    left_lane_exist = random.randint(0, 1)
    right_lane_exist = random.randint(0, 1)
    center_left_distance = random.uniform(3.0, 4.5)
    center_right_distance = random.uniform(3.0, 4.5)
    lane_info = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance]
    lane_speed_limit = random.uniform(10.0, 25.0)
    lane_info_with_speed = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit]

    # Initialize image generator
    image_generator = ImageGenerator(lane_info)

    # Construct ego vehicle and surround vehicles randomly
    ego_vehicle = EgoInfoGenerator.generateOnce()
    surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
    surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))
    
    # Calculate observations sequence and additional states for the current state
    states_simulator = StatesSimulator()
    states_simulator.loadCurrentState(lane_info_with_speed, ego_vehicle, surround_vehicles)
    _, cur_sur_vehs_states_t_order = states_simulator.runOnce()
    observations = image_generator.generateMultipleImages(cur_sur_vehs_states_t_order)
    additional_states = np.array([ego_vehicle.position_.y_, ego_vehicle.position_.theta_, ego_vehicle.velocity_ / SPEED_NORM, ego_vehicle.acceleration_ / ACC_NORM, ego_vehicle.curvature_, ego_vehicle.steer_, lane_speed_limit])

    # Transform formation
    observations = torch.from_numpy(observations).to(torch.float32).unsqueeze(0).to(torch.device('cuda:0'), non_blocking=True)
    additional_states = torch.from_numpy(additional_states).to(torch.float32).unsqueeze(0).to(torch.device('cuda:0'), non_blocking=True)

    traced_module = torch.jit.trace(model, (observations, additional_states))
    traced_module.save('./model/model0.pt')







