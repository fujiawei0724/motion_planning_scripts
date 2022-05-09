'''
Author: fujiawei0724
Date: 2022-05-05 21:06:19
LastEditors: fujiawei0724
LastEditTime: 2022-05-09 21:49:22
Description: network structure
'''

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from tensorboardX import SummaryWriter
from imageGenerator import ImageGenerator
from utils import *


class BackboneNetwork(nn.Module):
    
    def __init__(self, input_seq_length, hidden_dim, hidden_layer_num, output_dim):
        super(BackboneNetwork, self).__init__()
        # self.input_image_size = input_image_size
        self.input_seq_length = input_seq_length
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.hidden_layer_num = hidden_layer_num
        self.output_dim = output_dim
        self.lstm = nn.LSTM(40*61*1, hidden_dim, hidden_layer_num, batch_first=True)
        self.convs = nn.ModuleList([nn.Sequential(
                nn.Conv2d(1, 10, 21, 1),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(10, 20, 11, 1),
                nn.ReLU(True), 
                nn.MaxPool2d(2), 
                nn.Conv2d(20, 40, 5, 1), 
                nn.ReLU(True), 
            ) for _ in range(input_seq_length)
        ])
        self.fc1 = nn.Sequential(
            nn.Linear(6 + hidden_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512), 
            nn.Dropout(0.5), 
            nn.ReLU(True),
        )
        self.output = nn.Linear(512, output_dim)
    
    def forward(self, x, s):
        processed_xs = []
        for i, branch in enumerate(self.convs):
            cur_x = branch(x[:, i, :, :, :])
            cur_x = cur_x.view(cur_x.size(0), -1, 40*61*1)
            processed_xs.append(cur_x)
        concat_x = torch.cat(processed_xs, 1)
        x, _ = self.lstm(concat_x)
        x = x[:, -1, :]
        x = torch.cat((x, s), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

if __name__ == '__main__':
    # Produce test observe scenes sequence
    lane_info = [1, 1, 4.0, 4.0]
    agent_generator = AgentGenerator(lane_info[0], lane_info[1], lane_info[2], lane_info[3])
    image_generator = ImageGenerator()
    observed_scene_seq = []
    for _ in range(10):
        surround_vehicles = agent_generator.generateAgents(10)
        observed_scene = image_generator.generateSingleImage(lane_info, surround_vehicles)
        observed_scene_seq.append(observed_scene)
    observed_scene_seq = np.array(observed_scene_seq)
    # print(observed_scene_seq.shape)

    # Produce ego vehicle states
    ego_vehicle = EgoInfoGenerator.generateOnce()
    ego_vehicle_state = np.array([ego_vehicle.position_.theta_, ego_vehicle.velocity_, ego_vehicle.acceleration_, ego_vehicle.curvature_, ego_vehicle.steer_])

    # Test data input
    device = torch.device('cuda:0')
    model = BackboneNetwork(10, 512, 2, 231).to(device)
    # observed_scene_seq = torch.from_numpy(observed_scene_seq).to(torch.float32).to(device).unsqueeze(0)
    # ego_vehicle_state = torch.from_numpy(ego_vehicle_state).to(torch.float32).to(device).unsqueeze(0)
    # print('Input size_1: {}'.format(observed_scene_seq.size()))
    # print('Input size_2: {}'.format(ego_vehicle_state.size()))
    # output = model(observed_scene_seq, ego_vehicle_state)
    # print('Ouput size: {}'.format(output.size()))

    # Test batch data input
    observed_scene_seq_batch, ego_vehicle_state_batch = [], []
    for _ in range(128):
        observed_scene_seq_batch.append(observed_scene_seq)
        ego_vehicle_state_batch.append(ego_vehicle_state)
    observed_scene_seq_batch = np.array(observed_scene_seq_batch)
    ego_vehicle_state_batch = np.array(ego_vehicle_state_batch)
    observed_scene_seq_batch = torch.from_numpy(observed_scene_seq_batch).to(torch.float32).to(device)
    ego_vehicle_state_batch = torch.from_numpy(ego_vehicle_state_batch).to(torch.float32).to(device)
    print('Input size_1: {}'.format(observed_scene_seq_batch.size()))
    print('Input size_2: {}'.format(ego_vehicle_state_batch.size()))
    output_batch = model(observed_scene_seq_batch, ego_vehicle_state_batch)
    print('Output size: {}'.format(output_batch.size()))

    # Visualize network structure
    # with SummaryWriter('./log', comment='network visualization') as sw:
    #     sw.add_graph(model, observed_scene_seq, ego_vehicle_state)
    # os.mkdir('./log/')
    # torch.save(model, './log/test.pt')
    



    


        
        
        

        


