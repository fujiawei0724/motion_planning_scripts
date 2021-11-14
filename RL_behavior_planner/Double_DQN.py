# -- coding: utf-8 --
# @Time : 2021/11/14 下午4:22
# @Author : fujiawei0724
# @File : DDQN.py
# @Software: PyCharm

"""
Implement a Double DQN model.
"""

import torch

class DQN(torch.nn.Module):
    def __init__(self, input_num, action_num):
        super(DQN, self).__init__()
        # Construct network
        self.fc1_ = torch.nn.Linear(input_num, 128)
        self.fc2_ = torch.nn.Linear(128, 128)
        self.head = torch.nn.Linear(128, action_num)

    # Forward
    def forward(self, x):
        x = torch.nn.ReLU(self.fc1_(x))
        x = torch.nn.ReLU(self.fc2_(x))
        x = self.head(x)
        return x


