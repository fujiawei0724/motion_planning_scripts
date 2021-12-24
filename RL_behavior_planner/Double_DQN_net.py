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
        self.fc1_ = torch.nn.Linear(input_num, 512)
        self.fc2_ = torch.nn.Linear(512, 512)
        self.head = torch.nn.Linear(512, action_num)

    # Initialize weight
    def initWeights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == torch.nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # Forward
    def forward(self, x):
        x = torch.nn.ReLU()(self.fc1_(x))
        x = torch.nn.ReLU()(self.fc2_(x))
        x = self.head(x)
        return x

class DQN_resi(torch.nn.Module):
    def __init__(self, input_num, action_num):
        super(DQN_resi, self).__init__()
        # Construct network
        self.fc1_ = torch.nn.Linear(input_num, 512)
        self.fc2_ = torch.nn.Linear(512, 512)
        self.fc3_ = torch.nn.Linear(512, 512)
        self.fc4_ = torch.nn.Linear(512, 512)
        self.fc5_ = torch.nn.Linear(512, 512)
        self.fc6_ = torch.nn.Linear(512, 512)
        self.fc7_ = torch.nn.Linear(512, 512)
        self.fc8_ = torch.nn.Linear(512, 512)
        self.fc9_ = torch.nn.Linear(512, 512)
        self.head_ = torch.nn.Linear(512, action_num)

    def forward(self, x):
        x = torch.nn.ReLU()(self.fc1_(x))
        x = torch.nn.ReLU()(self.fc2_(x))
        x = torch.nn.ReLU()(self.fc3_(x))
        x_resi_1 = x.clone()
        x = torch.nn.ReLU()(self.fc4_(x))
        x = torch.nn.ReLU()(self.fc5_(x) + x_resi_1)
        x_resi_2 = x.clone()
        x = torch.nn.ReLU()(self.fc6_(x))
        x = torch.nn.ReLU()(self.fc7_(x) + x_resi_2)
        x_resi_3 = x.clone()
        x = torch.nn.ReLU()(self.fc8_(x))
        x = torch.nn.ReLU()(self.fc9_(x) + x_resi_3)
        x = self.head_(x)
        return x

