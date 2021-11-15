# -- coding: utf-8 --
# @Time : 2021/11/15 上午11:05
# @Author : fujiawei0724
# @File : train.py
# @Software: PyCharm

"""
Generate simulation data and use these data to train neural network.
"""
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from Double_DQN import DQN
from memory import MemoryReplay
from environment import Environment, StateInterface, ActionInterface

# Behavior planner
class DDQNBehaviorPlanner:
    def __init__(self):
        # Define environment
        self._env = Environment()
        # Define device
        self._device = torch.device('cuda:0')
        # Define action
        self._action = np.arange(0, 63, 1)

        # Define train constants
        # TODO: adjust parameters
        self._eps_start = 1
        self._eps_end = 0.1
        self._eps_decay= 1000000
        self._gamma = 0.99
        self._max_episode = 500000
        self._batch_size = 32
        self._buffer_full = 50000
        self._target_update = 10000
        self._optimize_frequency = 4
        self._evaluation_frequency = 100000

        # Define network parameters
        # TODO: add state representation here
        state_length = None
        self._policy_net = DQN(None, 63).to(self._device)
        self._target_net = DQN(None, 63).to(self._device)
        self._policy_net.apply(self._policy_net.initWeights)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        # Define optimizer
        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=0.0000625, eps=1.5e-4)

        # Define memory buffer
        self._memory_replay = MemoryReplay(self._batch_size)

        # Record iteration number
        self._steps_done = 0

        # Define model store path and log store path
        self._save_path = '../weights/'
        self._summary_writer = SummaryWriter('../logs/')

    # TODO: in our problem, what is environmental exploration mean? In other word, is continuous exploration valid?
    def selectAction(self, current_state, use_random=True):
        if use_random:
            sample = random.random()
            eps_threshold = self._eps_start + (self._eps_end - self._eps_start) * min(self._steps_done / self._eps_decay, 1.0)
            action_index = None
            if sample < eps_threshold:
                action_index = random.choice(self._action)
            else:
                with torch.no_grad():
                    action_index = self._policy_net(current_state.to(self._device)).max(1)[1]
            return action_index
        else:
            pass

if __name__ == '__main__':
    test_array = np.array([1.0, 2.0, 4.0, 6.0])
    a = random.choice(test_array)
    print(a)



