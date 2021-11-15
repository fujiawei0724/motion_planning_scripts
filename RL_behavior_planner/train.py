# -- coding: utf-8 --
# @Time : 2021/11/15 上午11:05
# @Author : fujiawei0724
# @File : train.py
# @Software: PyCharm

"""
Generate simulation data and use these data to train neural network.
"""

import numpy as np
import torch
from memory import MemoryReplay
from environment import Environment, StateInterface, ActionInterface

# Behavior planner
class RlBehaviorPlanner:
    def __init__(self):
        # Define environment
        self._env = Environment()
        # Define device
        self._device = torch.device('cuda:0')
        # Define action
        self._action = np.arange(0, 63, 1)
        
