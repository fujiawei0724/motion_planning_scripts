# -- coding: utf-8 --
# @Time : 2021/11/18 下午10:00
# @Author : fujiawei0724
# @File : DDPGTrainer.py
# @Software: PyCharm

"""
DDPG based method for behavior planner.
"""

import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import torch
from collections import namedtuple
from tensorboardX import SummaryWriter
from Double_DQN_net import DQN
from memory import MemoryReplay
from environment import Environment, StateInterface, ActionInterface
from utils import *

# Data in memory buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Actor
class Actor(torch.nn.Module):
    def __init__(self, input_dim, action_dim, action_scalar):
        super(Actor, self).__init__()
        # Network
        self.input_layer_ = torch.nn.Linear(input_dim, 400)
        self.hidden_layer_ = torch.nn.Linear(400, 300)
        self.output_layer_ = torch.nn.Linear(300, action_dim)
        self.action_scalar_ = action_scalar

    # Forward
    def forward(self, x):
        x = torch.nn.ReLU()(self.input_layer_(x))
        x = torch.nn.ReLU()(self.hidden_layer_(x))
        x = self.action_scalar_ * torch.nn.Tanh()(self.output_layer_(x))
        return x

# Critic
class Critic(torch.nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        # Network
        self.input_layer_ = torch.nn.Linear(input_dim + action_dim, 400)
        self.hidden_layer_ = torch.nn.Linear(400, 300)
        self.output_layer_ = torch.nn.Linear(300, 1)

    # Forward
    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = torch.nn.ReLU()(self.input_layer_(x))
        x = torch.nn.ReLU()(self.hidden_layer_(x))
        x = self.output_layer_(x)
        return x

class DDPGTrainer:
    def __init__(self):
        # Input and output dimension
        self.input_dim_ = 94
        self.action_dim_ = 63

        # TODO: check these two ratio carefully
        self.high_action_scalar_ = None
        self.low_action_scalar = None

        # Device
        self.device_ = torch.device('cuda:0')
        # Network
        self.actor_ = Actor(self.input_dim_, self.action_dim_, self.high_action_scalar_).to(self.device_)
        self.target_actor_ = Actor(self.input_dim_, self.action_dim_, self.high_action_scalar_).to(self.device_)
        self.critic_ = Critic(self.input_dim_, self.action_dim_).to(self.device_)
        self.target_critic_ = Critic(self.input_dim_, self.action_dim_).to(self.device_)
        # Initial parameters
        self.target_actor_.load_state_dict(self.actor_.state_dict())
        self.target_critic_.load_state_dict(self.critic_.state_dict())
        # Define optimizer
        self.actor_optimizer_ = torch.optim.Adam(self.actor_.parameters(), 0.0001)
        self.critic_optimizer_ = torch.optim.Adam(self.critic_.parameters(), 0.001)
        # Hyper-parameters
        self.max_episode_ = 100000
        self.max_steps_ = 500
        self.batch_size_ = 64
        self.update_iteration_ = 200
        self.buffer_size_ = 1000000
        self.buffer_full_ = 50000
        self.gamma_ = 0.99
        self.tau_ = 0.001
        self.exploration_noise_ = 0.1
        # Memory buffer
        self.memory_buffer_ = MemoryReplay(self.buffer_size_)
        # Log
        self.summary_ = SummaryWriter('./logs/')
        # Define save path
        self.save_path_ = './weights/'

    # Optimization
    def optimization(self):
        avg_value_loss, avg_policy_loss = 0.0, 0.0
        for _ in range(0, self.update_iteration_):
            # Sample batch size data
            samples_datas = self.memory_buffer_.getBatch(self.batch_size_)
            # Parse sample data
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for data in samples_datas:
                states.append(data.state)
                actions.append(data.action)
                rewards.append(data.reward)
                next_states.append(data.next_state)
                dones.append(1 - data.done)
            # Format
            states = torch.cat(states).to(self.device_)
            actions = torch.Tensor(actions).to(self.device_)
            rewards = torch.Tensor(rewards).unsqueeze(1).to(self.device_)
            next_states = torch.Tensor(next_states).to(self.device_)
            dones = torch.Tensor(dones).unsqueeze(1).to(self.device_)
            # Calculate predict value
            predict_value = self.critic_(states, actions)
            # Calculate target value
            target_value = (dones * self.gamma_ * self.target_critic_(next_states, self.target_actor_(next_states))).detach() + rewards
            # Calculate value loss
            value_loss = torch.nn.MSELoss()(predict_value, target_value)
            # Update critic
            self.critic_optimizer_.zero_grad()
            value_loss.backward()
            self.critic_optimizer_.step()
            # Record value loss
            avg_value_loss += value_loss.item()
            # Calculate policy loss
            policy_loss = -self.critic_(states, self.actor_(states)).mean()
            # Update actor
            self.actor_optimizer_.zero_grad()
            policy_loss.backward()
            self.actor_optimizer_.step()
            # Record
            avg_policy_loss += policy_loss.item()
            # Update target net
            for param, target_param in zip(self.actor_.parameters(), self.target_actor_.parameters()):
                target_param.data.copy_(self.tau_ * param.data + (1 - self.tau_) * target_param.data)
            for param, target_param in zip(self.critic_.parameters(), self.target_critic_.parameters()):
                target_param.data.copy_(self.tau_ * param.data + (1 - self.tau_) * target_param.data)

    # Train
    def train(self):
        episode = 0
        while episode < self.max_episode_:
            pass
