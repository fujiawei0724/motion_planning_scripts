# -- coding: utf-8 --
# @Time : 2021/11/23 下午8:03
# @Author : fujiawei0724
# @File : PPOTrainer.py
# @Software: PyCharm

"""
Train based on PPO.
"""

import os
import random

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import h5py
import time
import torch
from collections import namedtuple
from tensorboardX import SummaryWriter
from Double_DQN_net import DQN
from memory import MemoryReplay
from environment import Environment, StateInterface, ActionInterface
from utils import *

# Data in memory buffer
Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'done'))

# Actor-critic
class ActorCritic(torch.nn.Module):
    # Construct
    def __init__(self, state_dim, action_dim, hidden_num=512):
        super(ActorCritic, self).__init__()
        # Actor
        self.action_layer_ = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_num),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_num, hidden_num),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_num, action_dim),
            torch.nn.Softmax(dim=-1)
        )

        # Critic
        self.value_layer_ = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_num),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_num, hidden_num),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_num, 1)
        )

    # Act
    def act(self, x):
        return self.action_layer_(x)

    # Critic
    def critic(self, x):
        return self.value_layer_(x)

# PPO model
class PPOTrainer:
    def __init__(self):
        # Define device
        self.device_ = torch.device('cuda:0')
        # Define action
        self.action_ = np.arange(0, 63, 1)

        # Actor-critic
        state_dim = 94
        action_dim = 63
        self.actor_critic_ = ActorCritic(state_dim, action_dim).to(self.device_)
        # Optimizer
        self.optimizer_ = torch.optim.Adam(self.actor_critic_.parameters(), lr=0.002, betas=(0.9, 0.999))

        # Parameters
        # TODO: adjust the parameters
        self.max_iteration_num_ = 3
        self.max_environment_reset_episode_ = 10000
        self.max_vehicle_info_reset_num_ = 100
        self.gamma_ = 0.5
        self.optimization_epochs_ = 4
        self.memory_buffer_size_ = 100
        self.eps_clip_ = 0.2
        self.memory_buffer_ = []

        # Record
        self.calculation_done_ = 0

        # Store path
        self.summary_writer_ = SummaryWriter('./PPO_logs')
        self.weight_path_ = './PPO_weights/'
        if not os.path.exists(self.weight_path_):
            os.makedirs(self.weight_path_)

    # Optimize model
    def optimization(self):
        assert(len(self.memory_buffer_) == self.memory_buffer_size_)
        # Loss record
        loss_record = 0.0
        # Parse memory
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        for data in self.memory_buffer_:
            states.append(data.state)
            actions.append(data.action)
            log_probs.append(data.log_prob)
            rewards.append(data.reward)
            dones.append(data.done)

        # Calculate G
        G = []
        g_t = 0.0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if done:
                g_t = 0.0
            g_t = reward.item() + self.gamma_ * g_t
            G.insert(0, g_t)
        # Format
        states = torch.cat(states).to(self.device_)
        actions = torch.cat(actions).unsqueeze(1).to(self.device_)
        log_probs = torch.cat(log_probs).float().unsqueeze(1).to(self.device_)
        G = torch.Tensor(G).float().unsqueeze(1).to(self.device_)
        # Modify
        G = (G - G.mean()) / (G.std() + 1e-5)
        # Start optimization
        for _ in range(0, self.optimization_epochs_):
            # Calculate state corresponding action possibility
            new_action_probs = self.actor_critic_.act(states)
            # Distribution
            distribution = torch.distributions.Categorical(new_action_probs)
            # Calculate new probs
            new_log_probs = distribution.log_prob(actions.squeeze(1)).unsqueeze(1)
            # Calculate ratio
            ratio = torch.exp(new_log_probs - log_probs)
            # Calculate entropy
            entropy = distribution.entropy()
            # Calculate state corresponding value
            new_values = self.actor_critic_.critic(states)
            # Calculate advantage
            advantage = G - new_values.detach()
            # Calculate policy loss
            policy_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.eps_clip_, 1 + self.eps_clip_) * advantage)
            policy_loss = policy_loss.mean()
            # Calculate value loss
            value_loss = torch.nn.MSELoss()(new_values, G)
            # Calculate total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.means()
            # Record loss
            loss_record += loss.item()
            # Optimize
            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()

        # Finish optimization, clear memory
        self.memory_buffer_.clear()
        return loss_record / self.optimization_epochs_

    # Train
    def train(self):
        # Environment information reset iteration
        for env_reset_episode in range(0, self.max_environment_reset_episode_):
            # Construct training environment
            left_lane_exist = random.randint(0, 1)
            right_lane_exist = random.randint(0, 1)
            center_left_distance = random.uniform(3.0, 4.5)
            center_right_distance = random.uniform(3.0, 4.5)
            lane_speed_limit = random.uniform(10.0, 25.0)
            env = Environment()

            # Vehicles information reset iteration
            for vehicles_reset_episode in range(0, self.max_vehicle_info_reset_num_):
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

                # Load all information to env
                env.load(current_state_array)

                # Record reward
                total_reward = 0

                # Simulate a round, each round include three behavior sequences
                for i in range(0, self.max_iteration_num_):
                    print('Start calculation epoch: {}'.format(self.calculation_done_))
                    self.calculation_done_ += 1

                    # Get action
                    with torch.no_grad():
                        action_probs = self.actor_critic_.act(torch.from_numpy(current_state_array).to(torch.float32).to(self.device_))
                        distribution = torch.distributions.Categorical(action_probs)
                        action = distribution.sample()
                        log_prob = distribution.log_prob(action)
                    
