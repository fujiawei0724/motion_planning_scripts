# -- coding: utf-8 --
# @Time : 2021/11/15 上午11:05
# @Author : fujiawei0724
# @File : train.py
# @Software: PyCharm

"""
Generate simulation data and use these data to train RL behavior planner.
"""

import random
import numpy as np
import torch
from collections import namedtuple
from tensorboardX import SummaryWriter
from Double_DQN_net import DQN
from memory import MemoryReplay
from environment import Environment, StateInterface, ActionInterface
from utils import *

# Data in memory buffer
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

# Behavior planner
class DDQNTrainer:
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
        self._batch_size = 32
        self._buffer_full = 50000
        self._target_update = 10000
        self._optimize_frequency = 4
        self._evaluation_frequency = 100000

        # Define train constants for current situation
        self._max_iteration_num = 3
        self._max_environment_reset_episode = 10000
        self._max_vehicle_info_reset_num = 1000

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
    # Select action for train
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
            sample = random.random()
            eps_threshold = 0.05
            action_index = None
            if sample < eps_threshold:
                action_index = random.choice(self._action)
            else:
                with torch.no_grad():
                    action_index = self._policy_net(current_state.to(self._device)).max(1)[1]
            return action_index

    # Optimization for net
    def optimizeProcess(self):
        # Load data from memory
        memory_batch = self._memory_replay.getBatch(self._batch_size)
        # Split data
        state_batch, action_batch, next_state_batch, reward_batch = [], [], [], []
        for data in memory_batch:
            state_batch.append(data.state)
            action_batch.append(data.action)
            next_state_batch.append(data.next_state)
            reward_batch.append(data.reward)
        # Transform data
        state_batch = torch.cat(state_batch).to(self._device)
        action_batch = torch.cat(action_batch).unsqueeze(1).long().to(self._device)
        next_state_batch = torch.cat(next_state_batch).to(self._device)
        reward_batch = torch.Tensor(reward_batch).unsqueeze(1).to(self._device)

        # Forward calculate
        output = self._policy_net.forward(state_batch.to(self._device)).gather(1, action_batch)
        # Calculate policy net predict action for the next state
        next_state_action_predict_batch = self._policy_net.forward(next_state_batch.to(self._device)).max(1)[1].unsqueeze(1)
        # Calculate predict action corresponding Q by target net
        target_q = self._target_net.forward(next_state_batch.to(self._device)).gather(1, next_state_action_predict_batch)
        # Calculate ground truth
        # In the premise that the MDP process has a infinite length
        ground_truth = reward_batch + self._gamma * target_q
        # Loss calculation
        loss = torch.nn.SmoothL1Loss()(output, ground_truth)
        self._summary_writer.add_scalar('loss', loss, self._steps_done)
        # Optimization
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    # Train
    def train(self):
        # Environment information reset iteration
        for env_reset_episode in range(0, self._max_environment_reset_episode):
            # Construct training environment
            left_lane_exist = random.randint(0, 1)
            right_lane_exist = random.randint(0, 1)
            center_left_distance = random.uniform(3.0, 4.5)
            center_right_distance = random.uniform(3.0, 4.5)
            env = Environment()

            # Vehicles information reset iteration
            for vehicles_reset_episode in range(0, self._max_vehicle_info_reset_num):
                # Construct ego vehicle and surround vehicles randomly
                ego_vehicle = EgoInfoGenerator.generateOnce()
                surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
                surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))

                # Transform to state array
                current_state_array = StateInterface.worldToNetDataAll([left_lane_exist, right_lane_exist, center_left_distance, center_right_distance], ego_vehicle, surround_vehicles)

                # Load all information to env
                env.load(current_state_array)

                # Record reward
                total_reward = 0

                # Simulate a round, each round include three behavior sequences
                for _ in range(0, self._max_iteration_num):
                    # Generate behavior
                    action = self.selectAction(current_state_array)
                    # Execute selected action
                    next_state_array, reward = env.runOnce(action)
                    # Store information to memory buffer
                    self._memory_replay.update(Transition(current_state_array, action, next_state_array, reward))
                    # Update environment and current state
                    current_state_array = next_state_array
                    env.load(current_state_array)
                    # Sum reward
                    total_reward += reward

                    # Judge if update
                    if self._memory_replay.size() > self._buffer_full:
                        # Execute optimization
                        if self._steps_done % self._optimize_frequency == 0:
                            self.optimizeProcess()
                        # Update target network parameters
                        if self._steps_done % self._target_update == 0:
                            self._target_net.load_state_dict(self._policy_net.state_dict())
                            print('Update target net in {} round'.format(self._steps_done))
                        # Evaluate model
                        if self._steps_done % self._evaluation_frequency == 0:
                            evaluate_reward = self.evaluate()
                            self._summary_writer.add_scalar('evaluation', evaluate_reward, self._steps_done)
                            print('Step ', self._steps_done, ' Evaluation reward ', evaluate_reward)
                        self._steps_done += 1

                # Calculate current calculation number
                episode = env_reset_episode * self._max_vehicle_info_reset_num * self._max_iteration_num + vehicles_reset_episode * self._max_iteration_num
                if episode % 20 == 0:
                    torch.save(self._policy_net.state_dict(), self._save_path + 'checkpoint' + str(episode % 3) + '.pt')
                    print('Episode: ', episode, ', Steps: ', self._steps_done, ', Reward: ', total_reward)
                self._summary_writer.add_scalar('reward', total_reward, self._steps_done)

    # Evaluate training
    def evaluate(self, episodes=15):
        # Initialize data
        rewards = []
        env = Environment()

        for episode in range(0, episodes):
            # Load environment data randomly
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

            # Record reward
            total_reward = 0.0

            for _ in range(0, self._max_iteration_num):
                # Generate behavior
                action = self.selectAction(current_state_array, False)
                # Execute selected action
                next_state_array, reward = env.runOnce(action)
                # Update environment and current state
                current_state_array = next_state_array
                env.load(current_state_array)
                # Sum reward
                total_reward += reward

            rewards.append(total_reward)

        return np.mean(rewards)


if __name__ == '__main__':
    pass


