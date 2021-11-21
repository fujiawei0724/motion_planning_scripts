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
        self.high_action_scalar_ = 1.0
        self.low_action_scalar = -1.0

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
        # Define train constants for current situation
        self._max_iteration_num = 3
        self._max_environment_reset_episode = 10000
        self._max_vehicle_info_reset_num = 100
        self.max_episode_ = 100000
        self.max_steps_ = 500
        self.batch_size_ = 64
        self.update_iteration_ = 200
        self.buffer_size_ = 1000000
        self.buffer_full_ = 64
        self.gamma_ = 0.99
        self.tau_ = 0.001
        self.exploration_noise_ = 0.1

        # Memory buffer
        self.memory_buffer_ = MemoryReplay(self.buffer_size_)
        # """
        # DEBUG: add initial data to warm start up
        # """
        # # Read data
        # with h5py.File('./data/data.h5', 'r') as f:
        #     # print(f.keys())
        #     actions = f['actions'][()]
        #     current_states = f['current_states'][()]
        #     dones = f['dones'][()]
        #     next_states = f['next_states'][()]
        #     rewards = f['rewards'][()]
        # for _ in range(0, 64):
        #     ran_index = random.randint(0, 999)
        #     # action = actions[ran_index]
        #     # Generate a action distribution randomly
        #     action = np.random.rand(63, )
        #     current_state = current_states[ran_index]
        #     done = dones[ran_index]
        #     next_state = next_states[ran_index]
        #     reward = rewards[ran_index]
        #     transition = Transition(torch.from_numpy(current_state).unsqueeze(0).to(torch.float32).to(self.device_), torch.from_numpy(action).unsqueeze(0).to(torch.float32).to(self.device_), torch.from_numpy(next_state).unsqueeze(0).to(torch.float32).to(self.device_), reward, done)
        #     self.memory_buffer_.update(transition)
        # """
        # END DEBUG
        # """

        # Log
        self.summary_ = SummaryWriter('./DDPG_logs/')
        # Define save path
        self.save_path_ = './DDPG_weights/'
        if not os.path.exists(self.save_path_):
            os.makedirs(self.save_path_)

        self.calculation_done_ = 0
        self.step_done_ = 0

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
            actions = torch.cat(actions).to(self.device_)
            rewards = torch.Tensor(rewards).unsqueeze(1).to(self.device_)
            next_states = torch.cat(next_states).to(self.device_)
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

        return avg_value_loss / self.update_iteration_, avg_policy_loss / self.update_iteration_

    # Train
    def train(self):
        # Environment information reset iteration
        for env_reset_episode in range(0, self._max_environment_reset_episode):
            # Construct training environment
            left_lane_exist = random.randint(0, 1)
            right_lane_exist = random.randint(0, 1)
            center_left_distance = random.uniform(3.0, 4.5)
            center_right_distance = random.uniform(3.0, 4.5)
            lane_speed_limit = random.uniform(10.0, 25.0)
            env = Environment()

            # Vehicles information reset iteration
            for vehicles_reset_episode in range(0, self._max_vehicle_info_reset_num):
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
                for i in range(0, self._max_iteration_num):
                    print('Start calculation epoch: {}'.format(self.calculation_done_))
                    self.calculation_done_ += 1

                    # Get action with noise
                    with torch.no_grad():
                        action = self.actor_.forward(torch.from_numpy(current_state_array).to(torch.float32).to(self.device_))
                        action = action.squeeze(0).cpu().numpy()
                    action = (action + np.random.normal(0, self.exploration_noise_, size=self.action_dim_)).clip(self.low_action_scalar, self.high_action_scalar_)
                    # Process action
                    # TODO: check the distribution of action
                    action_info = np.argmax(action)

                    # Execute selected action
                    reward, next_state_array, done, _, _, _ = env.runOnce(action_info)
                    # Store information to memory buffer
                    self.memory_buffer_.update(Transition(torch.from_numpy(current_state_array).unsqueeze(0).to(torch.float32).to(self.device_), torch.from_numpy(action).unsqueeze(0).to(torch.float32).to(self.device_), torch.from_numpy(current_state_array).unsqueeze(0).to(torch.float32).to(self.device_), reward, done))
                    # Update environment and current state
                    current_state_array = next_state_array
                    env.load(current_state_array)
                    # Sum reward
                    total_reward += reward

                    if done:
                        break

                # Judge if update
                if self.memory_buffer_.size() >= self.buffer_full_:
                    # Start epitomize
                    print('Optimization No. {} round'.format(self.step_done_))
                    value_loss, policy_loss = self.optimization()
                    self.step_done_ += 1
                    if self.step_done_ != 0 and self.step_done_ % 20 == 0:
                        # Save information
                        self.summary_.add_scalar('loss/value_loss', value_loss, self.step_done_)
                        self.summary_.add_scalar('loss/policy_loss', policy_loss, self.step_done_)
                        self.summary_.add_scalar('reward', total_reward, self.step_done_)
                        print('Episode ', self.calculation_done_, ' step: ', self.step_done_, ' reward: ', total_reward)
                        # Save weights
                        torch.save(self.actor_.state_dict(), self.save_path_ + 'actor_checkpoint' + str(self.step_done_ % 3) + '.pt')
                        torch.save(self.critic_.state_dict(), self.save_path_ + 'critic_checkpoint' + str(self.step_done_ % 3) + '.pt')

if __name__ == '__main__':
    trainer = DDPGTrainer()
    trainer.train()

