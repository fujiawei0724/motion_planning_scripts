# -- coding: utf-8 --
# @Time : 2021/11/23 下午8:03
# @Author : fujiawei0724
# @File : PPOTrainer.py
# @Software: PyCharm

"""
Train based on PPO.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import h5py
import time
import torch
from torch import nn
from collections import namedtuple
from tensorboardX import SummaryWriter
from imageGenerator import ImageGenerator
from statesSimulator import StatesSimulator
from environment import Environment, StateInterface, ActionInterface
from utils import *

# Data in memory buffer
Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'done'))

# Normalization coefficients
SPEED_NORM = 25.0
ACC_NORM = 3.0

class ActorCritic(torch.nn.Module):
    def __init__(self, input_seq_length, hidden_dim, hidden_layer_num, output_dim):
        super(ActorCritic, self).__init__()
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
            # The additional number means the additional states length
            nn.Linear(7 + hidden_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512), 
            nn.Dropout(0.5), 
            nn.ReLU(True),
        )
        self.act_layer = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.Softmax(dim=-1),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.critic_layer = nn.Linear(512, 1)
    
    # Actor
    def act(self, x, s):
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
        x = self.act_layer(x)
        return x

    # Critic
    def critic(self, x, s):
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
        x = self.fc3(x)
        x = self.critic_layer(x)
        return x

# PPO model
class PPOTrainer:
    def __init__(self):
        # Define device
        self.device_ = torch.device('cuda:0')

        # Actor-critic
        self.actor_critic_ = ActorCritic(10, 512, 2, 231).to(self.device_)
        # Optimizer
        self.optimizer_ = torch.optim.Adam(self.actor_critic_.parameters(), lr=0.002, betas=(0.9, 0.999))

        # Parameters
        # TODO: adjust the parameters
        self.max_iteration_num_ = 3
        self.max_environment_reset_episode_ = 10000
        self.max_vehicle_info_reset_num_ = 100
        self.gamma_ = 0.5
        self.optimization_epochs_ = 4
        self.memory_buffer_size_ = 200
        self.eps_clip_ = 0.2
        self.memory_buffer_ = []

        # Record
        self.calculation_done_ = 0
        self.optimization_done_ = 0

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
        observations, additional_states, actions, log_probs, rewards, dones = [], [], [], [], [], []
        for data in self.memory_buffer_:
            observations.append(torch.from_numpy(data.state[0]).to(torch.float32).to(self.device_).unsqueeze(0))
            additional_states.append(torch.from_numpy(data.state[1]).to(torch.float32).to(self.device_).unsqueeze(0))
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
            g_t = reward + self.gamma_ * g_t
            G.insert(0, g_t)
        # Format
        observations = torch.cat(observations).to(self.device_)
        additional_states = torch.cat(additional_states).to(self.device_)
        actions = torch.Tensor(actions).unsqueeze(1).to(self.device_)
        log_probs = torch.Tensor(log_probs).float().unsqueeze(1).to(self.device_)
        G = torch.Tensor(G).float().unsqueeze(1).to(self.device_)
        # Modify
        G = (G - G.mean()) / (G.std() + 1e-5)
        # Start optimization
        for sub_epoch in range(0, self.optimization_epochs_):
            # Calculate state corresponding action possibility
            new_action_probs = self.actor_critic_.act(observations, additional_states)
            # Distribution
            distribution = torch.distributions.Categorical(new_action_probs)
            # Calculate new probs
            new_log_probs = distribution.log_prob(actions.squeeze(1)).unsqueeze(1)
            # Calculate ratio
            ratio = torch.exp(new_log_probs - log_probs)
            # Calculate entropy
            entropy = distribution.entropy()
            # Calculate state corresponding value
            new_values = self.actor_critic_.critic(observations, additional_states)
            # Calculate advantage
            advantage = G - new_values.detach()
            # Calculate policy loss
            policy_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.eps_clip_, 1 + self.eps_clip_) * advantage)
            policy_loss = policy_loss.mean()
            # Calculate value loss
            value_loss = torch.nn.MSELoss()(new_values, G)
            # Calculate total loss
            loss = policy_loss + 0.5 * value_loss - 0.1 * entropy.mean()
            self.summary_writer_.add_scalar('loss', loss, self.optimization_done_)
            print('Optimization epoch: {}, sub epoch: {}, loss: {}'.format(self.optimization_done_, sub_epoch, loss.item()))
            # Record loss
            loss_record += loss.item()
            # Optimize
            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()

        # Finish optimization, clear memory
        self.memory_buffer_.clear()
        self.optimization_done_ += 1

        return loss_record / self.optimization_epochs_

    # Train
    def train(self):
        # Initialize states simulator to create previous observations
        states_simulator = StatesSimulator()

        # Initialize environment model
        env = Environment()

        # Environment information reset iteration
        for env_reset_episode in range(0, self.max_environment_reset_episode_):
            # Construct training environment
            left_lane_exist = random.randint(0, 1)
            right_lane_exist = random.randint(0, 1)
            center_left_distance = random.uniform(3.0, 4.5)
            center_right_distance = random.uniform(3.0, 4.5)
            lane_info = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance]
            lane_speed_limit = random.uniform(10.0, 25.0)
            lane_info_with_speed = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit]

            # Initialize image generator
            image_generator = ImageGenerator(lane_info)

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

                # # Transform to state array
                # current_state_array = StateInterface.worldToNetDataAll([left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit], ego_vehicle, surround_vehicles)

                # # Load all information to env
                # env.load(current_state_array)
                # current_state_array = torch.from_numpy(current_state_array).unsqueeze(0).to(torch.float32).to(self.device_)

                env.load(lane_info_with_speed, ego_vehicle, surround_vehicles)

                # Record
                step_done = 0
                total_reward = 0

                # Simulate a round, each round include three behavior sequences
                for i in range(0, self.max_iteration_num_):
                    print('Start calculation epoch: {}'.format(self.calculation_done_))
                    self.calculation_done_ += 1

                    # Calculate observations sequence and additional states for the current state
                    states_simulator.loadCurrentState(lane_info_with_speed, ego_vehicle, surround_vehicles)
                    _, cur_sur_vehs_states_t_order = states_simulator.runOnce()
                    cur_observations = image_generator.generateMultipleImages(cur_sur_vehs_states_t_order)
                    cur_additional_states = np.array([ego_vehicle.position_.y_, ego_vehicle.position_.theta_, ego_vehicle.velocity_ / SPEED_NORM, ego_vehicle.acceleration_ / ACC_NORM, ego_vehicle.curvature_, ego_vehicle.steer_, lane_speed_limit])

                    # # Transform 
                    # cur_observations = torch.from_numpy(cur_observations).to(torch.float32).to(self.device_).unsqueeze(0)
                    # cur_additional_states = torch.from_numpy(cur_additional_states).to(torch.float32).to(self.device_).unsqueeze(0)

                    # Get action
                    with torch.no_grad():
                        action_probs = self.actor_critic_.act(torch.from_numpy(cur_observations).to(torch.float32).to(self.device_).unsqueeze(0), torch.from_numpy(cur_additional_states).to(torch.float32).to(self.device_).unsqueeze(0))
                        distribution = torch.distributions.Categorical(action_probs)
                        action = distribution.sample()
                        log_prob = distribution.log_prob(action)

                    # Update
                    reward, next_state, done, _, _, _ = env.runOnce(action)

                    # Calculate observations sequence and additional states for the next state
                    next_ego_vehicle, next_surround_vehicles = next_state[0], next_state[1]
                    states_simulator.loadCurrentState(lane_info_with_speed, next_ego_vehicle, next_surround_vehicles)
                    # _, next_sur_vehs_states_t_order = states_simulator.runOnce()
                    # next_observations = image_generator.generateMultipleImages(next_sur_vehs_states_t_order)
                    # next_additional_states = np.array([next_ego_vehicle.position_.y_, next_ego_vehicle.position_.theta_, next_ego_vehicle.velocity_ / SPEED_NORM, next_ego_vehicle.acceleration_ / ACC_NORM, next_ego_vehicle.curvature_, next_ego_vehicle.steer_, lane_speed_limit])

                    # Record reward
                    total_reward += reward
                    
                    # Store memory
                    self.memory_buffer_.append(Transition((cur_observations, cur_additional_states), action, log_prob, reward, done))
                    
                    # Judge if update
                    if len(self.memory_buffer_) >= self.memory_buffer_size_:
                        self.optimization()
                        # torch.cuda.empty_cache()
                    
                    # Update environment and current state
                    ego_vehicle, surround_vehicles = next_ego_vehicle, next_surround_vehicles
                    env.load(lane_info_with_speed, ego_vehicle, surround_vehicles)

                    # Judge done
                    if done:
                        break
                    step_done += 1

                self.summary_writer_.add_scalar('reward', total_reward, self.calculation_done_)
                if self.optimization_done_ % 10 == 0:
                    # Print info
                    print('Episode: {}, reward :{}'.format(self.calculation_done_, total_reward))
                    # Save model
                    torch.save(self.actor_critic_.state_dict(), self.weight_path_ + 'checkpoint.pth')


if __name__ == '__main__':
    ppo = PPOTrainer()
    ppo.train()
