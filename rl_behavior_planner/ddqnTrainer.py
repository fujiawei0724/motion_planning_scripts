# -- coding: utf-8 --
# @Time : 2021/11/15 上午11:05
# @Author : fujiawei0724
# @File : train.py
# @Software: PyCharm

"""
Generate simulation data and use these data to train RL behavior planner.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import h5py
import time
import torch
from collections import namedtuple
from tensorboardX import SummaryWriter
from ddqnNet import DQN, DQN_resi
from lstm import BackboneNetwork
from imageGenerator import ImageGenerator
from statesSimulator import StatesSimulator
from memory import MemoryReplay
from environment import Environment, StateInterface, ActionInterface
from utils import *

# Data in memory buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Normalization coefficients
SPEED_NORM = 25.0
ACC_NORM = 3.0

# Behavior planner
class DDQNTrainer:
    def __init__(self):
        # Define environment
        self._env = Environment()
        # Define device
        self._device = torch.device('cuda:0')
        # Define action
        self._action = np.arange(0, 231, 1)

        # Define train constants
        # TODO: adjust parameters
        self._eps_start = 1
        self._eps_end = 0.1
        self._eps_decay = 1000000
        self._gamma = 0.5
        self._batch_size = 128
        self._buffer_full = 10000
        self._buffer_size = 50000
        self._target_update = 10000
        self._optimize_frequency = 4
        self._evaluation_frequency = 1000

        # Define train constants for current situation
        self._max_iteration_num = 3
        self._max_environment_reset_episode = 10000
        self._max_vehicle_info_reset_num = 100

        # Define network parameters
        self._policy_net = BackboneNetwork(10, 512, 2, 231).to(self._device)
        self._target_net = BackboneNetwork(10, 512, 2, 231).to(self._device)
        # self._policy_net.apply(self._policy_net.initWeights)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._target_net.lstm.train()

        # Define optimizer
        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=0.0000625, eps=1.5e-4)

        # Define memory buffer
        self._memory_replay = MemoryReplay(self._buffer_size)

        # Record optimization iteration number
        self._steps_done = 0

        # Record calculation iteration number
        self._calculation_done = 0

        # Define model store path and log store path
        self._save_path = './DDQN_weights/'
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        self._summary_writer = SummaryWriter('./DDQN_logs/')

    # Select action for train
    def selectAction(self, observations, additional_states, use_random=True):
        # current_state = torch.from_numpy(current_state).to(torch.float32).to(self._device)
        observations = torch.from_numpy(observations).to(torch.float32).to(self._device).unsqueeze(0)
        additional_states = torch.from_numpy(additional_states).to(torch.float32).to(self._device).unsqueeze(0)
        if use_random:
            sample = random.random()
            eps_threshold = self._eps_start + (self._eps_end - self._eps_start) * min(self._steps_done / self._eps_decay, 1.0)
            action = None
            if sample < eps_threshold:
                action = torch.IntTensor([random.choice(self._action)]).to(self._device)
            else:
                with torch.no_grad():
                    action = self._policy_net(observations, additional_states).max(1)[1]
            return action
        else:
            sample = random.random()
            eps_threshold = 0.05
            action = None
            if sample < eps_threshold:
                action = torch.IntTensor([random.choice(self._action)]).to(self._device)
            else:
                with torch.no_grad():
                    action = self._policy_net(observations, additional_states).max(1)[1]
            return action

    # Optimization for net
    def optimizeProcess(self):
        # Training mode
        self._policy_net.train()

        # Load data from memory
        memory_batch = self._memory_replay.getBatch(self._batch_size)

        # Split data
        cur_observations_batch, cur_additional_states_batch, action_batch, next_observations_batch, next_additional_states_batch, reward_batch, done_batch = [], [], [], [], [], [], []
        for data in memory_batch:
            cur_observations_batch.append(data.state[0].unsqueeze(0))
            cur_additional_states_batch.append(data.state[1].unsqueeze(0))
            action_batch.append(data.action)
            next_observations_batch.append(data.next_state[0].unsqueeze(0))
            next_additional_states_batch.append(data.next_state[1].unsqueeze(0))
            reward_batch.append(data.reward)
            done_batch.append(data.done)
        
        # Transform data
        cur_observations_batch = torch.cat(cur_observations_batch).to(self._device)
        cur_additional_states_batch = torch.cat(cur_additional_states_batch).to(self._device)
        action_batch = torch.cat(action_batch).unsqueeze(1).long().to(self._device)
        next_observations_batch = torch.cat(next_observations_batch).to(self._device)
        next_additional_states_batch = torch.cat(next_additional_states_batch).to(self._device)
        reward_batch = torch.Tensor(reward_batch).unsqueeze(1).to(self._device)
        done_batch = torch.Tensor(done_batch).unsqueeze(1).to(self._device)

        # # DEBUG
        # print('Cur observations size: {}'.format(cur_observations_batch.size()))
        # print('Cur additional states size: {}'.format(cur_additional_states_batch.size()))
        # print('Action batch size: {}'.format(action_batch.size()))
        # print('Next observations size: {}'.format(next_observations_batch.size()))
        # print('Next additional states batch size: {}'.format(next_additional_states_batch.size()))
        # print('Reward batch size: {}'.format(reward_batch.size()))
        # print('Done batch size: {}'.format(done_batch))
        # # END DEBUG

        # Forward calculate
        output = self._policy_net.forward(cur_observations_batch, cur_additional_states_batch).gather(1, action_batch)

        # Calculate policy net predict action for the next state
        next_state_action_predict_batch = self._policy_net.forward(next_observations_batch, next_additional_states_batch).max(1)[1].unsqueeze(1)

        # Calculate predict action corresponding Q by target net
        target_q = self._target_net.forward(next_observations_batch, next_additional_states_batch).gather(1, next_state_action_predict_batch)

        # Calculate ground truth
        # In the premise that the MDP process has a infinite length
        ground_truth = reward_batch + (1 - done_batch) * self._gamma * target_q

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

        # Initialize states simulator to create previous observations
        states_simulator = StatesSimulator()

        # Initialize environment model
        env = Environment()

        # Environment information reset iteration
        for env_reset_episode in range(0, self._max_environment_reset_episode):
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
            for vehicles_reset_episode in range(0, self._max_vehicle_info_reset_num):
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

                # Load all information to env
                # env.load(current_state_array)
                env.load(lane_info_with_speed, ego_vehicle, surround_vehicles)

                # Record reward
                total_reward = 0

                # Simulate a round, each round include three behavior sequences
                for i in range(0, self._max_iteration_num):
                    print('Start calculation epoch: {}'.format(self._calculation_done))
                    self._calculation_done += 1

                    # Calculate observations sequence and additional states for the current state
                    states_simulator.loadCurrentState(lane_info_with_speed, ego_vehicle, surround_vehicles)
                    _, cur_sur_vehs_states_t_order = states_simulator.runOnce()
                    cur_observations = image_generator.generateMultipleImages(cur_sur_vehs_states_t_order)
                    cur_additional_states = np.array([ego_vehicle.position_.y_, ego_vehicle.position_.theta_, ego_vehicle.velocity_ / SPEED_NORM, ego_vehicle.acceleration_ / ACC_NORM, ego_vehicle.curvature_, ego_vehicle.steer_, lane_speed_limit])

                    # Generate behavior
                    action = self.selectAction(cur_observations, cur_additional_states)

                    # Execute selected action
                    # Next state include two parts: ego vehicle state and surround vehicles states
                    # Surround vehicles' position information has been transformed based on ego vehicle's postion
                    # The abscissa of ego vehicle is maintained with 30.0 (or other specific values)
                    reward, next_state, done, _, _, _ = env.runOnce(action)

                    # Calculate observations sequence and additional states for the next state
                    next_ego_vehicle, next_surround_vehicles = next_state[0], next_state[1]
                    states_simulator.loadCurrentState(lane_info_with_speed, next_ego_vehicle, next_surround_vehicles)
                    _, next_sur_vehs_states_t_order = states_simulator.runOnce()
                    next_observations = image_generator.generateMultipleImages(next_sur_vehs_states_t_order)
                    next_additional_states = np.array([next_ego_vehicle.position_.y_, next_ego_vehicle.position_.theta_, next_ego_vehicle.velocity_ / SPEED_NORM, next_ego_vehicle.acceleration_ / ACC_NORM, next_ego_vehicle.curvature_, next_ego_vehicle.steer_, lane_speed_limit])

                    # Store information to memory buffer
                    self._memory_replay.update(Transition((torch.from_numpy(cur_observations).to(torch.float32), torch.from_numpy(cur_additional_states).to(torch.float32)), action, (torch.from_numpy(next_observations).to(torch.float32), torch.from_numpy(next_additional_states).to(torch.float32)), reward, done))

                    # Update environment and current state
                    ego_vehicle, surround_vehicles = next_ego_vehicle, next_surround_vehicles
                    env.load(lane_info_with_speed, ego_vehicle, surround_vehicles)
                    
                    # Sum reward
                    total_reward += reward

                    if done:
                        break

                # Judge if update
                if self._memory_replay.size() >= self._buffer_full:
                    # Execute optimization
                    if self._steps_done % self._optimize_frequency == 0:
                        print('Optimization No. {} round'.format(self._steps_done))
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
                if self._steps_done % 20 == 0 and self._steps_done != 0:
                    torch.save(self._policy_net.state_dict(), self._save_path + 'checkpoint' + str(self._steps_done % 3) + '.pt')
                    print('Episode: ', self._calculation_done, ', Steps: ', self._steps_done, ', Reward: ', total_reward)
                self._summary_writer.add_scalar('reward', total_reward, self._steps_done)

    # Evaluate training
    def evaluate(self, episodes=15):
        # Evaluation mode
        self._policy_net.eval()

        # Initialize data
        rewards = []
        states_simulator = StatesSimulator()
        env = Environment()

        for episode in range(0, episodes):
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

            # Construct ego vehicle and surround vehicles randomly
            ego_vehicle = EgoInfoGenerator.generateOnce()
            surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
            surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))

            # Judge whether available
            if not Tools.checkInitSituation(ego_vehicle, surround_vehicles):
                print('Initial situation error, reset vehicles information!!!')
                continue

            # Load all information to env
            # env.load(current_state_array)
            env.load(lane_info_with_speed, ego_vehicle, surround_vehicles)

            # Record reward
            total_reward = 0

            for _ in range(0, self._max_iteration_num):
                # Calculate observations sequence and additional states for the current state
                states_simulator.loadCurrentState(lane_info_with_speed, ego_vehicle, surround_vehicles)
                _, cur_sur_vehs_states_t_order = states_simulator.runOnce()
                cur_observations = image_generator.generateMultipleImages(cur_sur_vehs_states_t_order)
                cur_additional_states = np.array([ego_vehicle.position_.y_, ego_vehicle.position_.theta_, ego_vehicle.velocity_ / SPEED_NORM, ego_vehicle.acceleration_ / ACC_NORM, ego_vehicle.curvature_, ego_vehicle.steer_, lane_speed_limit])

                # Generate behavior
                action = self.selectAction(cur_observations, cur_additional_states)

                # Execute selected action
                reward, next_state, done, _, _, _ = env.runOnce(action)                
                
                # Update environment and current state
                next_ego_vehicle, next_surround_vehicles = next_state[0], next_state[1]
                ego_vehicle, surround_vehicles = next_ego_vehicle, next_surround_vehicles
                env.load(lane_info_with_speed, ego_vehicle, surround_vehicles)
                
                # Sum reward
                total_reward += reward

                if done:
                    break

            rewards.append(total_reward)

        return np.mean(rewards)


if __name__ == '__main__':
    trainer = DDQNTrainer()
    trainer.train()


