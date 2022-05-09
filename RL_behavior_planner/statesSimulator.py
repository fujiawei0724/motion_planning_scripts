'''
Author: fujiawei0724
Date: 2022-05-07 18:09:21
LastEditors: fujiawei0724
LastEditTime: 2022-05-09 21:00:59
Description: simulate the states sequence from the current observed state.
'''

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np 
from collections import defaultdict
from utils import *
from imageGenerator import ImageGenerator


class StatesSimulator:
    '''
    param {seq_length} is the length of the required states sequence.
    param {t_interval} is the time gap between two adjacent states. 
    '''    
    def __init__(self, seq_length=10, t_intertval=0.2):
        self.seq_length_ = seq_length
        self.t_intertval_ = t_intertval
        self.lane_info_ = None
        self.ego_vehicle_ = None
        self.surround_vehicles_ = None
    

    def loadCurrentState(self, state_array):
        pass

    '''
    description: supply information.
    '''
    def loadCurrentState(self, lane_info, ego_vehicle, surround_vehicles):
        self.lane_info_ = lane_info
        self.ego_vehicle_ = ego_vehicle
        self.surround_vehicles_ = surround_vehicles

    '''
    description: calculate the previous state given the current state. Note that this simulation process is a very simple algorithm. 
    TODO: refine parameters to improve fidelity to the real situation.
    param {vehicle} current state (ego vehicle/surround vehicle). 
    return previous state of the current state with a designed time interval.
    '''    
    def stateBackwardSingleStep(self, vehicle):
        # Calculate speed and acceleration
        pre_velocity = max(vehicle.velocity_ - self.t_intertval_ * vehicle.acceleration_, 0.0)
        pre_acceleration = vehicle.acceleration_

        # Calculate distance
        # s = (vehicle.velocity_ ** 2.0 - pre_velocity ** 2.0) / 2.0 * pre_acceleration * self.t_intertval_
        s = pre_velocity * self.t_intertval_

        # Calculate theta, curvature, and steer
        pre_theta = vehicle.position_.theta_ * 0.9
        pre_curvature = vehicle.curvature_ * 0.9
        pre_steer = vehicle.steer_ * 0.9

        # Calculate abscissa and ordinate
        pre_x = vehicle.position_.x_ - s * np.cos(pre_theta)
        pre_y = vehicle.position_.y_ - s * np.sin(pre_theta)

        return Vehicle(vehicle.id_, PathPoint(pre_x, pre_y, pre_theta), vehicle.length_, vehicle.width_, pre_velocity, pre_acceleration, vehicle.time_stamp_ - self.t_intertval_, pre_curvature, pre_steer)
    
    '''
    description: backward multiple steps from the current state. 
    '''    
    def stateBackwardMultipleSteps(self, vehicle):
        state_steps = [vehicle]
        for _ in range(self.seq_length_ - 1):
            pre_vehicle = self.stateBackwardSingleStep(vehicle)
            state_steps.append(pre_vehicle)
            vehicle = pre_vehicle
        state_steps.reverse()
        return state_steps


    '''
    description: generate the states sequence.
    '''    
    def runOnce(self, t_order=True):
        # Generate the sequence of the ego vehicle
        ego_veh_states_seq = self.stateBackwardMultipleSteps(self.ego_vehicle_)

        # Generate the sequences of the surround vehciles
        sur_veh_states_seqs = defaultdict(list)
        for id, sur_veh in self.surround_vehicles_.items():
            sur_veh_states_seqs[id] = self.stateBackwardMultipleSteps(sur_veh)

        # Correct the position from the  relative ordinate of the ego vehicle in different time stamps
        for t in range(len(ego_veh_states_seq)):
            gap = ego_veh_states_seq[t].position_.x_ - 30.0
            ego_veh_states_seq[t].position_.x_ = 30.0
            for sur_info in sur_veh_states_seqs.values():
                sur_info[t].position_.x_ += gap
        
        if not t_order:
            return ego_veh_states_seq, sur_veh_states_seqs

        # Surround vehicles states time order
        sur_veh_states_t_order = defaultdict(list)
        for t in range(len(ego_veh_states_seq)):
            for id, _ in self.surround_vehicles_.items():
                sur_veh_states_t_order[t].append((id, sur_veh_states_seqs[id][t]))
            sur_veh_states_t_order[t] = dict(sur_veh_states_t_order[t])
        
        return ego_veh_states_seq, sur_veh_states_t_order
        

if __name__ == '__main__':
    # Supply test data
    lane_info = [1, 1, 4.0, 4.0]
    random.seed(0)
    agent_generator = AgentGenerator(lane_info[0], lane_info[1], lane_info[2], lane_info[3])
    ego_vehicle = EgoInfoGenerator.generateOnce()
    surround_vehicles = agent_generator.generateAgents(1)

    image_generator = ImageGenerator(lane_info)
    states_simulator = StatesSimulator()

    # Create states sequence and draw BEV
    states_simulator.loadCurrentState(lane_info, ego_vehicle, surround_vehicles)
    ego_veh_states_seq, sur_veh_states_t_order = states_simulator.runOnce()
    for t, sur_veh_states in sur_veh_states_t_order.items():
        # # Add ego vehicle for testing
        # sur_veh_states[0] = ego_veh_states_seq[t]
        canvas = image_generator.generateSingleImage( sur_veh_states, True)
        cv2.imshow('Canvas_{}'.format(t), canvas)
        cv2.waitKey(0)






    



    


