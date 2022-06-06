'''
Author: fujiawei0724
Date: 2022-05-30 16:54:57
LastEditors: fujiawei0724
LastEditTime: 2022-06-06 18:43:09
Description: Components for MCTS.
'''

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from rl_behavior_planner.utils import *
from rl_behavior_planner.environment import Environment


class SubForwardExtender(ForwardExtender):

    def __init__(self, lane_server, dt):
        self.lane_server_ = copy.deepcopy(lane_server)
        self.dt_ = dt
        
    # Forward one step
    def multiAgentForwardOnce(self, vehicle_intention, vehicles, lane_speed_limit):

        # Get the initial velocities of all vehicles
        initial_velocities = dict()
        for veh_id, veh in vehicles.items():
            initial_velocities[veh_id] = veh.velocity_

        # Get initial semantic vehicles
        cur_semantic_vehicles = self.lane_server_.getSemanticVehicles(vehicles)

        # Initialize cache
        states_cache = {}

        for veh_id, veh in vehicles.items():

            # print('Predicting vehicle id: {}'.format(veh_id))

            # Determine initial vehicles information
            desired_velocity = initial_velocities[veh_id]
            # init_time_stamp = veh.vehicle_.time_stamp_
            if veh_id == 0:
                desired_velocity += vehicle_intention.velocity_compensation_
                desired_velocity = np.clip(desired_velocity, 0.0, lane_speed_limit)

            desired_veh_state = self.forwardOnce(veh_id, vehicle_intention.lat_beh_, cur_semantic_vehicles, desired_velocity)

            # Cache
            states_cache[desired_veh_state.id_] = desired_veh_state
        
        # Parse data
        next_ego_vehicle = states_cache[0]
        states_cache.pop(0)
        next_sur_vehicles = states_cache
        
        return next_ego_vehicle, next_sur_vehicles


# Evaluate single state generate from subenvironment
# TODO: if the reward in mcts could only be gotten from the terminal of each episode, this evaluator will be useless
class StateEvaluator:
    pass


class SubEnvironment(Environment):

    def runOnce(self):
        pass
    
    def simulateSingleStep(self, vehicle_intention, ax=None):
        # Initialize state cache
        # TODO: check this logic, try to avoid the use of copy
        vehicles = copy.deepcopy(self.surround_vehicle_)
        vehicles[0] = copy.deepcopy(self.ego_vehicle_)

        # # DEBUG
        # if ax != None:
        #     self.visualization(ax)
        # # END DEBUG


        # Calculate next states for each vehicle
        # TODO: check this parameters about prediction time scan
        sub_forward_extender = SubForwardExtender(self.lane_server_, 0.4)
        n_ego_vehicle, n_sur_vehicles = sub_forward_extender.multiAgentForwardOnce(vehicle_intention, 
                                                                      vehicles, 
                                                                      self.lane_speed_limit_)

        # # DEBUG
        # self.ego_vehicle_ = n_ego_vehicle
        # self.surround_vehicle_ = n_sur_vehicles
        # if ax != None:
        #     self.visualization(ax)
        # # END DEBUG

        return n_ego_vehicle, n_sur_vehicles
        


        

        


if __name__ == '__main__':
    # Set random seed
    random.seed(3)

    # Construct environment 
    env = SubEnvironment()

    # Construct training environment
    left_lane_exist = random.randint(0, 1)
    right_lane_exist = random.randint(0, 1)
    center_left_distance = random.uniform(3.0, 4.5)
    center_right_distance = random.uniform(3.0, 4.5)
    lane_info = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance]
    lane_speed_limit = random.uniform(10.0, 25.0)
    lane_info_with_speed = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit]

    # Construct ego vehicle and surround vehicles randomly
    ego_vehicle = EgoInfoGenerator.generateOnce()
    ego_vehicle.curvature_, ego_vehicle.steer_ = 0.0, 0.0
    ego_vehicle.velocity_ = 2.0
    ego_vehicle.print()
    surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
    surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))

    # Load data to environment
    env.load(lane_info_with_speed, ego_vehicle, surround_vehicles)

    # Construct behavior/intention
    vehicle_intention = VehicleIntention(LateralBehavior.LaneKeeping, -5.0)

    # Calculate next state
    fig = plt.figure(0)
    ax = plt.axes()
    n_ego_vehicle, n_surround_vehicles = env.simulateSingleStep(vehicle_intention, ax)
    ax.axis('equal')
    n_ego_vehicle.print()

    plt.show()


    


