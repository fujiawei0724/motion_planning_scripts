'''
Author: fujiawei0724
Date: 2022-05-30 16:54:57
LastEditors: fujiawei0724
LastEditTime: 2022-05-31 22:43:14
Description: Monte Carlo Tree Search algorithm.
'''

import sys
sys.path.append('..')
import numpy as np
from rl_behavior_planner.utils import *
from rl_behavior_planner.environment import Environment


class SubForwardExtender(ForwardExtender):

    # Forward one step
    def multiAgentForwardOnce(self, vehicle_intention, vehicles, lane_speed_limit):
        
        ego_vehicle_id = 0

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
            if veh_id == ego_vehicle_id:
                desired_velocity += vehicle_intention.velocity_compensation_
                desired_velocity = np.clip(desired_velocity, 0.0, lane_speed_limit)

            desired_veh_state = self.forwardOnce(veh_id, vehicle_intention.lat_beh_, cur_semantic_vehicles, desired_velocity)

            # Cache
            states_cache[desired_veh_state.id_] = desired_veh_state
        
        # Parse data


class SubEnvironment(Environment):

    def runOnce(self):
        pass
    
    def simulateSingleStep(self, vehicle_intention):
        pass

        

        


if __name__ == '__main__':
    sub_environment = SubEnvironment()
    


