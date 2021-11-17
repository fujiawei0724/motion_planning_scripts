# -- coding: utf-8 --
# @Time : 2021/10/25 上午9:19
# @Author : fujiawei0724
# @File : EUDM.py
# @Software: PyCharm

"""
This code contains the EUDM behavior planner.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from EUDMCommon import *

if __name__ == '__main__':

    # Initialize lane
    center_lane_start_point = PathPoint(0.0, 0.0)
    center_lane_end_point = PathPoint(500.0, 0.0)
    center_lane = Lane(center_lane_start_point, center_lane_end_point, LaneId.CenterLane)
    center_lane_points_array = Visualization.transformPathPointsToArray(center_lane.path_points_)
    left_lane_start_point = PathPoint(0.0, 3.5)
    left_lane_end_point = PathPoint(500.0, 3.5)
    left_lane = Lane(left_lane_start_point, left_lane_end_point, LaneId.LeftLane)
    left_lane_points_array = Visualization.transformPathPointsToArray(left_lane.path_points_)
    right_lane_start_point = PathPoint(0.0, -3.5)
    right_lane_end_point = PathPoint(500.0, -3.5)
    right_lane = Lane(right_lane_start_point, right_lane_end_point, LaneId.RightLane)
    right_lane_points_array = Visualization.transformPathPointsToArray(right_lane.path_points_)

    # Initialize ego vehicle
    ego_vehicle = Vehicle(0, PathPoint(20.0, 0.0, 0.0), 5.0, 2.0, 5.0, 0.0, 0.0)
    ego_vehicle_polygon = Polygon(ego_vehicle.rectangle_.vertex_)

    # Initialize behavior generator
    behavior_generator = BehaviorGenerator(50)
    behavior_set = behavior_generator.generateBehaviors()

    # Select a behavior sequence
    # Set random seed
    random.seed(16856)
    behavior_set_length = len(behavior_set)
    behavior_sequence = behavior_set[random.randint(0, behavior_set_length - 1)]
    behavior_sequence.print()

    # Generate surround agent vehicles

    agent_generator = AgentGenerator()
    surround_vehicle_set = agent_generator.generateAgents(10)

    # Construct lane server and semantic vehicles
    all_vehicle = [ego_vehicle] + list(surround_vehicle_set.values())
    lanes = {center_lane.id_: center_lane, left_lane.id_: left_lane, right_lane.id_: right_lane}
    lane_server = LaneServer()
    lane_server.refresh(copy.deepcopy(lanes), copy.deepcopy(all_vehicle))

    # Construct forward extender
    forward_extender = ForwardExtender(lane_server, 0.4, 20.0)

    # Calculate ego trajectory and surround trajectory for each behavior
    behavior_sequence_ego_trajectory, behavior_sequence_surround_trajectories = forward_extender.multiAgentForward(behavior_sequence)

    # # For lane change left
    # lane_change_left_ego_trajectory, lane_change_left_surround_trajectories = None, None
    # if LateralBehavior.LaneChangeLeft in ego_vehicle_all_potential_behavior:
    #     forward_extender.lane_server_.refresh(copy.deepcopy(lanes), copy.deepcopy(all_vehicle))
    #     lane_change_left_ego_trajectory, lane_change_left_surround_trajectories = forward_extender.multiAgentForward(LateralBehavior.LaneChangeLeft)

    # # For lane change right
    # lane_change_right_ego_trajectory, lane_change_right_surround_trajectories = None, None
    # if LateralBehavior.LaneChangeRight in ego_vehicle_all_potential_behavior:
    #     forward_extender.lane_server_.refresh(copy.deepcopy(lanes), copy.deepcopy(all_vehicle))
    #     lane_change_right_ego_trajectory, lane_change_right_surround_trajectories = forward_extender.multiAgentForward(LateralBehavior.LaneChangeRight)

    # Calculate cost for each policy
    # TODO: add cost calculation for EUDM

    # Visualization initialization
    plt.figure(1, (12, 6))
    plt.title('Test lane, vehicle and semantic vehicle')

    # Visualization vehicle and trajectories
    for i in range(0, len(behavior_sequence_ego_trajectory.vehicle_states_)):
        # Clean visualization
        plt.cla()

        # Visualization lane
        plt.plot(center_lane_points_array[:, 0], center_lane_points_array[:, 1], c='m', linewidth=1.0)
        plt.plot(center_lane.left_boundary_points_[:, 0], center_lane.left_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
        plt.plot(center_lane.right_boundary_points_[:, 0], center_lane.right_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
        plt.plot(left_lane_points_array[:, 0], left_lane_points_array[:, 1], c='m', linewidth=1.0)
        plt.plot(left_lane.left_boundary_points_[:, 0], left_lane.left_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
        plt.plot(left_lane.right_boundary_points_[:, 0], left_lane.right_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
        plt.plot(right_lane_points_array[:, 0], right_lane_points_array[:, 1], c='m', linewidth=1.0)
        plt.plot(right_lane.left_boundary_points_[:, 0], right_lane.left_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
        plt.plot(right_lane.right_boundary_points_[:, 0], right_lane.right_boundary_points_[:, 1], c='black', ls='--', linewidth=1.0)
        if i == 0:
            # For current position
            ego_vehicle_polygon = Polygon(behavior_sequence_ego_trajectory.vehicle_states_[i].rectangle_.vertex_)
            plt.plot(*ego_vehicle_polygon.exterior.xy, c='r')
            # plt.text(ego_vehicle.position_.x_, ego_vehicle.position_.y_, 'id: {}, v: {}'.format(ego_vehicle.id_, ego_vehicle.velocity_), size=10.0)
            # Traverse surround vehicle
            for sur_veh_id, sur_veh_tra in behavior_sequence_surround_trajectories.items():
                sur_vehicle_polygon = Polygon(sur_veh_tra.vehicle_states_[i].rectangle_.vertex_)
                plt.plot(*sur_vehicle_polygon.exterior.xy, c='green')
                # plt.text(sur_veh_tra.vehicle_states_[i].position_.x_, sur_veh_tra.vehicle_states_[i].position_.y_, 'id: {}, v: {}'.format(sur_veh_id, sur_veh_tra.vehicle_states_[i].velocity_), size=10.0)

        else:
            # For predicted position
            # For current position
            ego_vehicle_polygon = Polygon(behavior_sequence_ego_trajectory.vehicle_states_[i].rectangle_.vertex_)
            plt.plot(*ego_vehicle_polygon.exterior.xy, c='r', ls='--')
            # plt.text(lane_keeping_ego_trajectory.vehicle_states_[i].position_.x_, lane_keeping_ego_trajectory.vehicle_states_[i].position_.y_, 'id: {}, v: {}, time stamp: {}'.format(ego_vehicle.id_, lane_keeping_ego_trajectory.vehicle_states_[i].velocity_, lane_keeping_ego_trajectory.vehicle_states_[i].time_stamp_), size=10.0)
            # Traverse surround vehicle
            for sur_veh_id, sur_veh_tra in behavior_sequence_surround_trajectories.items():
                sur_vehicle_polygon = Polygon(sur_veh_tra.vehicle_states_[i].rectangle_.vertex_)
                plt.plot(*sur_vehicle_polygon.exterior.xy, c='green', ls='--')
                # plt.text(sur_veh_tra.vehicle_states_[i].position_.x_, sur_veh_tra.vehicle_states_[i].position_.y_, 'id: {}, v: {}, time stamp: {}'.format(sur_veh_id, sur_veh_tra.vehicle_states_[i].velocity_, sur_veh_tra.vehicle_states_[i].time_stamp_), size=10.0)

        # Visualization each step
        plt.axis('equal')
        plt.xlim(0, 150)
        plt.pause(0.5)

    plt.show()