# -- coding: utf-8 --
# @Time : 2021/10/12 下午7:35
# @Author : fujiawei0724
# @File : decisionMaking.py
# @Software: PyCharm

"""
This code includes a single simulation for behavior planner.
"""
import copy

import numpy as np
from common import *

# For test
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

    """
    Ego vehicle's desired velocity was determined by the parameter Config.desired_velocity in common.py.
    """
    ego_vehicle = Vehicle(0, PathPoint(20.0, 0.0, 0.0), 5.0, 2.0, 5.0, 0.0, 0.0)
    ego_vehicle_polygon = Polygon(ego_vehicle.rectangle_.vertex_)

    # Generate surround agent vehicles
    # Set random seed
    random.seed(190888888)
    agent_generator = AgentGenerator()
    surround_vehicle_set = agent_generator.generateAgents(5)

    # Construct lane server and semantic vehicles
    all_vehicle = [ego_vehicle] + list(surround_vehicle_set.values())
    lanes = {center_lane.id_: center_lane, left_lane.id_: left_lane, right_lane.id_: right_lane}
    lane_server = LaneServer()
    lane_server.refresh(copy.deepcopy(lanes), copy.deepcopy(all_vehicle))

    # Get ego vehicle's all potential behaviors and set ego vehicle's potential behavior randomly, limited the ego vehicle's behavior
    ego_vehicle_all_potential_behavior = lane_server.getEgoVehicleBehaviors()

    # Construct forward extender
    forward_extender = ForwardExtender(lane_server, 0.4, 4.0)

    # Calculate ego trajectory and surround trajectory for each behavior
    # For lane keeping
    lane_keeping_ego_trajectory, lane_keeping_surround_trajectories = forward_extender.multiAgentForward(LateralBehavior.LaneKeeping)

    # For lane change left
    lane_change_left_ego_trajectory, lane_change_left_surround_trajectories = None, None
    if LateralBehavior.LaneChangeLeft in ego_vehicle_all_potential_behavior:
        forward_extender.lane_server_.refresh(copy.deepcopy(lanes), copy.deepcopy(all_vehicle))
        lane_change_left_ego_trajectory, lane_change_left_surround_trajectories = forward_extender.multiAgentForward(LateralBehavior.LaneChangeLeft)

    # For lane change right
    lane_change_right_ego_trajectory, lane_change_right_surround_trajectories = None, None
    if LateralBehavior.LaneChangeRight in ego_vehicle_all_potential_behavior:
        forward_extender.lane_server_.refresh(copy.deepcopy(lanes), copy.deepcopy(all_vehicle))
        lane_change_right_ego_trajectory, lane_change_right_surround_trajectories = forward_extender.multiAgentForward(LateralBehavior.LaneChangeRight)

    # Calculate cost for each policy
    # Construct policy evaluator
    policy_evaluator = PolicyEvaluater()

    # Calculate lane keeping cost
    policy_evaluator.loadData(LateralBehavior.LaneKeeping, lane_keeping_ego_trajectory, lane_keeping_surround_trajectories)
    lane_keeping_cost = policy_evaluator.calculateCost()
    print('Lane keeping cost: {}'.format(lane_keeping_cost))

    # Calculate lane change left cost
    if LateralBehavior.LaneChangeLeft in ego_vehicle_all_potential_behavior:
        policy_evaluator.loadData(LateralBehavior.LaneChangeLeft, lane_change_left_ego_trajectory, lane_change_left_surround_trajectories)
        lane_change_left_cost = policy_evaluator.calculateCost()
        print('Lane change left cost: {}'.format(lane_change_left_cost))

    # Calculate lane change right cost
    if LateralBehavior.LaneChangeRight in ego_vehicle_all_potential_behavior:
        policy_evaluator.loadData(LateralBehavior.LaneChangeRight, lane_change_right_ego_trajectory, lane_change_right_surround_trajectories)
        lane_change_right_cost = policy_evaluator.calculateCost()
        print('Lane change right cost: {}'.format(lane_change_right_cost))







    # Visualization
    plt.figure(1, (12, 6))
    plt.title('Test lane, vehicle and semantic vehicle')

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

    # Visualization vehicle and trajectories
    for i in range(0, len(lane_keeping_ego_trajectory.vehicle_states_)):
        if i == 0:
            # For current position
            ego_vehicle_polygon = Polygon(lane_keeping_ego_trajectory.vehicle_states_[i].rectangle_.vertex_)
            plt.plot(*ego_vehicle_polygon.exterior.xy, c='r')
            plt.text(ego_vehicle.position_.x_, ego_vehicle.position_.y_, 'id: {}, v: {}'.format(ego_vehicle.id_, ego_vehicle.velocity_), size=10.0)
            # Traverse surround vehicle
            for sur_veh_id, sur_veh_tra in lane_keeping_surround_trajectories.items():
                sur_vehicle_polygon = Polygon(sur_veh_tra.vehicle_states_[i].rectangle_.vertex_)
                plt.plot(*sur_vehicle_polygon.exterior.xy, c='green')
                # plt.text(sur_veh_tra.vehicle_states_[i].position_.x_, sur_veh_tra.vehicle_states_[i].position_.y_, 'id: {}, v: {}'.format(sur_veh_id, sur_veh_tra.vehicle_states_[i].velocity_), size=10.0)

        else:
            # For predicted position
            # For current position
            ego_vehicle_polygon = Polygon(lane_keeping_ego_trajectory.vehicle_states_[i].rectangle_.vertex_)
            plt.plot(*ego_vehicle_polygon.exterior.xy, c='r', ls='--')
            plt.text(lane_keeping_ego_trajectory.vehicle_states_[i].position_.x_, lane_keeping_ego_trajectory.vehicle_states_[i].position_.y_, 'id: {}, v: {}, time stamp: {}'.format(ego_vehicle.id_, lane_keeping_ego_trajectory.vehicle_states_[i].velocity_, lane_keeping_ego_trajectory.vehicle_states_[i].time_stamp_), size=10.0)
            # Traverse surround vehicle
            for sur_veh_id, sur_veh_tra in lane_keeping_surround_trajectories.items():
                sur_vehicle_polygon = Polygon(sur_veh_tra.vehicle_states_[i].rectangle_.vertex_)
                plt.plot(*sur_vehicle_polygon.exterior.xy, c='green', ls='--')
                # plt.text(sur_veh_tra.vehicle_states_[i].position_.x_, sur_veh_tra.vehicle_states_[i].position_.y_, 'id: {}, v: {}, time stamp: {}'.format(sur_veh_id, sur_veh_tra.vehicle_states_[i].velocity_, sur_veh_tra.vehicle_states_[i].time_stamp_), size=10.0)

    plt.axis('equal')
    plt.show()



