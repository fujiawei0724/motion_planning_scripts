'''
Author: fujiawei0724
Date: 2022-06-07 11:01:56
LastEditors: fujiawei0724
LastEditTime: 2022-06-16 21:24:52
Description: mcts algorithm.
'''

import sys
sys.path.append('..')
import random
import numpy as np
from collections import defaultdict
from subEnvironment import SubEnvironment, State
from rl_behavior_planner.utils import *

VEHICLE_INTENTION_SET = [VehicleIntention(lat_beh, lon_vel) 
                         for lat_beh in LateralBehavior 
                         for lon_vel in np.arange(-5.0, 5.0 + 1e-3, 1.0)]


# A group of states in time order
class MacroState:
    moves_num = 11 * 3
    def __init__(self, states=None, lane_change_num=None, lane_info_with_speed=None, intention=None):
        self.states_ = states
        self.lane_change_num_ = lane_change_num
        self.lane_info_with_speed_ = lane_info_with_speed
        
        # Record behavior information
        self.intention_ = intention 

    def reward(self):
        # TODO: add consideration of nonexistent lanes
        cost, is_collision, _, _, _ = MctsPolicyEvaluator.praise(self)
        if is_collision:
            return -1.0
        return 1.0 / cost
    
    def terminal(self):
        if self.states_[-1].terminal():
            return True
        return False

    # Generate random next state to construct the default policy
    def next_state(self, env, cur_intention):
        # Load data to environment and generate next state
        env.loadState(self.lane_info_with_speed_, self.states_[-1])
        next_state = env.simulateSingleStep(cur_intention)

        return next_state
    
    # Generate next macro state
    def next_macro_state(self, env):
        # Select intention randomly
        cur_intention = random.choice(VEHICLE_INTENTION_SET)
        
        # Calculate next state
        next_state = self.next_state(env, cur_intention)

        # Integrate ego macro state and the generated next state
        # TODO: check the logic about the copy of the current object (try to avoid the use of 'copy')
        next_macro_state = MacroState()
        next_macro_state.states_ = self.states_
        next_macro_state.states_.append(next_state)
        next_macro_state.lane_change_num_ = self.lane_change_num_
        next_macro_state.intention_ = cur_intention
        if cur_intention.lat_beh_ == LateralBehavior.LaneChangeLeft or cur_intention.lat_beh_ == LateralBehavior.LaneChangeRight:
            next_macro_state.lane_change_num_ += 1
        next_macro_state.lane_info_with_speed_ = self.lane_info_with_speed_

        return next_macro_state

        

# Node in the search tree
class Node:
    def __init__(self, macro_state, parent=None):
        self.visit_num_ = 1
        self.reward_ = 0.0
        self.macro_state_ = macro_state
        self.children_ = []
        self.parent_ = parent
    
    def add_child(self, child_macro_state):
        child = Node(child_macro_state, self)
        self.children_.append(child)
    
    def update(self, reward):
        self.reward_ += reward
        self.visit_num_ += 1
    
    def fully_expanded(self):
        # TODO: add domain knowledge here to limit the scale of the search tree
        if len(self.children_) == self.macro_state_.moves_num:
            return True
        return False
    
    def best_policy(self):
        best_score = -np.inf
        best_children = []
        for c in self.children_:
            score = c.reward_ / c.visit_num_
            if score == best_score:
                best_children.append(c)
            if score > best_score:
                best_children = [c]
                best_score = score
        if len(best_children) == 0:
            print('Fatal error!!!')
        return random.choice(best_children)
    
# Generate reward for the states sequence
class MctsPolicyEvaluator(PolicyEvaluator):
    
    @classmethod
    def calculateMultiLaneChangeCost(cls, change_num):
        return change_num * 0.3

    @classmethod
    def praise(cls, states_sequence):
        # Reconstruct data 
        ego_states = []
        sur_states = defaultdict(list)
        for st in states_sequence.states_:
            ego_states.append(st.ego_vehicle_)
            for sur_id, sur_veh in st.surround_vehicles_.items():
                sur_states[sur_id].append(sur_veh)
        ego_traj = Trajectory(ego_states)
        sur_trajs = dict()
        for s_id, s_states in sur_states.items():
            sur_trajs[s_id] = Trajectory(s_states)
        
        # Calculate cost
        safety_cost, is_collision = cls.calculateSafetyCost(ego_traj, sur_trajs, states_sequence.lane_speed_limit)
        lane_change_cost = cls.calculateMultiLaneChangeCost(states_sequence.lane_change_num_)
        efficiency_cost = cls.calculateEfficiencyCost(ego_traj, states_sequence.lane_speed_limit_)
        comfort_cost = cls.calculateComfortCost(ego_traj)
        print('Safety cost: {}'.format(safety_cost))
        print('Lane change cost: {}'.format(lane_change_cost))
        print('Efficiency cost: {}'.format(efficiency_cost))
        print('Comfort cost: {}'.format(comfort_cost))
        print('All cost: {}'.format(safety_cost + lane_change_cost + efficiency_cost + comfort_cost))
        return safety_cost + lane_change_cost + efficiency_cost + comfort_cost, is_collision, safety_cost, lane_change_cost, efficiency_cost
        
# Training the tree policy
class TreePolicyTrainer:
    def __init__(self, round_limit, time_limit, scalar):
        self.round_limit_ = round_limit
        self.time_limit_ = time_limit
        self.scalar_ = scalar
    
    '''
    description: train the tree policy
    param {root} root node of the search tree
    return {*}
    '''    
    def train(self, root, env):
        for iter in range(self.round_limit_):
            front = self.tree_policy(root, env)
            reward = self.default_policy(front.macro_state_, env)
            self.backup(front, reward)
        return root
    
    '''
    description: stretch a tree node
    param {node} start node, also the node manipulated 
    return {node} the ternimal node in the branch of the start node
    '''    
    def tree_policy(self, node, env):
        while node.macro_state_.terminal() == False:
            if len(node.children_) == 0:
                return self.expand(node, env)
            elif random.uniform(0, 1) < 0.5:
                node = self.best_child(node)
            else:
                if node.fully_expanded() == False:
                    return self.expand(node)
                else:
                    node = self.best_child(node)
        return node
    
    def expand(self, node, env):
        tried_children = [c.macro_state_ for c in node.children_]
        next_macro_state = node.macro_state_.next_macro_state(env)
        while next_macro_state in tried_children and next_macro_state.terminal() == False:
            next_macro_state = node.macro_state_.next_macro_state(env)
        node.add_child(next_macro_state)
        return node.children_[-1]

    def best_child(self, node):
        best_score = -np.inf
        best_children = []
        for c in node.children_:
            exploit = c.reward_ / c.visit_num_
            explore = np.sqrt(2.0 * np.log(node.visit_num_) / float(c.visit_num_))
            score = exploit + self.scalar_ * explore
            if score == best_score:
                best_children.append(c)
            if score > best_score:
                best_children = [c]
                best_score = score
        if len(best_children) == 0:
            print('Fatal error!!!')
        return random.choice(best_children)

    def default_policy(self, macro_state, env):
        while macro_state.terminal() == False:
            macro_state = macro_state.next_macro_state(env)
        return macro_state.reward()

    def backup(self, node, reward):
        while node != None:
            node.visit_num_ += 1
            node.reward_ += reward
            node = node.parent_
        return 
        
    

if __name__ == '__main__':
    # Load environment data randomly
    random.seed(0)
    left_lane_exist = random.randint(0, 1)
    right_lane_exist = random.randint(0, 1)
    center_left_distance = random.uniform(3.0, 4.5)
    center_right_distance = random.uniform(3.0, 4.5)
    lane_info = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance]
    lane_speed_limit = random.uniform(10.0, 25.0)
    lane_info_with_speed = [left_lane_exist, right_lane_exist, center_left_distance, center_right_distance, lane_speed_limit]

    # Construct ego vehicle and surround vehicles randomly
    ego_vehicle = EgoInfoGenerator.generateOnce()
    surround_vehicles_generator = AgentGenerator(left_lane_exist, right_lane_exist, center_left_distance, center_right_distance)
    surround_vehicles = surround_vehicles_generator.generateAgents(random.randint(0, 10))

    # Construct environment
    env = SubEnvironment()

    # Construct trainer
    scalar = 1.0 / (2.0 * np.sqrt(2.0))
    mcts_trainer = TreePolicyTrainer(20, None, scalar)

    # Initialize start node 
    root = Node(MacroState([State(ego_vehicle, surround_vehicles, 0.0)], 0, lane_info_with_speed))
    mcts_trainer.train(root, env)

    
    