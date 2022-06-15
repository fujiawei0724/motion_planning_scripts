'''
Author: fujiawei0724
Date: 2022-06-07 11:01:56
LastEditors: fujiawei0724
LastEditTime: 2022-06-15 22:30:16
Description: mcts algorithm.
'''

import sys
sys.path.append('..')
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
    def __init__(self, states=None, lane_change_num=None, lane_info_with_speed=None):
        self.states_ = states
        self.lane_change_num_ = lane_change_num
        self.lane_info_with_speed_ = lane_info_with_speed

    def reward(self):
        # TODO: add consideration of nonexistent lanes
        cost, is_collision, _, _, _ = MctsPolicyEvaluator.praise(self)
        if is_collision:
            return -1.0
        return 1.0 / cost
    
    def terminal(self):
        if self.states_[-1].ternimal():
            return True
        return False

    # Generate random next state to construct the default policy
    def next_state(self, env, cur_intention):
        # Load data to environment and generate next state
        env.load(self.lane_info_with_speed_, self.states_[-1])
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
        if cur_intention.lat_beh_ == LateralBehavior.LaneChangeLeft or cur_intention.lat_beh_ == LateralBehavior.LaneChangeRight:
            next_macro_state.lane_change_num_ += 1
        next_macro_state.lane_info_with_speed_ = self.lane_info_with_speed_

        return next_macro_state

        

# Node in the search tree
class Node:
    def __init__(self, state_sequence, parent=None):
        self.visit_num_ = 1
        self.reward_ = 0.0
        self.state_sequence_ = state_sequence
        self.children_ = []
        self.parent_ = parent
    
    def add_child(self, child_states_sequence, child_lane_change_num, child_lane_info_with_speed):
        child = Node(MacroState(child_states_sequence, child_lane_change_num, child_lane_info_with_speed), self)
        self.children_.append(child)
    
    def update(self, reward):
        self.reward_ += reward
        self.visit_num_ += 1
    
    def fully_expanded(self):
        # TODO: add domain knowledge here to limit the scale of the search tree
        if len(self.children_) == self.state_sequence_.moves_num:
            return True
        return False
    
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
    def train(self, root):
        for iter in range(self.round_limit_):
            pass
    
    '''
    description: stretch a tree node
    param {node} start node, also the node manipulated 
    return {node} the ternimal node in the branch of the start node
    '''    
    def tree_policy(self, node):
        while node.state_sequence_.terminal() == False:
            if len(node.children_) == 0:
                # TODO: expand node 
                pass
            elif random.uniform(0, 1) < 0.5:
                pass
            else:
                if node.fully_expanded() == False:
                    pass
                else:
                    pass
        return node
    
    def expand(self, node):
        pass

    def best_child(self, node):
        best_score = 0.0
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
    print(len(VEHICLE_INTENTION_SET))