'''
Author: fujiawei0724
Date: 2022-04-27 17:29:13
LastEditors: fujiawei0724
LastEditTime: 2022-04-28 22:23:47
Description: Generate the image to represent state.
'''

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from utils import *

black = (0, 0, 0)   
white = (255, 255, 255)

# Generate image from the state representation
class ImageGenerator:
    '''
    description: initialize the size of the image.
    '''    
    def __init__(self, size=(300, 60, 1), scale=3):
        self.image_size_ = size
        self.scale_ = scale

    '''
    description: generate grid image.
    param: 
    {lane_info} is a four-tuple (left lane existing information, right lane existing information, center left distance, center right distance).
    {surround_vehicles_info} is a vector that contains a set of different surround vehicles' information.
    return:
    An image represents the state information.
    '''    
    def generateSingleImage(self, lane_info, surround_vehicles_info):
        # Initialize canvas 
        canvas = np.zeros(self.image_size_, dtype='uint8')

        # Supple lanes
        cv2.line(canvas, (round(self.image_size_[1] / 2), 0), (round(self.image_size_[1] / 2), self.image_size_[0]), white, 2)
        if lane_info[0] == 1:
            cv2.line(canvas, (round(self.image_size_[1] / 2 + lane_info[2]), 0), (round(self.image_size_[1] / 2 + lane_info[2]), self.image_size_[0]), white, 2)
        if lane_info[1] == 1:
            cv2.line(canvas, (round(self.image_size_[1] / 2 - lane_info[3]), 0), (round(self.image_size_[1] / 2 - lane_info[3]), self.image_size_[0]), white, 2)

        # Supple surround vehicles
        for sur_veh in surround_vehicles_info.values():
            v_1, v_2 = ImageGenerator.calculateVertice(sur_veh)
            print('Frenet x: {}, y: {}'.format(sur_veh.position_.x_, sur_veh.position_.y_))
            # print('Image x: {}, y: {}'.format(self.positionTransform(v_1), self.positionTransform(v_2)))
            cv2.rectangle(canvas, self.positionTransform(v_1), self.positionTransform(v_2), black, -1)

        cv2.imshow('Canvas', canvas)
        cv2.waitKey(0)
    
    '''
    description: calculate the rectangle's vertice of a vehicle.
    '''
    @staticmethod 
    def calculateVertice(vehicle):
        center_position = vehicle.position_
        length = vehicle.length_
        width = vehicle.width_
        v_1 = (round(center_position.x_ + length * 0.5 * np.cos(center_position.theta_) - width * 0.5 * np.sin(center_position.theta_)), round(center_position.y_ + length * 0.5 * np.sin(center_position.theta_) + width * 0.5 * np.cos(center_position.theta_)))
        v_2 = (round(center_position.x_ - length * 0.5 * np.cos(center_position.theta_) + width * 0.5 * np.sin(center_position.theta_)), round(center_position.y_ - length * 0.5 * np.sin(center_position.theta_) - width * 0.5 * np.cos(center_position.theta_)))
        return v_1, v_2
    
    '''
    description: transform the position from frenet to image. 

    '''    
    def positionTransform(self, frenet_pos):
        # return (self.image_size_[0] - frenet_pos[0], round(self.image_size_[1] / 2) + frenet_pos[1])
        return (round(self.image_size_[1] / 2) + frenet_pos[1], self.image_size_[0] - frenet_pos[0])

        






if __name__ == '__main__':
    # canvas = np.zeros((300, 300, 1), dtype='uint8')
    # white = (255, 255, 255)
    # cv2.line(canvas, (0, 0), (300, 300), white)
    # cv2.imshow('Canvas', canvas)
    # cv2.waitKey(0)

    # cv2.line(canvas, (300, 0), (0, 300), white, 3)
    # cv2.imshow('Canvas', canvas)
    # cv2.waitKey(0)

    # cv2.rectangle(canvas, (10, 10), (60, 60), white)
    # cv2.imshow('Canvas', canvas)
    # cv2.waitKey(0)

    # cv2.rectangle(canvas, (50, 200), (200, 225), white, 5)
    # cv2.imshow('Canvas', canvas)
    # cv2.waitKey(0)

    # cv2.rectangle(canvas, (200, 50), (225, 125), white, -1)
    # cv2.imshow('Canvas', canvas)
    # cv2.waitKey(0)

    # random.seed(0)
    lane_info = [1, 1, 10, 10]
    agent_generator = AgentGenerator(lane_info[0], lane_info[1], lane_info[2], lane_info[3])
    surround_vehicles = agent_generator.generateAgents(1)
    image_generator = ImageGenerator()
    image_generator.generateSingleImage(lane_info, surround_vehicles)