'''
Author: fujiawei0724
Date: 2022-04-27 17:29:13
LastEditors: fujiawei0724
LastEditTime: 2022-04-27 22:23:48
Description: Generate the image to represent state.
'''

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from utils import *


# Generate image from the state representation
class ImageGenerator:
    '''
    description: generate grid image.
    param: 
    {lane_info} is a four-tuple (left lane existing information, right lane existing information, center left distance, center right distance).
    {surround_vehicles_info} is a vector that contains a set of different surround vehicles' information.
    return:
    An image represents the state information.
    '''    
    @staticmethod
    def generateSingleImage(lane_info, surround_vehicles_info):
        # Initialize canvas 
        canvas = np.zeros((300, 300, 1), dtype='uint8')
        white = (255, 255, 255)
        
        # Supple lanes
        cv2.line(canvas, (150, 0), (150, 300), white, 2)
        if lane_info[0] == 1:
            cv2.line(canvas, (145, 0), (145, 300), white, 2)
        if lane_info[1] == 1:
            cv2.line(canvas, (155, 0), (155, 300), white, 2)

        # Supple surround vehicles
        for sur_veh in surround_vehicles_info:
            v_1, v_2 = ImageGenerator.calculateVertice(sur_veh)
            cv2.rectangle(canvas, v_1, v_2, white, -1)

        cv2.imshow('Canvas', canvas)
        cv2.waitKey(0)
    
    '''
    description: calculate the rectangle's vertice of a vehicle
    '''    
    @staticmethod
    def calculateVertice(vehicle):
        center_position = vehicle.position_
        length = vehicle.length_
        width = vehicle.width_
        v_1 = (center_position.x_ + length * 0.5 * np.cos(center_position.theta_) - width * 0.5 * np.sin(center_position.theta_), center_position.y_ + length * 0.5 * np.sin(center_position.theta_) + width * 0.5 * np.cos(center_position.theta_))
        v_2 = (center_position.x_ - length * 0.5 * np.cos(center_position.theta_) + width * 0.5 * np.sin(center_position.theta_), center_position.y_ - length * 0.5 * np.sin(center_position.theta_) - width * 0.5 * np.cos(center_position.theta_))
        return v_1, v_2





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

    ImageGenerator.generateSingleImage([1, 1, 0, 0], None)