'''
Author: fujiawei0724
Date: 2022-07-09 15:32:45
LastEditors: fujiawei0724
LastEditTime: 2022-07-10 16:01:50
Description: Calculate the point in the quintic spline with the minimum distance to a specific point.
'''

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

Point = namedtuple('Point', ['x_', 'y_', 'theta_', 'kappa_'])

class QuinticSpline:
    def __init__(self, begin_state, end_state):
        # Calculate mileage
        l = np.sqrt((begin_state.x_ - end_state.x_) * (begin_state.x_ - end_state.x_) + (begin_state.y_ - end_state.y_) * (begin_state.y_ - end_state.y_))
        self.l = l

        p0x = begin_state.x_
        p0y = begin_state.y_
        t0x = np.cos(begin_state.theta_)
        t0y = np.sin(begin_state.theta_)
        k0x = -begin_state.kappa_*np.sin(begin_state.theta_)
        k0y = begin_state.kappa_*np.cos(begin_state.theta_)

        # Finish parameter initialization
        p1x = end_state.x_
        p1y = end_state.y_
        t1x = np.cos(end_state.theta_)
        t1y = np.sin(end_state.theta_)
        k1x = -end_state.kappa_*np.sin(end_state.theta_)
        k1y = end_state.kappa_*np.cos(end_state.theta_)

        # Generate parameters
        self.a0 = +(                   p0x )
        self.a1 = (+(                   t0x ) * l) / l
        self.a2 = (+(                   k0x ) * l * l / 2.0) / l ** 2.0
        self.a3 = (+( + 10 * p1x - 10 * p0x )\
                +( -  4 * t1x -  6 * t0x ) * l\
                +( +      k1x -  3 * k0x ) * l * l / 2.0) / l ** 3.0
        self.a4 = (+( - 15 * p1x + 15 * p0x )\
                +( +  7 * t1x +  8 * t0x ) * l\
                +( -  2 * k1x +  3 * k0x ) * l * l / 2.0) / l ** 4.0
        self.a5 = (+( +  6 * p1x -  6 * p0x )\
                +( -  3 * t1x -  3 * t0x ) * l\
                +( +      k1x -      k0x ) * l * l / 2.0) / l ** 5.0
                
        self.b0 = +(                   p0y )
        self.b1 = (+(                   t0y ) * l) / l
        self.b2 = (+(                   k0y ) * l * l / 2.0) / l ** 2.0
        self.b3 = (+( + 10 * p1y - 10 * p0y )\
                +( -  4 * t1y -  6 * t0y ) * l\
                +( +      k1y -  3 * k0y ) * l * l / 2.0) / l ** 3.0
        self.b4 = (+( - 15 * p1y + 15 * p0y )\
                +( +  7 * t1y +  8 * t0y ) * l\
                +( -  2 * k1y +  3 * k0y ) * l * l / 2.0) / l ** 4.0
        self.b5 = (+( +  6 * p1y -  6 * p0y )\
                +( -  3 * t1y -  3 * t0y ) * l\
                +( +      k1y -      k0y ) * l * l / 2.0) / l ** 5.0
    
    def generate_path(self, point_num=100):
        xs = []
        ys = []
        for i in range(0, 100):
            param = i / point_num * self.l
            x = self.a0 + self.a1 * param + self.a2 * param ** 2 + self.a3 * param ** 3 + self.a4 * param ** 4 + self.a5 * param ** 5
            y = self.b0 + self.b1 * param + self.b2 * param ** 2 + self.b3 * param ** 3 + self.b4 * param ** 4 + self.b5 * param ** 5
            xs.append(x)
            ys.append(y)
        return xs, ys
        
    
    def calculate_nearest_point(self, x0, y0):
        s = self.l * 0.5
        for _ in range(10):
            f_s = -self.a0*self.a1 - self.b0*self.b1 + (-3*self.a3**2 - 6*self.a2*self.a4 - 6*self.a1*self.a5 - 3*self.b2**2 - 6*self.b2*self.b3 - 3*self.b3**2 - 6*self.b1*self.b5)*s**5 + (-7*self.a3*self.a4 - 7*self.a2*self.a5 - 7*self.b2*self.b4 - 7*self.b3*self.b4)*s**6 + (-4*self.a4**2 - 8*self.a3*self.a5 - 4*self.b4**2 - 8*self.b2*self.b5 - 8*self.b3*self.b5)*s**7 + (-9*self.a4*self.a5 - 9*self.b4*self.b5)*s**8 + (-5*self.a5**2 - 5*self.b5**2)*s**9 + self.a1*x0 + s*(-self.a1**2 - 2*self.a0*self.a2 - self.b1*2 + 2*self.a2*x0) + self.b1*y0 + s**2*(-3*self.a1*self.a2 - 3*self.a0*self.a3 - 3*self.b0*self.b2 - 3*self.b0*self.b3 + 3*self.a3*x0 + 3*self.b2*y0 + 3*self.b3*y0) + s**3*(-2*self.a2**2 - 4*self.a1*self.a3 - 4*self.a0*self.a4 - 4*self.b1*self.b2 - 4*self.b1*self.b3 - 4*self.b0*self.b4 + 4*self.a4*x0 + 4*self.b4*y0) + s**4*(-5*self.a2*self.a3 - 5*self.a1*self.a4 - 5*self.a0*self.a5 - 5*self.b1*self.b4 - 5*self.b0*self.b5 + 5*self.a5*x0 + 5*self.b5*y0)
            f_d_s = -self.a1**2 - 2*self.a0*self.a2 - self.b1**2 + (-15*self.a3**2 - 30*self.a2*self.a4 - 30*self.a1*self.a5 - 15*self.b2**2 - 30*self.b2*self.b3 - 15*self.b3**2 - 30*self.b1*self.b5)*s**4 + (-42*self.a3*self.a4 - 42*self.a2*self.a5 - 42*self.b2*self.b4 - 42*self.b3*self.b4)*s**5 + (-28*self.a4**2 - 56*self.a3*self.a5 - 28*self.b4**2 - 56*self.b2*self.b5 - 56*self.b3*self.b5)*s**6 + (-72*self.a4*self.a5 - 72*self.b4*self.b5)*s**7 + (-45*self.a5**2 - 45*self.b5**2)*s**8 + 2*self.a2*x0 + s*(-6*self.a1*self.a2 - 6*self.a0*self.a3 - 6*self.b0*self.b2 - 6*self.b0*self.b3 + 6*self.a3*x0 + 6*self.b2*y0 + 6*self.b3*y0) + s**2*(-6*self.a2**2 - 12*self.a1*self.a3 - 12*self.a0*self.a4 - 12*self.b1*self.b2 - 12*self.b1*self.b3 - 12*self.b0*self.b4 + 12*self.a4*x0 + 12*self.b4*y0) + s**3*(-20*self.a2*self.a3 - 20*self.a1*self.a4 - 20*self.a0*self.a5 - 20*self.b1*self.b4 - 20*self.b0*self.b5 + 20*self.a5*x0 + 20*self.b5*y0)
            s = s - f_s / f_d_s
            s = np.clip(s, 0.0, self.l)
        print('s: {}'.format(s))
        target_x = self.a0 + self.a1 * s + self.a2 * s ** 2 + self.a3 * s ** 3 + self.a4 * s ** 4 + self.a5 * s ** 5
        target_y = self.b0 + self.b1 * s + self.b2 * s ** 2 + self.b3 * s ** 3 + self.b4 * s ** 4 + self.b5 * s ** 5
        return target_x, target_y

class PiecewiseQuinticSpline:
    def __init__(self, points_list, points_gap=None):
        if points_gap != None:
            points_gap.insert(0, 0)
            self.points_gap = np.cumsum(points_gap)
        self.points_list = points_list
        self.n = len(points_list)
        assert self.n >= 3
        
        self.x = []
        self.y = []
        for point in points_list:
            self.x.append(point.x_)
            self.y.append(point.y_)


    def calculate_nearest_scatter_point_index(self, x0, y0):
        def dis(x_s, y_s, x_e, y_e):
            return np.sqrt((x_e - x_s) ** 2 + (y_e - y_s) ** 2)

        left, right = 0, len(self.points_list) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if dis(self.points_list[mid].x_, self.points_list[mid].y_, x0, y0) >= dis(self.points_list[mid - 1].x_, self.points_list[mid - 1].y_, x0, y0):
                right = mid - 1
            else:
                left = mid + 1
        print('Left: {}, right: {}'.format(left, right))
        return right
    
    def calculate_nearest_point(self, x0, y0):
        # Get nearest scatter point index
        target_index = self.calculate_nearest_scatter_point_index(x0, y0)

        # Get the nearest point for different situations
        if target_index == 0:
            q_spline = QuinticSpline(self.points_list[target_index], self.points_list[target_index + 1])
            xs, ys = q_spline.generate_path()
            return q_spline.calculate_nearest_point(x0, y0), xs, ys
        elif target_index == self.n - 1:
            q_spline = QuinticSpline(self.points_list[target_index - 1], self.points_list[target_index])
            xs, ys = q_spline.generate_path()
            return q_spline.calculate_nearest_point(x0, y0), xs, ys
        else:
            q_spline_pre = QuinticSpline(self.points_list[target_index - 1], self.points_list[target_index])
            q_spline_next = QuinticSpline(self.points_list[target_index], self.points_list[target_index + 1])
            x_pre, y_pre = q_spline_pre.calculate_nearest_point(x0, y0)
            x_next, y_next = q_spline_next.calculate_nearest_point(x0, y0)
            xs_pre, ys_pre = q_spline_pre.generate_path()
            xs_next, ys_next = q_spline_next.generate_path()
            xs, ys = xs_pre + xs_next, ys_pre + ys_next
            
            if np.sqrt((x_pre - x0) ** 2 + (y_pre - y0) ** 2) <= np.sqrt((x_next - x0) ** 2 + (y_next - y0) ** 2):
                return x_pre, y_pre, xs, ys
            else:
                return x_next, y_next, xs, ys



def test_quintic_spline():
    # Set start point and end point
    start_point = Point(0.0, 0.0, 0.0, 0.0)
    end_point = Point(1.0, 1.0, 1.0, 0.0)
    quintic_spline = QuinticSpline(start_point, end_point)

    # Generate path
    xs, ys = quintic_spline.generate_path()

    # Set target point
    query_x, query_y = 1.0, 0.8
    min_x, min_y = quintic_spline.calculate_nearest_point(query_x, query_y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, c='r', linewidth=1.0)
    type_0 = ax.scatter([min_x], [min_y], c='g', s=50.0)
    type_1 = ax.scatter([query_x], [query_y], c='b', s=50.0)
    ax.legend((type_0, type_1), ('Nearest point', 'Query point'), loc=0)
    plt.axis('equal')
    plt.show()

def test_piecewise_quintic_spline():
    # Set some points randomly
    point_0 = Point(0.0, 0.0, 0.0, 0.0)
    point_1 = Point(1.0, 1.0, 1.0, 0.0)
    point_2 = Point(1.5, 2.0, 1.5, 0.5)
    point_3 = Point(3.0, 3.0, 0.0, 0.0)
    point_4 = Point(4.0, 5.0, 0.5, 0.5)

    # Generate piecewise quintic spline
    piecewise_quintic_spline = PiecewiseQuinticSpline([point_0, point_1, point_2, point_3, point_4])

    # Set target point
    query_x, query_y = 4.0, 2.8

    # Calculate nearest point
    min_x, min_y, path_segment_x, path_segment_y = piecewise_quintic_spline.calculate_nearest_point(query_x, query_y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(path_segment_x, path_segment_y, c='r', linewidth=1.0)
    ax.scatter(piecewise_quintic_spline.x, piecewise_quintic_spline.y, c='r', s=1.0)
    type_0 = ax.scatter([min_x], [min_y], c='g', s=50.0)
    type_1 = ax.scatter([query_x], [query_y], c='b', s=50.0)
    ax.legend((type_0, type_1), ('Nearest point', 'Query point'), loc=0)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    # test_quintic_spline()
    test_piecewise_quintic_spline()

