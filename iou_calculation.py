#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Weixian Fu
"""
import numpy as np
from Geometry3D import *
import math


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def iou_result(bbox_true, bbox_pred, dimension=2):
    """
    use 2D or 3D iou calculation
    :param bbox_true:
    :param bbox_pred:
    :param dimension:
    :return:
    """
    if dimension == 2:
        return iou_result_2D(bbox_true, bbox_pred)
    else:
        return iou_result_3D(bbox_true, bbox_pred)


def iou_result_2D(bbox_true, bbox_pred):
    true_position = [bbox_true['position_x'], bbox_true['position_y']]
    _, _, yaw_pred = euler_from_quaternion(bbox_true['orientation_x'], bbox_true['orientation_y'],
                                           bbox_true['orientation_z'], bbox_true['orientation_w'])
    true_dimension = [bbox_true['dimensions_x'], bbox_true['dimensions_y']]
    pred_position = [bbox_pred['position_x'], bbox_pred['position_y'], bbox_pred['position_z']]
    _, _, yaw_true = euler_from_quaternion(bbox_pred['orientation_x'], bbox_pred['orientation_y'],
                                           bbox_pred['orientation_z'], bbox_pred['orientation_w'])
    pred_dimension = [bbox_pred['dimensions_x'], bbox_pred['dimensions_y']]

    return iou_2D(true_position, yaw_true, true_dimension, pred_position, yaw_pred, pred_dimension)


def iou_result_3D(bbox_true, bbox_pred):
    true_position = [bbox_true['position_x'], bbox_true['position_y'], bbox_true['position_z']]
    _, _, yaw_pred = euler_from_quaternion(bbox_true['orientation_x'], bbox_true['orientation_y'],
                                           bbox_true['orientation_z'], bbox_true['orientation_w'])
    true_dimension = [bbox_true['dimensions_x'], bbox_true['dimensions_y'], bbox_true['dimensions_z']]
    pred_position = [bbox_pred['position_x'], bbox_pred['position_y'], bbox_pred['position_z']]
    _, _, yaw_true = euler_from_quaternion(bbox_pred['orientation_x'], bbox_pred['orientation_y'],
                                           bbox_pred['orientation_z'], bbox_pred['orientation_w'])
    pred_dimension = [bbox_pred['dimensions_x'], bbox_pred['dimensions_y'], bbox_pred['dimensions_z']]

    return True


def iou_2D(true_position, yaw_true, true_dimension, pred_position, yaw_pred, pred_dimension, iou_threshold=0.5):
    """
    judge two rectangle is overlaped or not,
    :param iou_threshold: iou threshold, default is 0.5
    :param true_position: position of true msg, [x, y, z]
    :param yaw_true: angel ( yaw from function euler_from_quaternion(x, y, z, w) ) of true msg
    :param true_dimension: dimension of true msg (long, width, height)
    :param pred_position: position of pred msg, [x, y, z]
    :param yaw_pred: angel ( yaw from function euler_from_quaternion(x, y, z, w) ) of pred msg
    :param pred_dimension: dimension of pred msg (long, width, height)
    :return: Bool
    """
    standup_min_pred, standup_max_pred = rectangle_standup(pred_position, yaw_pred, pred_dimension)
    standup_min_true, standup_max_true = rectangle_standup(true_position, yaw_true, true_dimension)
    overlap = 0
    all_area = math.inf
    if min(standup_max_pred[0], standup_max_true[0]) > max(standup_min_pred[0], standup_min_true[0]):
        if min(standup_max_pred[1], standup_max_true[1]) > max(standup_min_pred[1], standup_min_true[1]):
            overlap = (min(standup_max_pred[0], standup_max_true[0]) - max(standup_min_pred[0], standup_min_true[0])) * \
                      (min(standup_max_pred[1], standup_max_true[1]) - max(standup_min_pred[1], standup_min_true[1]))
            all_area = (standup_max_pred[0] - standup_min_pred[0]) * (standup_max_pred[1] - standup_min_pred[1]) + \
                       (standup_max_true[0] - standup_min_true[0]) * (standup_max_true[1] - standup_min_true[1]) - \
                       overlap
    return overlap / all_area > iou_threshold


def rectangle_func(position, yaw, dimension):
    """
    return four vertex of a rectangle
    :param position: [x, y]
    :param yaw: angel
    :param dimension: [long, width]
    :return: vertex, 4*2
    """
    vertex = np.zeros((4, 2))
    vertex[0] = [position[0] + dimension[0] / 2 * math.cos(yaw) + dimension[1] / 2 * math.sin(yaw),
                 position[1] + dimension[1] / 2 * math.sin(yaw) - dimension[0] / 2 * math.cos(yaw)]
    vertex[1] = [position[0] + dimension[0] / 2 * math.cos(yaw) - dimension[1] / 2 * math.sin(yaw),
                 position[1] + dimension[1] / 2 * math.sin(yaw) + dimension[0] / 2 * math.cos(yaw)]
    vertex[2] = [position[0] - dimension[0] / 2 * math.cos(yaw) - dimension[1] / 2 * math.sin(yaw),
                 position[1] - dimension[0] / 2 * math.sin(yaw) + dimension[1] / 2 * math.cos(yaw)]
    vertex[3] = [position[0] - dimension[0] / 2 * math.cos(yaw) + dimension[1] / 2 * math.sin(yaw),
                 position[1] - dimension[0] / 2 * math.sin(yaw) - dimension[1] / 2 * math.cos(yaw)]
    return vertex


def rectangle_standup(position, yaw, dimension):
    """
    help function of iou calculation and bbox, it will return a list, [[min(x), min(y)], [max[x], max[y]]]
    return two standup
    :param position: [x, y]
    :param yaw: angel
    :param dimension: [long, width]
    :return: two point 2*2
    """
    vertex = rectangle_func(position, yaw, dimension) # location of 4 vertex
    standup_min = np.min(vertex, axis=0)
    standup_max = np.max(vertex, axis=0)
    return standup_min, standup_max


def Parallelepiped_from_df(position, dimension, orientation):
    """
    This function have not yet implemented, it is supposed to generate a Geometry Parallelepiped by given bbox information.
    :param position: [x,y,z] the center of a Parallelepiped, the position information of rosbag
    :param dimension: [long, width, height]
    :param orientation: [x, y, z, w]
    :return: return a Parallelepiped
    """

    return


def Parallelepiped_intersection(Parall1, Parall2, iou_threshold):
    '''
    given to Parallelepiped and iou threshold, calculate the intersection and iou.
    compare the iou with threshold, return a value which iou is greater than the threshold

    @Parall1: Parallelepiped, can be generative from Geometry3D.Parallelepiped
    @Parall2: Parallelepiped, can be generative from Geometry3D.Parallelepiped
    @iou_threshold: threshold
    '''
    intersection_things = intersection(Parall1, Parall2)
    iou = intersection_things.volume() / (Parall1.volume() + Parall2.volume() - intersection_things.volume())
    return iou >= iou_threshold


if __name__ == '__main__':
    cph = Parallelepiped(Point(0, 0, 0), Vector(2, 0, 0), Vector(0, 2, 0), Vector(0, 0, 2))
