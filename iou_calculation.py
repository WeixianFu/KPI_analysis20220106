#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Weixian Fu
"""
import copy

import numpy as np
from Geometry3D import *
import math
import pandas as pd


def euler_from_quaternion(x: float, y: float, z: float, w: float) -> (float, float, float):
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

# old version of iou calculation
# def iou_result(bbox_true: pd.core.frame.DataFrame, bbox_pred: pd.core.frame.DataFrame,
#                dimension: int = 2, iou_threshold: float = 0.5) -> bool:
#     """
#     the old version function, should not be used anymore
#     implement an 2D or 3D iou calculation
#
#     :param iou_threshold: threshold of iou matching
#     :param bbox_true: dataframe of true bbox, the shape of this dataframe is (25.0)
#     :param bbox_pred: dataframe of pred bbox, the shape of this dataframe is (25.0)
#     :param dimension: 3D of 2D matching
#     :return: bool
#     """
#     assert bbox_pred.shape == (25,) and bbox_true.shape == (25,)
#     if dimension == 2:
#         return iou_result_2D(bbox_true, bbox_pred, iou_threshold)
#     else:
#         return iou_result_3D(bbox_true, bbox_pred, iou_threshold)
#
#
# def iou_result_2D(bbox_true: pd.core.frame.DataFrame, bbox_pred: pd.core.frame.DataFrame,
#                   iou_threshold: float = 0.5) -> bool:
#     """
#     the old version function, should not be used anymore
#     2D iou matching function, should be called by function iou_result(bbox_true, bbox_pred, dimension,s threshold)
#
#     :param iou_threshold: threshold of iou matching
#     :param bbox_true: dataframe of true bbox, the shape of this dataframe is (25.0)
#     :param bbox_pred: dataframe of pred bbox, the shape of this dataframe is (25.0)
#     :return: bool
#     """
#     true_position = [bbox_true['position_x'], bbox_true['position_y']]
#     _, _, yaw_pred = euler_from_quaternion(bbox_true['orientation_x'], bbox_true['orientation_y'],
#                                            bbox_true['orientation_z'], bbox_true['orientation_w'])
#     true_dimension = [bbox_true['dimensions_x'], bbox_true['dimensions_y']]
#     pred_position = [bbox_pred['position_x'], bbox_pred['position_y'], bbox_pred['position_z']]
#     _, _, yaw_true = euler_from_quaternion(bbox_pred['orientation_x'], bbox_pred['orientation_y'],
#                                            bbox_pred['orientation_z'], bbox_pred['orientation_w'])
#     pred_dimension = [bbox_pred['dimensions_x'], bbox_pred['dimensions_y']]
#
#     return iou_2D(true_position, yaw_true, true_dimension, pred_position, yaw_pred, pred_dimension, iou_threshold)
#
#
# def iou_result_3D(bbox_true: pd.core.frame.DataFrame, bbox_pred: pd.core.frame.DataFrame,
#                   iou_threshold: float = 0.5) -> bool:
#     """
#     the old version function, should not be used anymore
#     3D iou matching function, should be called by function iou_result(bbox_true, bbox_pred, dimension,s threshold)
#
#     :param bbox_true: dataframe of true bbox, the shape of this dataframe is (25.0)
#     :param bbox_pred: dataframe of pred bbox, the shape of this dataframe is (25.0)
#     :return: bool
#     """
#     true_position = [bbox_true['position_x'], bbox_true['position_y'], bbox_true['position_z']]
#     _, _, yaw_pred = euler_from_quaternion(bbox_true['orientation_x'], bbox_true['orientation_y'],
#                                            bbox_true['orientation_z'], bbox_true['orientation_w'])
#     true_dimension = [bbox_true['dimensions_x'], bbox_true['dimensions_y'], bbox_true['dimensions_z']]
#     pred_position = [bbox_pred['position_x'], bbox_pred['position_y'], bbox_pred['position_z']]
#     _, _, yaw_true = euler_from_quaternion(bbox_pred['orientation_x'], bbox_pred['orientation_y'],
#                                            bbox_pred['orientation_z'], bbox_pred['orientation_w'])
#     pred_dimension = [bbox_pred['dimensions_x'], bbox_pred['dimensions_y'], bbox_pred['dimensions_z']]
#
#     return iou_3D(true_position, yaw_true, true_dimension, pred_position, yaw_pred, pred_dimension, iou_threshold)
#
#
# def iou_2D(true_position: list, yaw_true: float, true_dimension: list, pred_position: list,
#            yaw_pred: float, pred_dimension: list, iou_threshold: float = 0.5) -> bool:
#     """
#     the old version function, should not be used anymore
#     judge the overlaped area of two rectangle is greater than iou threshold or not
#
#     :param iou_threshold: iou threshold, default is 0.5
#     :param true_position: position of true msg, [x, y, z]
#     :param yaw_true: angel ( yaw from function euler_from_quaternion(x, y, z, w) ) of true msg
#     :param true_dimension: dimension of true msg (long, width, height)
#     :param pred_position: position of pred msg, [x, y, z]
#     :param yaw_pred: angel ( yaw from function euler_from_quaternion(x, y, z, w) ) of pred msg
#     :param pred_dimension: dimension of pred msg (long, width, height)
#     :return: Bool
#     """
#     standup_min_pred, standup_max_pred = rectangle_standup(pred_position, yaw_pred, pred_dimension)
#     standup_min_true, standup_max_true = rectangle_standup(true_position, yaw_true, true_dimension)
#     overlap = 0
#     all_area = math.inf
#     if min(standup_max_pred[0], standup_max_true[0]) > max(standup_min_pred[0], standup_min_true[0]):
#         if min(standup_max_pred[1], standup_max_true[1]) > max(standup_min_pred[1], standup_min_true[1]):
#             overlap = (min(standup_max_pred[0], standup_max_true[0]) - max(standup_min_pred[0], standup_min_true[0])) * \
#                       (min(standup_max_pred[1], standup_max_true[1]) - max(standup_min_pred[1], standup_min_true[1]))
#             all_area = (standup_max_pred[0] - standup_min_pred[0]) * (standup_max_pred[1] - standup_min_pred[1]) + \
#                        (standup_max_true[0] - standup_min_true[0]) * (standup_max_true[1] - standup_min_true[1]) - \
#                        overlap
#     return overlap / all_area > iou_threshold
#
#
# def iou_3D(true_position: list, yaw_true: float, true_dimension: list, pred_position: list,
#            yaw_pred: float, pred_dimension: list, iou_threshold: float = 0.5) -> bool:
#     """
#     the old version function, should not be used anymore
#     judge the overlaped volumn of two cuboid is greater than iou threshold or not
#
#     :param iou_threshold: iou threshold, default is 0.5
#     :param true_position: position of true msg, [x, y, z]
#     :param yaw_true: angel ( yaw from function euler_from_quaternion(x, y, z, w) ) of true msg
#     :param true_dimension: dimension of true msg (long, width, height)
#     :param pred_position: position of pred msg, [x, y, z]
#     :param yaw_pred: angel ( yaw from function euler_from_quaternion(x, y, z, w) ) of pred msg
#     :param pred_dimension: dimension of pred msg (long, width, height)
#     :return: Bool
#     """
#     standup_min_pred, standup_max_pred = cuboid_standup(pred_position, yaw_pred, pred_dimension)
#     standup_min_true, standup_max_true = cuboid_standup(true_position, yaw_true, true_dimension)
#     overlap = 0
#     all_volume = math.inf
#     if min(standup_max_pred[0], standup_max_true[0]) > max(standup_min_pred[0], standup_min_true[0]):
#         if min(standup_max_pred[1], standup_max_true[1]) > max(standup_min_pred[1], standup_min_true[1]):
#             if min(standup_max_pred[2], standup_max_true[2]) > max(standup_min_pred[2], standup_min_true[2]):
#                 overlap = (min(standup_max_pred[0], standup_max_true[0]) - max(standup_min_pred[0],
#                                                                                standup_min_true[0])) * \
#                           (min(standup_max_pred[1], standup_max_true[1]) - max(standup_min_pred[1],
#                                                                                standup_min_true[1])) * \
#                           (min(standup_max_pred[2], standup_max_true[2]) - max(standup_min_pred[2],
#                                                                                standup_min_true[2]))
#                 all_volume = (standup_max_pred[0] - standup_min_pred[0]) * (standup_max_pred[1] - standup_min_pred[1]) \
#                              * (standup_max_pred[2] - standup_min_pred[2]) + \
#                              (standup_max_true[0] - standup_min_true[0]) * (standup_max_true[1] - standup_min_true[1]) \
#                              * (standup_max_true[2] - standup_min_true[2]) - overlap
#     return overlap / all_volume > iou_threshold
#
#
# def rectangle_func(position: list, yaw: float, dimension: list) -> np.ndarray:
#     """
#     the old version function, should not be used anymore
#     return four vertex of a rectangle
#
#     :param position: [x, y]
#     :param yaw: angel
#     :param dimension: [long, width]
#     :return: vertex, 4*2
#     """
#     vertex = np.zeros((4, 2))
#     vertex[0] = [position[0] + dimension[0] / 2 * math.cos(yaw) + dimension[1] / 2 * math.sin(yaw),
#                  position[1] + dimension[1] / 2 * math.sin(yaw) - dimension[0] / 2 * math.cos(yaw)]
#     vertex[1] = [position[0] + dimension[0] / 2 * math.cos(yaw) - dimension[1] / 2 * math.sin(yaw),
#                  position[1] + dimension[1] / 2 * math.sin(yaw) + dimension[0] / 2 * math.cos(yaw)]
#     vertex[2] = [position[0] - dimension[0] / 2 * math.cos(yaw) - dimension[1] / 2 * math.sin(yaw),
#                  position[1] - dimension[0] / 2 * math.sin(yaw) + dimension[1] / 2 * math.cos(yaw)]
#     vertex[3] = [position[0] - dimension[0] / 2 * math.cos(yaw) + dimension[1] / 2 * math.sin(yaw),
#                  position[1] - dimension[0] / 2 * math.sin(yaw) - dimension[1] / 2 * math.cos(yaw)]
#     return vertex
#
#
# def rectangle_standup(position: list, yaw: float, dimension: list) -> (np.ndarray, np.ndarray):
#     """
#     the old version function, should not be used anymore
#     help function of iou calculation and bbox, it will return a list, [[min(x), min(y)], [max[x], max[y]]]
#
#     :param position: [x, y]
#     :param yaw: angel
#     :param dimension: [long, width]
#     :return: two point 2*2
#     """
#     vertex = rectangle_func(position, yaw, dimension)  # location of 4 vertex
#     standup_min = np.min(vertex, axis=0)
#     standup_max = np.max(vertex, axis=0)
#     return standup_min, standup_max
#
#
# def cuboid_standup(position: list, yaw: float, dimension: list) -> (np.ndarray, np.ndarray):
#
#     """
#     the old version function, should not be used anymore
#     help function of iou calculation and bbox, it will return a list, [[min(x), min(y), min(z)], [max[x], max[y], max(z)]]
#
#     :param position: [x, y]
#     :param yaw: angel
#     :param dimension: [long, width]
#     :return: two point 2*2
#     """
#     standup_min, standup_max = rectangle_standup(position, yaw, dimension)
#     standup_min = np.append(standup_min, [position[2] - dimension[2]])
#     standup_max = np.append(standup_max, [position[2] + dimension[2]])
#     return standup_min, standup_max


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


def df_to_box_array_2D(df: pd.core.frame.DataFrame) -> np.ndarray:
    """
    return an array of 2D bounding box

    :param df: the input dataframe, should have the same class_label and in the same frame(same msg_number)
    :return: an array of bounding box
    """
    bbox_list = []
    for index, df_object in df.iterrows():
        position = [df_object['position_x'], df_object['position_y']]
        _, _, yaw = euler_from_quaternion(df_object['orientation_x'], df_object['orientation_y'],
                                          df_object['orientation_z'], df_object['orientation_w'])
        dimension = [df_object['dimensions_x'], df_object['dimensions_y']]
        vertex = np.zeros((4, 2))
        vertex[0] = [position[0] + dimension[0] / 2 * math.cos(yaw) + dimension[1] / 2 * math.sin(yaw),
                     position[1] + dimension[1] / 2 * math.sin(yaw) - dimension[0] / 2 * math.cos(yaw)]
        vertex[1] = [position[0] + dimension[0] / 2 * math.cos(yaw) - dimension[1] / 2 * math.sin(yaw),
                     position[1] + dimension[1] / 2 * math.sin(yaw) + dimension[0] / 2 * math.cos(yaw)]
        vertex[2] = [position[0] - dimension[0] / 2 * math.cos(yaw) - dimension[1] / 2 * math.sin(yaw),
                     position[1] - dimension[0] / 2 * math.sin(yaw) + dimension[1] / 2 * math.cos(yaw)]
        vertex[3] = [position[0] - dimension[0] / 2 * math.cos(yaw) + dimension[1] / 2 * math.sin(yaw),
                     position[1] - dimension[0] / 2 * math.sin(yaw) - dimension[1] / 2 * math.cos(yaw)]
        standup_min = np.min(vertex, axis=0)
        standup_max = np.max(vertex, axis=0)
        bbox_list.append([standup_min[0], standup_min[1], standup_max[0], standup_max[1]])
    bbox_array = np.array(bbox_list)

    return bbox_array


def df_to_box_array_3D(df: pd.core.frame.DataFrame) -> np.ndarray:
    """
    return an array of 3D bounding box

    :param df: the input dataframe, should have the same class_label and in the same frame(same msg_number)
    :return: an array of bounding box
    """
    bbox_list = []
    for index, df_object in df.iterrows():
        position = [df_object['position_x'], df_object['position_y'], df_object['position_z']]
        _, _, yaw = euler_from_quaternion(df_object['orientation_x'], df_object['orientation_y'],
                                          df_object['orientation_z'], df_object['orientation_w'])
        dimension = [df_object['dimensions_x'], df_object['dimensions_y'], df_object['dimensions_z']]
        vertex = np.zeros((6, 2))
        vertex[0] = [position[0] + dimension[0] / 2 * math.cos(yaw) + dimension[1] / 2 * math.sin(yaw),
                     position[1] + dimension[1] / 2 * math.sin(yaw) - dimension[0] / 2 * math.cos(yaw)]
        vertex[1] = [position[0] + dimension[0] / 2 * math.cos(yaw) - dimension[1] / 2 * math.sin(yaw),
                     position[1] + dimension[1] / 2 * math.sin(yaw) + dimension[0] / 2 * math.cos(yaw)]
        vertex[2] = [position[0] - dimension[0] / 2 * math.cos(yaw) - dimension[1] / 2 * math.sin(yaw),
                     position[1] - dimension[0] / 2 * math.sin(yaw) + dimension[1] / 2 * math.cos(yaw)]
        vertex[3] = [position[0] - dimension[0] / 2 * math.cos(yaw) + dimension[1] / 2 * math.sin(yaw),
                     position[1] - dimension[0] / 2 * math.sin(yaw) - dimension[1] / 2 * math.cos(yaw)]
        standup_min = np.min(vertex, axis=0)
        standup_max = np.max(vertex, axis=0)
        standup_min = np.append(standup_min, [position[2] - dimension[2]])
        standup_max = np.append(standup_max, [position[2] + dimension[2]])
        bbox_list.append([standup_min[0], standup_min[1], standup_min[2],
                          standup_max[0], standup_max[1], standup_max[2]])
    bbox_array = np.array(bbox_list)

    return bbox_array


def box_iou_2D(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    return the intersection over union matrix of anchor bounding box array and ground truth bounding box array

    :param boxes1: array anchor bounding box, size: N*4
    :param boxes2: array ground truth bounding box, size: K*4
    :return: IOU matrix of anchor and ground truth, size: N*K
    """

    # function of calculate bbox area
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # calculate the intersecion area
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # claculate union area
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def box_iou_3D(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    return the intersection over union matrix of anchor bounding box array and ground truth bounding box array

    :param boxes1: array anchor bounding box, size: N*6
    :param boxes2: array ground truth bounding box, size: K*6
    :return: IOU matrix of anchor and ground truth, size: N*K
    """
    # volume function
    box_volume = lambda boxes: ((boxes[:, 3] - boxes[:, 0]) *
                                (boxes[:, 4] - boxes[:, 1]) *
                                (boxes[:, 5] - boxes[:, 2]))
    volumes1 = box_volume(boxes1)
    volumes2 = box_volume(boxes2)
    # intersection volume function
    inter_upperlefts = np.maximum(boxes1[:, None, :3], boxes2[:, :3])
    inter_lowerrights = np.minimum(boxes1[:, None, 3:], boxes2[:, 3:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    inter_volumes = inters[:, :, 0] * inters[:, :, 1] * inters[:, :, 2]
    # union volume function
    union_volumes = volumes1[:, None] + volumes2 - inter_volumes
    return inter_volumes / union_volumes


def iou_cv(true_frame_class_df: pd.core.frame.DataFrame, pred_frame_class_df: pd.core.frame.DataFrame,
           dimensions: int = 2, threshold: float = 0.5) -> list:
    """
    matching whth iou, the matching rule is:
    1. find the ground truth bbox k and anchor bbox n with the max iou, and iou>threshold
    2. remove other iou whith ground truth k and anchor bbox n
    3. return to step 1, finish until all ground truth or anchor is matched, or all iou remained is lower than threshold

    :param true_frame_class_df: ground truth dataframe
    :param pred_frame_class_df: anchor dataframe
    :param dimensions: 2D or 3D
    :param threshold: iou threshold
    :return: an 3*1 array, [a, b, c] , a is the number of  matched pair of ground truth and anchor, b is the number of
    unmatched ground truth, c is the number of unmatched anchor.
    anchor
    """
    if dimensions == 2:
        return iou_new_2D(true_frame_class_df, pred_frame_class_df, threshold=threshold)
    elif dimensions == 3:
        return iou_new_3D(true_frame_class_df, pred_frame_class_df, threshold=threshold)


def iou_new_2D(true_frame_class_df: pd.core.frame.DataFrame, pred_frame_class_df: pd.core.frame.DataFrame,
               threshold: int = 0.5) -> list:
    """
    2D help function of iou_cv

    :param true_frame_class_df: ground truth dataframe
    :param pred_frame_class_df: anchor dataframe
    :param threshold: iou threshold
    :return: an 3*1 array, [a, b, c] , a is the number of  matched pair of ground truth and anchor, b is the number of
    unmatched ground truth, c is the number of unmatched anchor.
    anchor
    """
    bbox_pred = df_to_box_array_2D(pred_frame_class_df)
    bbox_true = df_to_box_array_2D(true_frame_class_df)
    iou_matrix = box_iou_2D(bbox_true, bbox_pred)
    iou_matrix_copy = copy.deepcopy(iou_matrix)
    match_counter = 0
    while iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
        if np.max(iou_matrix) < threshold:
            break
        else:
            index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            match_counter += 1
            iou_matrix = np.delete(iou_matrix, index[0], 0)
            iou_matrix = np.delete(iou_matrix, index[1], 1)
    return [match_counter, max(0, len(iou_matrix_copy) - match_counter),
            max(0, len(iou_matrix_copy[0]) - match_counter)]


def iou_new_3D(true_frame_class_df: pd.core.frame.DataFrame, pred_frame_class_df: pd.core.frame.DataFrame,
               threshold: int = 0.5) -> list:
    """
    3D help function of iou_cv

    :param true_frame_class_df: ground truth dataframe
    :param pred_frame_class_df: anchor dataframe
    :param threshold: iou threshold
    :return: an 3*1 array, [a, b, c] , a is the number of  matched pair of ground truth and anchor, b is the number of
    unmatched ground truth, c is the number of unmatched anchor.
    anchor
    """
    bbox_pred = df_to_box_array_3D(pred_frame_class_df)
    bbox_true = df_to_box_array_3D(true_frame_class_df)
    iou_matrix = box_iou_3D(bbox_true, bbox_pred)
    iou_matrix_copy = copy.deepcopy(iou_matrix)
    match_counter = 0
    while iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
        if np.max(iou_matrix) < threshold:
            break
        else:
            index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            match_counter += 1
            iou_matrix = np.delete(iou_matrix, index[0], 0)
            iou_matrix = np.delete(iou_matrix, index[1], 1)
    return [match_counter, max(0, len(iou_matrix_copy) - match_counter),
            max(0, len(iou_matrix_copy[0]) - match_counter)]


if __name__ == '__main__':
    cph = Parallelepiped(Point(0, 0, 0), Vector(2, 0, 0), Vector(0, 2, 0), Vector(0, 0, 2))

    box1 = [[1, 2, 4, 4], [2, 6, 8, 10], [5, 5, 7, 6], [3, 2, 6, 3]]
    box1 = np.array(box1)
    print(box1.shape)
    box2 = [[1, 2, 3, 4], [2, 5, 9, 13]]
    box2 = np.array(box2)
    c = box_iou_2D(box1, box2)
    print(c)
    print(np.max(c))
    print(np.unravel_index(np.argmax(c), c.shape))
