#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Weixian Fu
"""

import numpy as np
from iou_calculation import iou_result
import pandas as pd


def data_matching(df_true: pd.core.frame.DataFrame, df_pred: pd.core.frame.DataFrame,
                  dimension: int = 2, iou_threshold: float = 0.5) -> (list, list):
    """
    data matching function
    :param iou_threshold: threshold of iou matching
    :param dimension: 1 or 2, (2D or 3D), default = 2
    :param df_true: dataframe of true data
    :param df_pred: dataframe of pred data
    :return: two list, pred_list = ['Car', 'Car', ..., ], pred_list = ['Car', 'Van', 'UnKonwn', ... ,]
    """
    pred_list = []
    true_list = []

    for num in set(df_pred['msg_number']):
        pred_msg_frame = df_pred[(df_pred['msg_number'] == num)]
        true_msg_frame = df_true[(df_true['msg_number'] == num)]
        true_class = list(true_msg_frame['class_label_true'])
        true_class_set = set(true_class)
        dict_true = np.zeros(true_msg_frame.shape[0])

        for index, msg_bbox_pred in pred_msg_frame.iterrows():
            pred_matched_or_not = False
            if msg_bbox_pred['class_label_pred'] not in true_class_set:
                pred_list.append(msg_bbox_pred['class_label_pred'])
                true_list.append('UnKnown')
                pred_matched_or_not = True
            else:
                bbox_counter = -1
                for index_true, msg_bbox_true in true_msg_frame.iterrows():
                    bbox_counter += 1
                    if iou_result(msg_bbox_true, msg_bbox_pred, dimension, iou_threshold):
                        pred_list.append(msg_bbox_pred['class_label_pred'])
                        true_list.append(msg_bbox_true['class_label_true'])
                        dict_true[bbox_counter] = 1
                        pred_matched_or_not = True
                        break
            if not pred_matched_or_not:
                pred_list.append(msg_bbox_pred['class_label_pred'])
                true_list.append('UnKnown')

        for num1 in range(len(dict_true)):
            if dict_true[num1] == 0:
                true_list.append(true_class[num1])
                pred_list.append('UnKnown')

    for num in (set(df_true['msg_number']) - set(df_pred['msg_number'])):
        for index, msg_bbox_true in df_true[(df_true['msg_number'] == num)].iterrows():
            true_list.append(msg_bbox_true['class_label_pred'])
            pred_list.append('UnKnown')

    return true_list, pred_list


# def msg_object_matching(df_true, pred_object, dimension=2):
#     """
#     help function for data_matching function, given a pred_object and a dataframe of all true object in the same frame
#     :param df_true:
#     :param pred_object:
#     :return:
#     """
#     vehicle = pred_object["class_label_pred"] # a str, example: 'Car'
#     temp_pred_list = []
#     temp_true_list = []
#     dict_list = np.zeros(df_true.shape[0])
#     pred_object_position = [pred_object['position_x'], pred_object['position_y'], pred_object['position_z']]
#     pred_object_orientation = [pred_object['orientation_x'], pred_object['orientation_y'],
#                                pred_object['orientation_z'], pred_object['orientation_w']]
#     true_object_counter = 0
#     for index, msg_object_true in df_true.iterrows():
#         if msg_object_true["class_label_pred"] != vehicle:
#             temp_pred_list.append(msg_object_true["class_label_pred"])
#             temp_true_list.append('UnKnown')
#         else:
#             object_position = [msg_object_true['position_x'], msg_object_true['position_y'],
#                                msg_object_true['position_z']]
#             object_orientation = [msg_object_true['orientation_x'], msg_object_true['orientation_y'],
#                                   msg_object_true['orientation_z'], msg_object_true['orientation_w']]
#             if iou_result(pred_object_position, pred_object_orientation, object_position, object_orientation,
#                           dimension):
#                 temp_pred_list.append(vehicle)
#                 temp_true_list.append(vehicle)
#                 dict_list[]


if __name__ == '__main__':
    import time
    import math
    from data_initialization import rosbag_ouput_to_dataframe
    from distance_filter import distance_filter_remove

    start = time.clock()  # start time
    list_args = [
        ["/home/ubuntu18/liangdao/Data/19-merge.bag", "/home/ubuntu18/liangdao/Data/19_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/20-merge.bag", "/home/ubuntu18/liangdao/Data/20_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/21-merge.bag", "/home/ubuntu18/liangdao/Data/21_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/40-merge.bag", "/home/ubuntu18liangdao/Data/40_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/42-merge.bag", "/home/ubuntu18/liangdao/Data/42_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/48-merge.bag", "/home/ubuntu18/liangdao/Data/48_ALT_OD_CP2.bag"]
    ]
    # obtain data from rosbag document

    msg_df_pred, msg_df_pred_number = rosbag_ouput_to_dataframe("/home/ubuntu18/liangdao/Data/19-merge.bag",
                                                                topic=['/ld_object_lists'])
    msg_df_true, msg_df_true_number = rosbag_ouput_to_dataframe("/home/ubuntu18/liangdao/Data/19_ALT_OD_CP2.bag",
                                                                topic=['/ld_object_lists'])
    # distance filter
    distance_range = [[0, 30], [-10, 10], [-math.inf, math.inf]]
    # msg_df_true_filtered = distance_filter(msg_df_true, distance_range)
    # msg_df_pred_filtered = distance_filter(msg_df_pred, distance_range)

    # distance filter remove
    msg_df_true_filtered_removed = distance_filter_remove(msg_df_true, distance_range)
    msg_df_pred_filtered_removed = distance_filter_remove(msg_df_pred, distance_range)

    end = time.clock()  # end time
    print('run time:', str(end - start))
