#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Weixian Fu
"""

import math
import time
import pandas as pd


def distance_filter_helpfunc(position: list, distance_range: list = None) -> bool:
    """
    help function for func distance_filter.
    This function can be used to judge a point in the range of (x_range, y_range,z_range) or not

    :param position = [x,y,z], x,y,z position, if x ,y or z is 'NotGiven', this function will return False
    :param distance_range, the range of x, y, z example x = [[-100, math.inf],  [-math.inf, math.inf], [-math.inf, math.inf]]
    """
    if distance_range is None:
        distance_range = [[-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf]]
    if position[0] == 'NotGiven' or position[1] == 'NotGiven' or position[2] == 'NotGiven':
        dis_filter = False
    else:
        dis_filter = distance_range[0][0] <= position[0] < distance_range[0][1] and \
                     distance_range[1][0] <= position[1] < distance_range[1][1] and \
                     distance_range[2][0] <= position[2] < distance_range[2][1]
    return dis_filter


def distance_filter(df: pd.core.frame.DataFrame, distance_range: list = None) -> pd.core.frame.DataFrame:
    """
    get df filtered with distance range

    :param df: dataframe returnd by rosbag_ouput_to_dataframe func
    :param distance_range: the range of x, y, z example x = [[-100, math.inf],  [-math.inf, math.inf], [-math.inf, math.inf]]
    :return: a new dataframe with a new column named 'distance_filter' which contain a Bool value of whether this \point is in filter distance range added to df
    """

    if distance_range is None:
        distance_range = [[-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf]]
    df['distance_filter'] = df.apply(
        lambda x: distance_filter_helpfunc([x['position_x'], x['position_y'], x['position_z']], distance_range), axis=1)
    return df


def distance_filter_remove(df: pd.core.frame.DataFrame, distance_range: list = None) -> pd.core.frame.DataFrame:
    """
    get a new dataframe from df by remove all filtered('distance_filter' == False) data

    :param df: dataframe returnd by rosbag_ouput_to_dataframe func
    :param distance_range: the range of x, y, z example x = [[-100, math.inf],  [-math.inf, math.inf], [-math.inf, math.inf]]
    :return: return dataframe from df where 'distance_filter' == True
    """

    temp_df = distance_filter(df, distance_range)
    df_filter = temp_df[(temp_df['distance_filter'] == True)]
    df_filter_remove = df_filter.drop(columns=['distance_filter'])

    return df_filter_remove


if __name__ == '__main__':
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

    from data_initialization import rosbag_ouput_to_dataframe

    msg_df_pred, msg_df_pred_number = rosbag_ouput_to_dataframe("/home/ubuntu18/liangdao/Data/19-merge.bag",
                                                                topic=['/ld_object_lists'])
    msg_df_true, msg_df_true_number = rosbag_ouput_to_dataframe("/home/ubuntu18/liangdao/Data/19_ALT_OD_CP2.bag",
                                                                topic=['/ld_object_lists'])
    # distance filter
    msg_df_true_filtered = distance_filter(msg_df_true, distance_range=[[0, 30], [-10, 10], [-math.inf, math.inf]])
    msg_df_pred_filtered = distance_filter(msg_df_pred, distance_range=[[0, 30], [-10, 10], [-math.inf, math.inf]])

    end = time.clock()  # end time
    print('run time:', str(end - start))
