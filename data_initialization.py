#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Weixian Fu
"""

import rosbag
import pandas as pd
import os
import time


def rosbag_ouput_to_dataframe(path: str, mode: str = 'r', topic: str = None) -> (pd.core.frame.DataFrame, int):
    """
    this function can be used to return a dataframe from rosbag document, this data frame contain 25 columns:
    ['timestamp_std', 'msg_number', 'msg_item_id', 'class_label_true', 'class_label_pred',
    'position_x','position_y','position_z',
    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
    'tracking_points.x', 'tracking_points.y', 'tracking_points.z',
    'jsk_position_x', 'jsk_position_y', 'jsk_position_z',
    'jsk_orientation_x', 'jsk_orientation_y', 'jsk_orientation_z', 'jsk_orientation_w',
    'dimensions_x', 'dimensions_y', 'dimensions_z']
    if the msg from .bag dont have any data, use 'NotGiven' as default

    :param path: path of .bag file, example: "/home/weixianf/liangdao/Data/40-merge.bag"
    :param mode: either 'r', 'w', or 'a',default = 'r'
    :param topic: list of topics or a single topic. if an empty list is given all topics will be read. default = []
    """

    if topic is None:
        topic = []
    msg_columns = ['timestamp_std', 'msg_number', 'msg_item_id', 'class_label_true', 'class_label_pred',
                   'position_x', 'position_y', 'position_z',
                   'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                   'tracking_points.x', 'tracking_points.y', 'tracking_points.z',
                   'jsk_position_x', 'jsk_position_y', 'jsk_position_z',
                   'jsk_orientation_x', 'jsk_orientation_y', 'jsk_orientation_z', 'jsk_orientation_w',
                   'dimensions_x', 'dimensions_y', 'dimensions_z'
                   ]

    msg_df_data = []  # save all msg item to a dataframe named msg_df
    msg_count = 0  # counter the number of msg
    with rosbag.Bag(path, mode) as temp_bag:
        for topic, msg, t in temp_bag.read_messages(topic):
            timestamp_std = t.to_sec()  # convert t(rospy time format) to timestamp (floating point, example:
            # 1635926387.065)
            if not msg.objects:  # if msg have no object, use 'NotGiven' and -1 as default
                msg_item_df = [timestamp_std, msg_count] + [-1] + ['NotGiven'] * 22
                msg_df_data.append(msg_item_df)
            else:
                for msg_item in msg.objects:
                    msg_item_df = [timestamp_std, msg_count, msg_item.id, msg_item.class_label_true,
                                   msg_item.class_label_pred,
                                   msg_item.pose.position.x, msg_item.pose.position.y, msg_item.pose.position.z,
                                   msg_item.pose.orientation.x, msg_item.pose.orientation.y,
                                   msg_item.pose.orientation.z, msg_item.pose.orientation.w,
                                   msg_item.tracking_points.x, msg_item.tracking_points.y, msg_item.tracking_points.z,
                                   msg_item.jsk_pose.position.x, msg_item.jsk_pose.position.y,
                                   msg_item.jsk_pose.position.z,
                                   msg_item.jsk_pose.orientation.x, msg_item.jsk_pose.orientation.y,
                                   msg_item.jsk_pose.orientation.z, msg_item.jsk_pose.orientation.w,
                                   msg_item.dimensions.x, msg_item.dimensions.y, msg_item.dimensions.z
                                   ]
                    msg_df_data.append(msg_item_df)
            msg_count = msg_count + 1
        msg_df = pd.DataFrame(msg_df_data, columns=msg_columns)
    return msg_df, msg_count


def rosbag_save_to_csv(bag_path: str, bag_topic: str = None, csv_path: str = 'default', csv_name: str = 'default') -> None:
    '''
    save bag data to csv, use rosbag_ouput_to_dataframe as help function
    @bag_path: rosbag document path, example: example: "/home/weixianf/liangdao/Data/40-merge.bag"
    csv_path: end with /
    csv_name:name of csv document
    '''

    if bag_topic is None:
        bag_topic = []
    if csv_path == 'default':
        backslash_distance = len(bag_path) - 1
        for i in range(len(bag_path) - 1, -1, -1):
            if bag_path[i] == '/':
                backslash_distance = i
                break
        csv_path = bag_path[0:backslash_distance + 1] + 'csv/'
    elif not csv_path.endswith('/'):
        csv_path += '/'

    if csv_name == 'default':
        backslash_distance = len(bag_path) - 1
        for i in range(len(bag_path) - 1, -1, -1):
            if bag_path[i] == '/':
                backslash_distance = i
                break
        csv_name = bag_path[backslash_distance + 1:len(bag_path) - 4]
    elif csv_name.endswith('.csv'):
        csv_name = csv_name[0:len(csv_name) - 4]

    results_dir = csv_path + csv_name + '.csv'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    print("Writing robot joint state data to CSV")

    temp_df = rosbag_ouput_to_dataframe(bag_path, topic=bag_topic)
    temp_df.to_csv(results_dir)


def unique_df_label_helpfunc(df: pd.core.frame.DataFrame, pred_or_true: str) -> str:
    """
    help function for unique_df_for_msg_number(df, pred_or_true), return the most frequent class label
    :param df: dataframe from rosbag
    :param pred_or_true: 'class_label_true', 'class_label_pred'
    """
    dict1 = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0, 'Van': 0, 'Truck': 0, 'Multi': 0}
    vote = 0
    vote_label = 'NotGiven'
    for index, item in df.iterrows():
        dict1[item[pred_or_true]] = dict1[item[pred_or_true]] + 1
        if dict1[item[pred_or_true]] > vote:
            vote = dict1[item[pred_or_true]]
            vote_label = item[pred_or_true]
    return vote_label


def unique_df_for_msg_number(df: pd.core.frame.DataFrame, pred_or_true: str) -> pd.core.frame.DataFrame:
    """
    return the unique datdframe information for every msg

    :param df: dataframe from rosbag
    :param pred_or_true: 'class_label_true', 'class_label_pred'
    """
    total_msg_number_list = list(set(df['msg_number']))
    total_temp_list = []
    for i in total_msg_number_list:
        temp_msg_df = df[(df['msg_number'] == i)]
        temp_means_list = temp_msg_df.mean().tolist()
        temp_means_list = temp_means_list[0:3] + [unique_df_label_helpfunc(temp_msg_df, pred_or_true)] + \
                          [unique_df_label_helpfunc(temp_msg_df, pred_or_true)] + temp_means_list[3:23]
        total_temp_list.append(temp_means_list)
        # print(i)
    # print(total_temp_list)

    msg_columns = ['timestamp_std', 'msg_number', 'msg_item_id', 'class_label_true', 'class_label_pred',
                   'position_x', 'position_y', 'position_z',
                   'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                   'tracking_points.x', 'tracking_points.y', 'tracking_points.z',
                   'jsk_position_x', 'jsk_position_y', 'jsk_position_z',
                   'jsk_orientation_x', 'jsk_orientation_y', 'jsk_orientation_z', 'jsk_orientation_w',
                   'dimensions_x', 'dimensions_y', 'dimensions_z'
                   ]

    total_df = pd.DataFrame(total_temp_list, columns=msg_columns)

    return total_df


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
    # msg_df_pred, msg_df_pred_number = rosbag_ouput_to_dataframe("/home/ubuntu18/liangdao/Data/21-merge.bag",
    #                                                             topic=['/ld_object_lists'])
    # msg_df_true, msg_df_true_number = rosbag_ouput_to_dataframe("/home/ubuntu18/liangdao/Data/21_ALT_OD_CP2.bag",
    #                                                             topic=['/ld_object_lists'])
    # # distance filter
    # msg_df_true_filtered = distance_filter(msg_df_true, distance_range=[[0, 30], [-10, 10], [-math.inf, math.inf]])
    # msg_df_pred_filtered = distance_filter(msg_df_pred, distance_range=[[0, 30], [-10, 10], [-math.inf, math.inf]])

    end = time.clock()  # end time
    print('run time:', str(end - start))

    predc = 0
    predmc = 0
    truemc = 0
    truec = 0
    for i in list_args:
        msg_df_pred, msg_df_pred_number = rosbag_ouput_to_dataframe(i[1], topic=['/ld_object_lists'])
        msg_df_true, msg_df_true_number = rosbag_ouput_to_dataframe(i[0], topic=['/ld_object_lists'])
        break
        predc += msg_df_pred[(msg_df_pred['msg_item_id'] == -1)].shape[0]
        predmc += msg_df_pred_number
        truec += msg_df_true[(msg_df_true['msg_item_id'] == -1)].shape[0]
        # truec += msg_df_true['msg_number'].shape[0]
        truemc += msg_df_true_number
    a = 0
    b = 0
    c = 0
    print(msg_df_true.shape[0])
    aa = msg_df_pred[(msg_df_pred['msg_number'])==1]
    print(aa.shape)
    for index, msg in msg_df_true.iterrows():
        if msg['class_label_pred']!=msg['class_label_true']:
            a+=1
            print(msg['class_label_pred'], msg['class_label_true'])
        elif msg['class_label_pred'] == msg['class_label_true']:
            b+=1
        c+=1
    print(a, b, c)


    # msg_df_true_filtered = distance_filter(msg_df_true, range=[[0, 30], [-10, 10], [-math.inf, math.inf]])
    # msg_df_pred_filtered = distance_filter(msg_df_pred, range=[[0, 30], [-10, 10], [-math.inf, math.inf]])
    # print('----------------1------------------')
    # print(len(list(set(msg_df_true_filtered[(msg_df_true_filtered['distance_filter']==True)]['msg_item_id']))))
    # print(len(list(set(msg_df_pred_filtered[(msg_df_pred_filtered['distance_filter']==True)]['msg_number']))))

    # msg_df_pred_f_unique = unique_for_msg_number()

    # print(set(msg_df_pred_filtered['msg_number']) == set(msg_df_true_filtered['msg_number']))

    # temp = msg_df_true_filtered[(msg_df_true_filtered['distance_filter']==True)]
    # print(temp.shape)
    # print(msg_df_true_filtered.shape)
    #
    # print(len(set(msg_df_true_filtered['msg_number'])))
    # a = set(msg_df_true_filtered['msg_number'])
    # print(type(a))
    # print(len(msg_df_true_filtered.mean()))

    # print(set(msg_df_pred_filtered['class_label_pred']))

    # print(max(set(msg_df_true['msg_number'])), max(set(msg_df_pred['msg_number'])))
    #
    # a = unique_df_for_msg_number(msg_df_true_filtered[(msg_df_true_filtered['distance_filter']==True)], 'class_label_true')
    # print(list(set(a['msg_item_id'])))
    # b = unique_df_for_msg_number(msg_df_pred_filtered[(msg_df_pred_filtered['distance_filter']==True)], 'class_label_pred')
    # print(list(set(b['msg_item_id'])))

    # print('------pred--------')
    # print(msg_df_pred[(msg_df_pred['msg_number'] == 2239)][
    #           ['position_x', 'position_y', 'position_z', 'timestamp_std', 'dimensions_x']])
    # print(msg_df_pred[(msg_df_pred['msg_number'] == 2239)][
    #           ['dimensions_x', 'dimensions_y', 'dimensions_z', 'timestamp_std', 'class_label_pred']])
    # print(msg_df_pred[(msg_df_pred['msg_number'] == 2239)][
    #           ['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']])
    # print('------true--------')
    # print(msg_df_true[(msg_df_true['msg_number'] == 2239)][
    #           ['position_x', 'position_y', 'position_z', 'timestamp_std', 'dimensions_x']])
    # print(msg_df_true[(msg_df_true['msg_number'] == 2239)][
    #           ['dimensions_x', 'dimensions_y', 'dimensions_z', 'timestamp_std', 'class_label_true']])

    # aa = msg_df_true_filtered[(msg_df_true_filtered['position_z']=='NotGiven') & (msg_df_true_filtered['distance_filter']==True)]
    # print(aa.shape)
