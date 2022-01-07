# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time
import math
import matplotlib.pyplot as plt
import pandas as pd
from data_initialization import rosbag_ouput_to_dataframe
from distance_filter import distance_filter_remove
from data_matching import data_matching
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    start = time.clock()  # start time
    list_args = [
        ["/home/ubuntu18/liangdao/Data/19-merge.bag", "/home/ubuntu18/liangdao/Data/19_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/20-merge.bag", "/home/ubuntu18/liangdao/Data/20_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/21-merge.bag", "/home/ubuntu18/liangdao/Data/21_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/40-merge.bag", "/home/ubuntu18/liangdao/Data/40_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/42-merge.bag", "/home/ubuntu18/liangdao/Data/42_ALT_OD_CP2.bag"],
        ["/home/ubuntu18/liangdao/Data/48-merge.bag", "/home/ubuntu18/liangdao/Data/48_ALT_OD_CP2.bag"]
    ]
    # obtain data from rosbag document
    pred_list_all = []
    true_list_all = []
    for i in list_args:
        print('------------',i,'-----------------')
        true_path = i[0]
        pred_path = i[1]

        msg_df_pred, msg_df_pred_number = rosbag_ouput_to_dataframe(pred_path, topic=['/ld_object_lists'])
        msg_df_true, msg_df_true_number = rosbag_ouput_to_dataframe(true_path, topic=['/ld_object_lists'])
        # distance filter
        distance_range = [[0, 30], [-10, 10], [-math.inf, math.inf]]
        # msg_df_true_filtered = distance_filter(msg_df_true, distance_range)
        # msg_df_pred_filtered = distance_filter(msg_df_pred, distance_range)

        # distance filter remove
        msg_df_true_filtered_removed = distance_filter_remove(msg_df_true, distance_range)
        msg_df_pred_filtered_removed = distance_filter_remove(msg_df_pred, distance_range)
        # print(msg_df_pred_filtered_removed.shape, msg_df_true_filtered_removed.shape)

        # matching

        pred_list, true_list = data_matching(msg_df_true_filtered_removed, msg_df_pred_filtered_removed, dimension=2)
        pred_list_all.extend(pred_list)
        true_list_all.extend(true_list)
    print(classification_report(pred_list_all, true_list_all))
    cm = confusion_matrix(pred_list_all, true_list_all)
    print(cm)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Car', 'Cyclist', 'Motorcycle', 'Pedestrian', 'Truck', 'UnKnown', 'Van'])
    disp.plot()
    # plt.figure(dpi=100)
    plt.savefig('confusion.png')









    # for item in set(msg_df_pred_filtered_removed['msg_number']):
    #     print(item)



    end = time.clock()  # end time
    print('run time:', str(end - start))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
