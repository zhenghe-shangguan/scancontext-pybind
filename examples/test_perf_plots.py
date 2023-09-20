import os
import time
from matplotlib import pyplot as plt


if __name__ == "__main__":
    res_path = "/home/cyngn/catkin_ws/src/scancontext-pybind/test/2023-07-25-18-23-39_0830_fastlio_res/test1_no_nearby_pose_constraint/"

    rot_err_list = []
    trans_err_list = []
    scd_rot_diff_list, icp_rot_diff_list = [], []
    scd_trans_diff_list, icp_trans_diff_list = [], []
    init_pose_rot_diff_list, init_pose_trans_diff_list = [], []

    with open(res_path + 'rot_err_list.txt', 'r') as f:
        for val in f:
            rot_err_list.append(float(val))
    with open(res_path + 'trans_err_list.txt', 'r') as f:
        for val in f:
            trans_err_list.append(float(val))


    with open(res_path + 'scd_rot_diff_list.txt', 'r') as f:
        for val in f:
            scd_rot_diff_list.append(float(val))
    with open(res_path + 'icp_rot_diff_list.txt', 'r') as f:
        for val in f:
            icp_rot_diff_list.append(float(val))
    with open(res_path + 'scd_trans_diff_list.txt', 'r') as f:
        for val in f:
            scd_trans_diff_list.append(float(val))
    with open(res_path + 'icp_trans_diff_list.txt', 'r') as f:
        for val in f:
            icp_trans_diff_list.append(float(val))
    with open(res_path + 'init_pose_rot_diff_list.txt', 'r') as f:
        for val in f:
            init_pose_rot_diff_list.append(float(val))
    with open(res_path + 'init_pose_trans_diff_list.txt', 'r') as f:
        for val in f:
            init_pose_trans_diff_list.append(float(val))


    plt.xlabel('Loop Closure Idx')
    plt.ylabel('Delta_Yaw / deg')
    plt.title('Delta_Yaw (2nd_visit - 1st_visit) Estimation Plot for Loop Closures')
    plt.plot(range(len(scd_rot_diff_list)), scd_rot_diff_list, label='SC++', color='red')
    plt.plot(range(len(icp_rot_diff_list)), icp_rot_diff_list, label='ICP(pt2plane)', color='blue')
    plt.plot(range(len(init_pose_rot_diff_list)), init_pose_rot_diff_list, label='FastLIO2', color='green')
    plt.xticks(range(len(scd_rot_diff_list)))
    legend = plt.legend(loc='upper right', fontsize='x-large')

    # plt.xlabel('Loop Closure Idx')
    # plt.ylabel('Delta_Trans_xy / meter')
    # plt.title('Delta_Trans_xy (2nd_visit - 1st_visit) Estimation Plot for Loop Closures')
    # # plt.plot(range(len(scd_trans_diff_list)), scd_trans_diff_list, label='SC++', color='red')
    # plt.plot(range(len(icp_trans_diff_list)), icp_trans_diff_list, label='ICP(pt2plane)', color='blue')
    # plt.plot(range(len(init_pose_trans_diff_list)), init_pose_trans_diff_list, label='FastLIO2', color='green')
    # plt.xticks(range(len(scd_trans_diff_list)))
    # legend = plt.legend(loc='upper right', fontsize='x-large')

    # plt.subplot(1, 2, 1)
    # plt.xlabel('loop closure index')
    # plt.ylabel('rot / deg')
    # plt.title('Abs Error-plot of ScanContext++ Place Recognition (Rotation Angle)')
    # plt.plot(range(len(rot_err_list)), rot_err_list, label='Rot Error SCD2ICP', color='red')
    # legend = plt.legend(loc='upper right', fontsize='x-large')

    # plt.subplot(1, 2, 2)
    # plt.xlabel('loop closure index')
    # plt.ylabel('trans / meter')
    # plt.title('Erro-plot of ScanContext++ Place Recognition (Translation)')
    # plt.plot(range(len(trans_err_list)), trans_err_list, label='Trans Error SCD2ICP', color='blue')
    # legend = plt.legend(loc='upper right', fontsize='x-large')


    plt.show()