import os 
import math
import copy
import time 

import numpy as np 
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pyscancontext as sc

from test_utils import * 
from matplotlib import pyplot as plt


# loop parameters 
ROOF_REMOVAL_HEIGHT = 5.0
LOOP_THRES = 0.2
POSE_DIFF_THRES = 5.0
VOXEL_SIZE = 0.35
ICP_DIST_THRE = 3.0
ICP_MAX_ITER = 2000
CHECK_ICP_ALIGN_VIZ_TOGGLE = False

def from_ts_pose_to_mat(ts_pose):
    trans = np.array(ts_pose[1:4], dtype=np.float64)
    trans = np.reshape(trans, (trans.size, 1))
    rot = R.from_quat(ts_pose[4:8])
    rot_mat = rot.as_matrix()
    mat = np.concatenate((rot_mat, trans), axis=1)
    last_line = np.array([0.0, 0.0, 0.0, 1.0])
    last_line = np.reshape(last_line, (1, last_line.size))
    mat = np.concatenate((mat, last_line), axis=0, dtype=np.float64)
    return mat

def calc_pos_diff(pose1, pose2):
    pos1 = np.array(pose1[1:4])
    pos2 = np.array(pose2[1:4])
    return np.linalg.norm(pos1 - pos2)


if __name__ == "__main__":
    curr_dir = "/home/cyngn/catkin_ws/src/scancontext-pybind/examples/data/fastli2_res/2023-07-25-18-23-39_0830_fastlio_res"

    # load time_stamped poses
    ts_pose_dict = dict()
    cnt = 0
    with open(os.path.join(curr_dir, "gnc_optimized_poses.txt"), 'r') as pose_file, open(os.path.join(curr_dir, "kf_times2.txt"), 'r') as time_file:
        for ts, pose in zip(time_file, pose_file):
            pose_elem = pose.split()
            trans = [float(pose_elem[3]), float(pose_elem[7]), float(pose_elem[11])]
            rot = R.from_matrix([
                [float(pose_elem[0]), float(pose_elem[1]), float(pose_elem[2])], 
                [float(pose_elem[4]), float(pose_elem[5]), float(pose_elem[6])], 
                [float(pose_elem[8]), float(pose_elem[9]), float(pose_elem[10])]
            ])
            quat = list(rot.as_quat())
            ts_pose_dict[cnt] = [float(ts)] + trans + quat
            cnt += 1
            
    # loop detection by scds
    # place recognizer 
    scm = sc.SCManager()
    scd_dict = dict()
    down_cloud_dict = dict()
    loop_dict = dict()
    loop_trans_dict = dict()
    loop_rot_dict = dict()
    time_list = [[] for i in range(5)]
    for filename in os.listdir(os.path.join(curr_dir, 'pcd')):
        seq_id = int(filename[:-4])

        time_init = time.time()

        # load pcd
        pcd_path = os.path.join(curr_dir, 'pcd', filename)
        pcd_o3d = o3d.io.read_point_cloud(pcd_path)
        time1 = time.time()

        # crop the roof: remove the pcd with height above ROOF_REMOVAL_HEIGHT
        bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-float("inf"), -float("inf"), -float("inf")]), np.array([float("inf"), float("inf"), ROOF_REMOVAL_HEIGHT]))
        pcd_o3d = pcd_o3d.crop(bbox)
        time2 = time.time()

        # downsample and estimate normals
        pcd_o3d.voxel_down_sample(voxel_size=VOXEL_SIZE)
        pcd_o3d.estimate_normals()
        time3 = time.time()

        # generate descriptor
        down_cloud_dict[seq_id] = pcd_o3d
        scd = scm.make_scancontext(np.asarray(pcd_o3d.points))
        scd_dict[seq_id] = scd
        time4 = time.time()

        # retrieval (querying with online construction of kdtree)
        scm.add_scancontext(scd)
        nn_idx, nn_dist, yaw_diff = scm.detect_loop()

        time_total = time.time()

        if abs(yaw_diff) > math.pi:
            yaw_diff -= yaw_diff / abs(yaw_diff) * math.pi * 2

        if nn_idx == -1: # skip the very first scans (see NUM_EXCLUDE_RECENT in Scancontext.h) 
            continue 
        if nn_dist < LOOP_THRES: # and calc_pos_diff(ts_pose_dict[seq_id], ts_pose_dict[nn_idx]) < POSE_DIFF_THRES:
            time_set = [time1 - time_init, time2 - time1, time3 - time2, time4 - time3, time_total - time4]
            time_set = [i * 1000.0 for i in time_set]
            for i in range(len(time_list)):
                time_list[i].append(time_set[i])

            loop_dict[seq_id] = nn_idx
            loop_trans_dict[seq_id] = nn_dist
            loop_rot_dict[seq_id] = abs(np.rad2deg(yaw_diff))

            print(f'query: scan {seq_id}')
            print(f' detected nn node - idx: {nn_idx}, distance: {nn_dist:.3f}, yaw_diff: {np.rad2deg(yaw_diff):.1f} deg')
            # note: if use 60 sectors for a SCD, yaw_diff's minimum resolution is 6 deg.

    
    plt.xlabel('Loop Closure Idx')
    plt.ylabel('Time Cost / ms')
    plt.title('SC++ Time Cost Break-down for all Loop Closures')
    bar_color_list = ['red', 'orange', 'green', 'c', 'magenta']
    bar_label_list = ['load pcd', 'crop roof', 'voxel downsample', 'scd generation', 'scd loop detection']
    plt.bar(range(len(time_list[0])), time_list[0], color=bar_color_list[0], label=bar_label_list[0])
    time_bottom_curr = time_list[0]
    for i in range(len(time_list) - 1):
        plt.bar(range(len(time_list[i+1])), time_list[i+1], bottom=time_bottom_curr, color=bar_color_list[i+1], label=bar_label_list[i+1])
        time_bottom_curr = [j + k for j, k in zip(time_list[i+1], time_bottom_curr)]
    plt.xticks(range(0, len(time_bottom_curr), 500))
    legend = plt.legend(loc='center right', fontsize='x-large')
    plt.show()

    # statistics
    loop_gt_trans_dict = dict()
    loop_gt_rot_dict = dict()
    init_pose_rot_diff_list, init_pose_trans_diff_list = [], []
    count = 0
    for id1, id2 in loop_dict.items():
        pcd1 = copy.deepcopy(down_cloud_dict[id1])
        pcd2 = copy.deepcopy(down_cloud_dict[id2])

        # get initial transform matrix
        pose1_mat = from_ts_pose_to_mat(ts_pose_dict[id1])
        pose2_mat = from_ts_pose_to_mat(ts_pose_dict[id2])
        transform_init = np.linalg.inv(pose2_mat)@pose1_mat
        print(f"id1 = {id1}, id2 = {id2}, transform_init = {transform_init}")

        # get the rot and trans diff
        trans = np.array(transform_init[0:3, 3], dtype=np.float64)
        rot = R.from_matrix(transform_init[0:3, 0:3])
        rot_euler = rot.as_euler('zxy', degrees=True)
        init_pose_rot_diff_list.append(abs(rot_euler[0]))
        init_pose_trans_diff_list.append(np.linalg.norm(trans[0:2]))
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, ICP_DIST_THRE, transform_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER)
        )
        transform = reg_p2p.transformation
        count += 1

        loop_gt_trans_dict[id1] = np.linalg.norm(transform[0:3, 3])
        loop_gt_rot_dict[id1] = abs(np.rad2deg(math.acos((transform[0, 0] + transform[1, 1] + transform[2, 2] - 1) / 2.0)))
        print(f"[{count} / {len(loop_dict)}]: {reg_p2p}\n")
        print(f"    trans_amount = {loop_gt_trans_dict[id1]}, trans_xy_amount = {np.linalg.norm(transform[0:2, 3])}, rot_amount/yaw_amount = {loop_gt_rot_dict[id1]}\n")

        if CHECK_ICP_ALIGN_VIZ_TOGGLE:
            pcd1.paint_uniform_color([1, 0.706, 0])
            pcd2.paint_uniform_color([0, 0.651, 0.929])
            pcd1.transform(transform)

            # Create a visualization window
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd1)
            vis.add_geometry(pcd2)

            # Set the view point and visualize
            vis.get_render_option().point_size = 1
            vis.run()
            vis.destroy_window()


    # plot the error
    rot_err_list = []
    trans_err_list = []
    scd_rot_diff_list, icp_rot_diff_list = [], []
    scd_trans_diff_list, icp_trans_diff_list = [], []
    for (_, diff1), (_, diff2) in zip(loop_rot_dict.items(), loop_gt_rot_dict.items()):
        rot_err_list.append(abs(abs(diff1) - abs(diff2)))
        scd_rot_diff_list.append(diff1)
        icp_rot_diff_list.append(diff2)
    for (_, diff1), (_, diff2) in zip(loop_trans_dict.items(), loop_gt_trans_dict.items()):
        trans_err_list.append(abs(abs(diff1) - abs(diff2)))
        scd_trans_diff_list.append(diff1)
        icp_trans_diff_list.append(diff2)

    with open('rot_err_list.txt','w') as f:
        for val in rot_err_list:
            f.write(str(val) + '\n')
    with open('trans_err_list.txt','w') as f:
        for val in trans_err_list:
            f.write(str(val) + '\n')
    
    with open('scd_rot_diff_list.txt','w') as f:
        for val in scd_rot_diff_list:
            f.write(str(val) + '\n')
    with open('icp_rot_diff_list.txt','w') as f:
        for val in icp_rot_diff_list:
            f.write(str(val) + '\n')
    with open('scd_trans_diff_list.txt','w') as f:
        for val in scd_trans_diff_list:
            f.write(str(val) + '\n')
    with open('icp_trans_diff_list.txt','w') as f:
        for val in icp_trans_diff_list:
            f.write(str(val) + '\n')
    with open('init_pose_rot_diff_list.txt','w') as f:
        for val in init_pose_rot_diff_list:
            f.write(str(val) + '\n')
    with open('init_pose_trans_diff_list.txt','w') as f:
        for val in init_pose_trans_diff_list:
            f.write(str(val) + '\n')    

        



# todo 
# '''
#  visualization 
# '''
# # read gt pose to draw the matches 
# poses = np.loadtxt(os.path.join(seq_dir, 'poses.txt'))
# poses_xyz = poses[:, (3,7,11)] # actually cam pose 
# poses_o3d = o3d.geometry.PointCloud()
# poses_o3d.points = o3d.utility.Vector3dVector(poses_xyz)
# o3d.visualization.draw_geometries([poses_o3d])


