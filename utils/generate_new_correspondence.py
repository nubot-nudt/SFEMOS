#!/usr/bin/env python3
# @author    Jiafeng Cui
# Copyright (c) 2022 Jiafeng Cui, all rights reserved

import numpy as np
import yaml
import os
import copy
import time
import open3d as o3d
from sklearn.neighbors import KDTree
import math
import sys

def load_pointcloud(pc_path):
    data = np.fromfile(pc_path, dtype=np.float32)
    data = data.reshape(-1, 4)
    pointcloud = data[:,:3]
    return pointcloud


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)


def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args:
        pose_path: (Complete) filename for the pose file
        Returns:
        A numpy array of size nx4x4 with n poses as 4x4 transformation
        matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']
    
    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')
    
    return np.array(poses)


def load_lidar_poses(pose_path, T_cam_velo):
    """ load poses in lidar coordinate system """
    # load poses in camera system
    poses = np.array(load_poses(pose_path))
    T_cam0_w = np.linalg.inv(poses[0])

    # load calibrations: camera0 to velodyne
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)
    # convert poses in LiDAR coordinate system
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(T_cam0_w).dot(pose).dot(T_cam_velo))
    new_poses = np.array(new_poses)
    poses = new_poses

    return poses


def save_correspondence(f_id, correspondence, motionflow_dir=None):
    if not os.path.exists(motionflow_dir):
        os.makedirs(motionflow_dir)
    frame_path = os.path.join(motionflow_dir, str(f_id).zfill(6) + '.index')
    np.save(frame_path, correspondence)


def load_oss_pcs(seq, folder):
    """Load all files in a folder and sort."""
    seq_num = [4541,1101,4661,801,271,2761,1101,1101,4071,1591,1201]
    num = seq_num[seq]
    all_paths = list(range(num))
    all_paths = [os.path.join(folder, "{0:06d}".format(path)+".bin") for path in all_paths]
    return all_paths

if __name__ == "__main__":
    config_path = "/SFEMOS/config/semantic-kitti-mos.yaml"
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(config_path))
    dataset_folder = "/semantickitti/semantic_kitti/dataset/sequences"
    gt_folder = "/SFEMOS/gt"
    seq = "08"
    correspondence_path = os.path.join(gt_folder, seq, "new_correspondence_gt")
    pc_files = load_oss_pcs(int(seq), os.path.join(dataset_folder,seq,"non_ground_velodyne"))
    pc_files.sort()
    calib_file = load_calib(os.path.join(dataset_folder, seq, "calib.txt"))
    # poses = load_lidar_poses(os.path.join(dataset_folder, seq, "poses.txt"), calib_file)
    poses = load_lidar_poses(os.path.join(dataset_folder, seq, "ICP_POSES.txt"), calib_file)
    start = time.time()
    for index in range(len(pc_files)):
        # if index != 43 and index != 4019 and index != 1644 and index != 85 and index != 3245:
        #    continue
        current_pc = load_pointcloud(pc_files[index])
        current_pose = poses[index]
        correspondence = np.zeros((current_pc.shape[0]))
        if index == 0:
            save_correspondence(index, correspondence, correspondence_path)
            continue
        else:
            last_pc = load_pointcloud(pc_files[index - 1])
            last_pose = poses[index - 1]
            last_pc_transform_to_current_coordinate = (np.linalg.inv(current_pose) @ \
                                (last_pose @ np.hstack((last_pc, np.ones((last_pc.shape[0], 1)))).T)).T
            tree = KDTree(last_pc_transform_to_current_coordinate[:,:-1])
            distances, indices = tree.query(current_pc, k=1)
            indices[distances > 1] = -1
            correspondence = indices.reshape(-1)
            save_correspondence(index, correspondence, correspondence_path)
    end = time.time()
    print("cost: ", end - start)

