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


if __name__ == "__main__":
    config_path = "~/SFEMOS/config/semantic-kitti-mos.yaml"
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(config_path))
    dataset_folder = "~/semantickitti/semantic_kitti/dataset/sequences"
    gt_folder = "~/SFEMOS/gt"
    seq = "08"
    correspondence_path = os.path.join(gt_folder, seq, "correspondence_gt")
    pc_files = [os.path.join(dataset_folder,seq,"non_ground_velodyne",file) for file in os.listdir(os.path.join(dataset_folder,seq,"non_ground_velodyne"))]
    pc_files.sort()
    calib_file = load_calib(os.path.join(dataset_folder, seq, "calib.txt"))
    # poses = load_lidar_poses(os.path.join(dataset_folder, seq, "poses.txt"), calib_file)
    poses = load_lidar_poses(os.path.join(dataset_folder, seq, "ICP_POSES.txt"), calib_file)
    # load mos and instance groundtruth
    label_files = [os.path.join(dataset_folder,seq,"non_ground_labels",file) for file in os.listdir(os.path.join(dataset_folder,seq,"non_ground_labels"))]
    label_files.sort()

    start = time.time()
    for index in range(len(pc_files)):
        if index < 43:
            continue
        if index > 43:
            break
        current_pc = load_pointcloud(pc_files[index])
        current_pose = poses[index]
        correspondence = np.zeros((current_pc.shape[0]))
        labels = np.fromfile(label_files[index], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels = copy.deepcopy(semantic_labels)
        for k, v in config["moving_learning_map"].items():
            mos_labels[semantic_labels == k] = v
        if index == 0:
            # save_correspondence(index, correspondence, correspondence_path)
            continue
        else:
            last_pc = load_pointcloud(pc_files[index - 1])
            last_pose = poses[index - 1]
            last_pc_transform_to_current_coordinate = (np.linalg.inv(current_pose) @ \
                                (last_pose @ np.hstack((last_pc, np.ones((last_pc.shape[0], 1)))).T)).T
            tree = KDTree(last_pc_transform_to_current_coordinate[:,:-1])
            distances, indices = tree.query(current_pc, k=1)
            correspondence = indices.reshape(-1)


            # dynamic and static sample
            indices = np.arange(len(current_pc))
            dynamic_index = indices[mos_labels > 0]
            static_index = indices[mos_labels == 0]
            assert len(dynamic_index) + len(static_index) == len(current_pc)
            # if len(dynamic_index) > 4096:
            #     sample_dynamic_index = np.random.choice(len(dynamic_index), 4096, replace=False)
            #     dynamic_index = dynamic_index[sample_dynamic_index]
            sample_static_index = np.random.choice(len(static_index), 32000 - len(dynamic_index), replace=False)
            static_index = static_index[sample_static_index]
            dynamic_static_index = np.concatenate((dynamic_index, static_index))
            source = current_pc[dynamic_static_index, :]
            # target = last_pc_transform_to_current_coordinate[correspondence[dynamic_static_index], :-1]

            if current_pc.shape[0] > 32000:
                sample_index = np.random.choice(current_pc.shape[0], 32000, replace=False)
            else:
                sample_index = np.concatenate((np.arange(current_pc.shape[0]), np.random.choice(current_pc.shape[0], 32000 - current_pc.shape[0], replace=True)),axis=-1)
            target = current_pc[sample_index, :]
            random_indices = indices[sample_index]
            random_mos_labels = mos_labels[sample_index]
            random_dynamic_index = random_indices[random_mos_labels > 0]
            print("index: ", index, " moving points: ", len(dynamic_index), " , ", len(random_dynamic_index))
            # if current_pc.shape[0] > 8192:
            #     sample_index = np.random.choice(current_pc.shape[0], 8192, replace=False)
            #     source = current_pc[sample_index, :]
            # if last_pc_transform_to_current_coordinate.shape[0] > 8192:
            #     last_sample_index = np.random.choice(last_pc_transform_to_current_coordinate.shape[0], 8192, replace=False)
            # target = last_pc_transform_to_current_coordinate[last_sample_index,:-1]
            # correspondence_sample = correspondence[sample_index]
            # target = last_pc_transform_to_current_coordinate[correspondence_sample,:-1]


            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source)
            # pcd.points = o3d.utility.Vector3dVector(current_pc[dynamic_index, :])
            color = [1,0,0]
            color = np.tile(color, (len(source), 1))
            # color = np.tile(color, (len(current_pc[dynamic_index, :]), 1))
            pcd.colors = o3d.utility.Vector3dVector(color)

            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(target)
            # pcd2.points = o3d.utility.Vector3dVector(current_pc[random_dynamic_index, :])
            color2 = [0,1,0]
            color2 = np.tile(color2, (len(target), 1))
            # color2 = np.tile(color2, (len(current_pc[random_dynamic_index, :]), 1))
            pcd2.colors = o3d.utility.Vector3dVector(color2)
            o3d.visualization.draw_geometries([pcd, pcd2])
            # save_correspondence(index, correspondence, correspondence_path)
    end = time.time()
    print("cost: ", end - start)

