#!/usr/bin/env python3
# @author    Jiafeng Cui
# Copyright (c) 2022 Jiafeng Cui, all rights reserved

import numpy as np
import yaml
import os
import copy
import time
import open3d as o3d
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from util import load_oss_pcs, load_oss_labels

class ClassificationMetrics(nn.Module):
    """
    Define the classification metrics class 
    for calculating classification task metrics
    """
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes)).long()

    def compute_confusion_matrix(self, pred_labels, gt_labels):
        pred_labels = torch.tensor(pred_labels).long()
        gt_labels = torch.tensor(gt_labels.astype(np.float32)).long()

        idxs = torch.stack([pred_labels, gt_labels], dim=0)
        ones = torch.ones((idxs.shape[-1])).type_as(gt_labels)
        self.conf_matrix = self.conf_matrix.index_put_(tuple(idxs), ones, accumulate=True)

    def getStats(self, confusion_matrix):
        # we only care about moving class
        tp = confusion_matrix.diag()[1]
        fp = confusion_matrix.sum(dim=1)[1] - tp
        fn = confusion_matrix.sum(dim=0)[1] - tp
        return tp, fp, fn

    def getIoU(self, confusion_matrix):
        tp, fp, fn = self.getStats(confusion_matrix)
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        return iou

    def getacc(self, confusion_matrix):
        tp, fp, fn = self.getStats(confusion_matrix)
        total_tp = tp.sum()
        total = tp.sum() + fp.sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean


def load_pointcloud(pc_path):
    data = np.fromfile(pc_path, dtype=np.float32)
    data = data.reshape(-1, 4)
    pointcloud = data[:,:3]
    return pointcloud


def load_label(label_path, config_path):
    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels.reshape((-1))
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(config_path))
    semantic_labels = labels & 0xFFFF
    instance_labels = labels >> 16
    mos_labels = copy.deepcopy(semantic_labels)
    for k, v in config["moving_learning_map"].items():
    # for k, v in config["learning_map"].items():
        mos_labels[semantic_labels == k] = v
    return mos_labels, instance_labels, semantic_labels


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


def save_scene_flow(f_id, motionflow_xyz, motionflow_dir=None):
    if not os.path.exists(motionflow_dir):
        os.makedirs(motionflow_dir)
    frame_path = os.path.join(motionflow_dir, str(f_id).zfill(6) + '.flow')
    np.save(frame_path, motionflow_xyz)


def genereate_correspondence_line_set(src, target, uniform_line_color=None, line_colors=None):
    """
    Args:
        src: (num_points, 3) | xyz
        target: (num_points, 3) | xyz
        uniform_line_color:
        line_colors:

    Returns:

    """
    points = np.concatenate((src, target), axis=0)

    lines = np.arange(src.shape[0] * 2).reshape(-1, 2, order='F')
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    if uniform_line_color is not None:
        colors = np.expand_dims(uniform_line_color, 0).repeat(len(lines), axis=0)
    elif line_colors is not None:
        colors = line_colors
    else:
        colors = np.expand_dims([0.47, 0.53, 0.7], 0).repeat(len(lines), axis=0)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def caculate_the_releative_pose(pc1_xyz, pc2_xyz, flags=None):
    '''
     return the transformation from pc1 to pc2
     verified by Jiafeng Cui
    '''
    # Here, maybe pc1_xyz is f1_ins_xyzi[:, :3], pc2_xyz = f2_part_xyz[:, :3]
    # make pointcloud class for open3D registraiton_icp
    debug = False
    if debug:
        pcd1 = make_open3d_point_cloud(pc1_xyz, color=[1, 0, 0])
        pcd2 = make_open3d_point_cloud(pc2_xyz, color=[0, 0, 1])
    else:
        pcd1 = make_open3d_point_cloud(pc1_xyz)  #, color=[1,0,0])
        pcd2 = make_open3d_point_cloud(pc2_xyz)  #, color=[0,0,1])
    # define the initialize matrix used for icp
    init_matrix = np.eye(4)
    if flags == "two_instance":
        init_matrix[0:3, 3] = pc2_xyz.mean(axis=0) - pc1_xyz.mean(axis=0)

        iteration = 200  # 1000
    else:
        iteration = 200
    if debug:
        start = time.time()

    trans = o3d.pipelines.registration.registration_icp(
        pcd1,
        pcd2,
        0.2,
        init_matrix,  #np.eye(4), # max_correspondence_distance, init
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # estimation_method
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))  # criteria

    if trans.fitness < 0.4:

        trans = o3d.pipelines.registration.registration_icp(
            pcd1,
            pcd2,
            0.2,
            np.eye(4),  # max_correspondence_distance, init
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # estimation_method
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))  # criteria
    if debug:
        end = time.time()
        print(f"point_num:{len(pc1_xyz)} cost_time:{end-start}")
    if debug:
        tmp_tranform_xyz = (trans.transformation @ np.hstack((pc1_xyz[:, :3], np.ones((pc1_xyz.shape[0], 1)))).T).T
        pcd3 = make_open3d_point_cloud(tmp_tranform_xyz[:, :3], color=[0, 1, 0])
        corrds = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        vis_flow_lineset = genereate_correspondence_line_set(pc1_xyz, tmp_tranform_xyz[:, :3])
        o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, vis_flow_lineset])  #corrds,
    return trans


def eulerAnglesToRotationMatrix(theta) :
    # theta is rad
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


def main(args):
    config_path = "~/SFEMOS/config/semantic-kitti-mos.yaml"
    dataset_folder = args.root_dir
    gt_folder = args.save_dir
    seq = "04"
    mini_points_for_one_instance = 50
    mos_gt_path = os.path.join(gt_folder, seq, "mos_gt")
    sceneflow_gt_path = os.path.join(gt_folder, seq, "dynamic_residual_gt")
    intance_gt_path = os.path.join(gt_folder, seq, "intance_gt")
    scan_folder = os.path.join(dataset_folder, seq, 'non_ground_velodyne')
    semantic_folder = os.path.join(dataset_folder, seq, 'non_ground_labels')
    pc_files = load_oss_pcs(int(seq), scan_folder)
    label_files = load_oss_labels(int(seq), semantic_folder)
    calib_file = load_calib(os.path.join(dataset_folder, seq, "calib.txt"))
    poses = load_lidar_poses(os.path.join(dataset_folder, seq, "ICP_POSES.txt"), calib_file)
    start = time.time()
    evaluator = ClassificationMetrics(n_classes=2)

    for index in range(len(pc_files)):
        current_pc = load_pointcloud(pc_files[index])
        current_mos_labels, current_instance_labels, current_semantic_labels = load_label(label_files[index], config_path)
        current_pose = poses[index]
        current_moving_mask = (current_semantic_labels > 250)
        scene_flow = np.zeros((current_pc.shape[0], 3))
        if index == 0:
            save_scene_flow(index, scene_flow, sceneflow_gt_path)
            continue
        else:
            last_pc = load_pointcloud(pc_files[index - 1])
            last_mos_labels, last_instance_labels, last_semantic_labels = load_label(label_files[index - 1], config_path)
            last_pose = poses[index - 1]
            last_moving_mask = (last_semantic_labels > 250)
            last_pc_transform_to_current_coordinate = (np.linalg.inv(current_pose) @ \
                                (last_pose @ np.hstack((last_pc, np.ones((last_pc.shape[0], 1)))).T)).T
            # T_world_to_last @ T_current_to_world @ current_points_in_current = current_points_in_last
            current_pc_transform_to_last_coordinate = (np.linalg.inv(last_pose) @ \
                (current_pose@ np.hstack((current_pc,np.ones((current_pc.shape[0],1)))).T)).T
            current_moving_instance_ids, current_moving_instance_pcnum = np.unique(current_instance_labels[current_moving_mask], return_counts=True)
            last_moving_instance_ids, last_moving_instance_pcnum = np.unique(last_instance_labels[last_moving_mask], return_counts=True)
            for ins_id, ins_pcnum in zip(current_moving_instance_ids, current_moving_instance_pcnum):
                if ins_id == 0: # id=0, background points
                    continue
                if ins_pcnum < mini_points_for_one_instance:
                    # print(f'sequence:{seq} lidar_id:{index} ins_id:{ins_id} ins_pcnum:{ins_pcnum} too less points to ICP\n')
                    continue
                # this may find points whose ins_id is correct while not in moving mask
                current_ins_mask = (current_instance_labels == ins_id)
                # make sure points in both instance and moving mask
                current_ins_points = current_pc[current_ins_mask & current_moving_mask]
                # if this instance in current frame is also in last frame, and the instance points in last frame is more then 20% of mini_points_for_one_instance
                # same as current
                if (ins_id in last_moving_instance_ids) and ((last_instance_labels == ins_id) & last_moving_mask).sum() > 0.2 * mini_points_for_one_instance:
                    last_ins_points = last_pc_transform_to_current_coordinate[(last_instance_labels == ins_id) & last_moving_mask][:, :3]
                    flags = 'two_instance'

                else:
                    last_ins_points = last_pc_transform_to_current_coordinate[last_moving_mask][:, :3]
                    if ins_id in last_moving_instance_ids:
                        temp_content = 'in last_moving_instance_ids'
                    else:
                        temp_content = 'not in last_moving_instance_ids'
                    flags = 'unmatched'
                trans = caculate_the_releative_pose(current_ins_points, last_ins_points, flags) 
                tmp_tranform_xyz = (trans.transformation @ np.hstack((current_ins_points, np.ones((current_ins_points.shape[0], 1)))).T).T
                ins_flow_xyz = current_ins_points - tmp_tranform_xyz[:, :3]
                # make sure points in both instance and moving mask
                scene_flow[current_ins_mask & current_moving_mask] = ins_flow_xyz
            
            scene_flow_norm = np.linalg.norm(scene_flow, axis=1)
            pred_mos = (scene_flow_norm > 0).astype(np.float32)
            evaluator.compute_confusion_matrix(pred_mos, current_mos_labels)
            # # add ego motion
            # ego_motion = current_pc - current_pc_transform_to_last_coordinate[:, :3]
            # scene_flow += ego_motion
            save_scene_flow(index, scene_flow, sceneflow_gt_path)
    end = time.time()
    print("cost: ", end - start, " secs ")
    iou = evaluator.getIoU(evaluator.conf_matrix)
    acc = evaluator.getacc(evaluator.conf_matrix)
    print("sequence: ", seq)
    print("IOU: ", iou)
    print("ACC: ", acc)    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--root_dir', type=str, default="/semantickitti/semantic_kitti/dataset/", help='Path for the origial dataset.')
    parser.add_argument('--save_dir', type=str, default='/SFEMOS/preprocess_res/gt', help='Path for saving preprocessing results.')
    args = parser.parse_args()
    main(args)