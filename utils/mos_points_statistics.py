#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Developed by Jiadai Sun

import os
import sys
import glob
import json
import yaml
import numpy as np

from tqdm import tqdm
# from glob import glob
# from icecream import ic


def load_files(folder):
    """Load all files in a folder and sort."""
    root = os.path.realpath(os.path.expanduser(folder))
    all_paths = sorted(os.listdir(root))
    all_paths = [os.path.join(root, path) for path in all_paths]
    return all_paths



def get_frame_data(pc_path, label_path):
    pc_data = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
    label = np.fromfile(label_path, dtype=np.uint32).reshape((-1))
    sem_label = label & 0xFFFF
    ins_label = label >> 16

    # return sem_label, ins_label
    return pc_data, sem_label, ins_label


def get_label_data(label_path):
    label = np.fromfile(label_path, dtype=np.uint32).reshape((-1))
    sem_label = label & 0xFFFF
    ins_label = label >> 16

    return sem_label, ins_label


class SeqKITTI():
    def __init__(self, dataset_path, split):
        self.dataset_path = dataset_path
        self.sf_dir = "/SFEMOS/gt"
        if split == "train":
            self.seqs = ['01','03','04','05','06','07','09','10']
        if split == "valid":
            self.seqs = ['10']
        if split == "test":
            self.seqs = ['11','12','13','14','15','16','17','18','19','20','21']


    def get_file_list(self, seq_id):
        velodyne_seq_path = os.path.join(self.dataset_path, "sequences", seq_id, "non_ground_velodyne")
        velodyne_seq_files = sorted(glob.glob(os.path.join(velodyne_seq_path, "*.bin")))

        # load gt semantic segmentation files
        gtsemantic_seq_path = os.path.join(self.dataset_path, "sequences", seq_id, "non_ground_labels")
        gtsemantic_seq_files = sorted(glob.glob(os.path.join(gtsemantic_seq_path, "*.label")))

        # assert len(velodyne_seq_files) == len(gtsemantic_seq_files)

        # return gtsemantic_seq_files
        return velodyne_seq_files, gtsemantic_seq_files

    def count_dynamic_frames(self, write_to_txt=False):

        if write_to_txt:
            fo = open("KITTI_train_split_dynamic_pointnumber.txt", "w")

        self.moving_threshold_num_points = 100

        for seq in self.seqs:

            velodyne_seq_files, gtsemantic_seq_files = self.get_file_list(
                seq_id=seq)
            num_moving_frames = 0

            for frame_idx in range(len(velodyne_seq_files)):

                f1_xyzi, f1_semlabel, f1_inslabel = \
                    get_frame_data(
                        pc_path=velodyne_seq_files[frame_idx], label_path=gtsemantic_seq_files[frame_idx])

                f1_moving_label_mask = (f1_semlabel > 250)

                if f1_moving_label_mask.sum() > self.moving_threshold_num_points:
                    num_moving_frames += 1
                
                if write_to_txt:
                    linestr = f"{seq} " + "%06d"%frame_idx + f" {f1_moving_label_mask.sum()}\n"
                    fo.write(linestr)

            print(f"Seq {seq} | Moving frames / all == {num_moving_frames}/{len(velodyne_seq_files)} = {num_moving_frames / len(velodyne_seq_files)}")

        pass


    def save_dynamic_frames(self, write_to_txt=False):
        skip_frames = dict()

        self.moving_threshold_num_points = 100

        for seq in self.seqs:
            frames = []
            velodyne_seq_files, gtsemantic_seq_files = self.get_file_list(
                seq_id=seq)
            num_static_frames = 0

            for frame_idx in range(len(velodyne_seq_files)):

                f1_semlabel, _ = get_label_data(gtsemantic_seq_files[frame_idx])

                f1_moving_label_mask = (f1_semlabel > 250)

                if f1_moving_label_mask.sum() > 10:
                    num_static_frames = 0
                    continue
                else:
                    num_static_frames += 1
                    if num_static_frames == 5:
                        frames.append(frame_idx)
                        num_static_frames = 0
            skip_frames[seq] = frames
            print("seq: ", seq, " has ", len(frames), " frames skipped\n")
        with open("train_skip_static_frames.json", 'w') as f:
            json.dump(skip_frames, f)


    def count_seqs_points(self,):

        for seq in self.seqs:

            length_min = 1000000
            length_max = -1

            velodyne_seq_files, gtsemantic_seq_files = self.get_file_list(seq_id=seq)
            # assert len(velodyne_seq_files) == len(gtsemantic_seq_files)

            for frame_idx in range(len(velodyne_seq_files)):
                f1_xyzi = np.fromfile(velodyne_seq_files[frame_idx], dtype=np.float32).reshape((-1, 4))

                if f1_xyzi.shape[0] < length_min:
                    length_min = f1_xyzi.shape[0]
                if f1_xyzi.shape[0] > length_max:
                    length_max = f1_xyzi.shape[0]

            print(f"Seq {seq} | min: {length_min} / max: {length_max}")

    def count_moving_points_in_seqs(self,):

        for seq in self.seqs:

            length_min = 1000000
            length_max = -1
            # load point cloud files
            # velodyne_seq_path = os.path.join(dataset_path, "sequences", seq, "non_ground_velodyne")
            # velodyne_seq_files = sorted(glob.glob(os.path.join(velodyne_seq_path, "*.bin")))

            velodyne_seq_files, gtsemantic_seq_files = self.get_file_list(seq_id=seq)
            # assert len(velodyne_seq_files) == len(gtsemantic_seq_files)
            num_moving_frames = 0
            moving_points = 0
            static_points = 0
            static_frames = []
            for frame_idx in range(len(gtsemantic_seq_files)):

                pc1, f1_semlabel, f1_inslabel = \
                    get_frame_data(pc_path=velodyne_seq_files[frame_idx], label_path=gtsemantic_seq_files[frame_idx])
                loc1 = np.logical_and(abs(pc1[:,0]) < 50, abs(pc1[:,1]) < 50)
                f1_semlabel = f1_semlabel[loc1]

                # mapping rae semantic labels to LiDAR-MOS labels
                f1_moving_label_mask = (f1_semlabel > 250)
                f1_semlabel[f1_moving_label_mask] = 251
                f1_semlabel[~f1_moving_label_mask] = 9
                if f1_moving_label_mask.sum() < 100:
                    static_frames.append(frame_idx)
                a, b = np.unique(f1_semlabel, return_counts=True)
                static_points += b[0]
                if len(b) == 1:
                    continue
                moving_points += b[1]
                print(a, b)

            # print(f"Seq {seq} | min: {length_min} / max: {length_max}")
            print("moving: ", moving_points, "static: ", static_points)
            print("static frames: ",len(static_frames))


    def count_mean_std_in_seqs(self,):
        x_sum = np.zeros(1,dtype=np.float32)
        y_sum = np.zeros(1,dtype=np.float32)
        z_sum = np.zeros(1,dtype=np.float32)
        sqrt_x_sum = np.zeros(1,dtype=np.float32)
        sqrt_y_sum = np.zeros(1,dtype=np.float32)
        sqrt_z_sum = np.zeros(1,dtype=np.float32)
        point_sum = np.zeros(1,dtype=np.int32)
        for seq in self.seqs:
            scene_flow_root = os.path.join(self.sf_dir, seq, "scene_flow_gt")
            scene_flow_files = load_files(scene_flow_root)
            for i in range(len(scene_flow_files)):
                if i == 0:
                    continue
                sf_path = scene_flow_files[i]
                scene_flow = np.load(sf_path)
                x_sum += np.sum(scene_flow, axis=0)[0]
                y_sum += np.sum(scene_flow, axis=0)[1]
                z_sum += np.sum(scene_flow, axis=0)[2]
                sqrt_x_sum += np.sum(scene_flow, axis=0)[0] * np.sum(scene_flow, axis=0)[0]
                sqrt_y_sum += np.sum(scene_flow, axis=0)[1] * np.sum(scene_flow, axis=0)[1]
                sqrt_z_sum += np.sum(scene_flow, axis=0)[2] * np.sum(scene_flow, axis=0)[2]
                point_sum += len(scene_flow)
        mean_x = x_sum / point_sum
        mean_y = y_sum / point_sum
        mean_z = z_sum / point_sum
        # var_x = sqrt_x_sum / point_sum - mean_x * mean_x
        # var_y = sqrt_y_sum / point_sum - mean_y * mean_y
        # var_z = sqrt_z_sum / point_sum - mean_z * mean_z
        var_x = (sqrt_x_sum - x_sum * x_sum / point_sum) / (point_sum - 1)
        var_y = (sqrt_y_sum - y_sum * y_sum / point_sum) / (point_sum - 1)
        var_z = (sqrt_z_sum - z_sum * z_sum / point_sum) / (point_sum - 1)
        std_x = np.sqrt(var_x)
        std_y = np.sqrt(var_y)
        std_z = np.sqrt(var_z)
        print("dataset std: ", std_x, std_y, std_z)





if __name__ == '__main__':

    dataset_path = '/semantickitti/semantic_kitti/dataset'
    split = "train"
    seqKITTI = SeqKITTI(dataset_path, split)
    seqKITTI.count_mean_std_in_seqs()
    # ['00':0.001,'01':0.02,'02':0.001,'03':0.002,'04':0.02,'05':0.005,'06':0.002,'07':0.014,'08':0.006,'09':0.005,'10':0.006]
