import os
import copy
import time
import yaml
import json
import torch
import numpy as np
from utils.util import load_files, load_calib, load_lidar_poses, load_oss_pcs, load_oss_labels
from torch.utils.data import Dataset


class KITTIDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.num_points = config.num_points
        if config.debug:
            self.sequences = [1]
        else:
            # self.sequences = [1,3,4,5,6,7,9,10]
            self.sequences = [0,1,2,3,4,5,6,7,9,10]
        if yaml.__version__ >= '5.1':
            self.config = yaml.load(open(config.config_path), Loader=yaml.FullLoader)
        else:
            self.config = yaml.load(open(config.config_path))
        # with open("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/utils/train_skip_static_frames.json", 'r') as f:
        #     self.skip_frames = json.load(f)
        # self.mean = config.mean
        # self.std = config.std
        self.dataset = []
        self.pcs = {}
        self.poses = {}
        self.labels = {}
        self.scene_flows = {}
        self.residual_flows = {}
        self.correspondences = {}
        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))
            velodyne_root = os.path.join(config.dataset_dir, seqstr, "non_ground_velodyne")
            calib_file = os.path.join(config.dataset_dir, seqstr, "calib.txt")
            calib = load_calib(calib_file)
            poses_file = os.path.join(config.dataset_dir, seqstr, "ICP_POSES.txt")
            label_root = os.path.join(config.dataset_dir, seqstr, "non_ground_labels")
            scene_flow_root = os.path.join(config.sf_dir, seqstr, "scene_flow_gt")
            correspondence_root = os.path.join(config.sf_dir, seqstr, "correspondence_gt")
            self.pcs[seq] = load_oss_pcs(seq, velodyne_root)
            self.poses[seq] = load_lidar_poses(poses_file, calib)
            self.labels[seq] = load_oss_labels(seq, label_root)
            self.scene_flows[seq] = load_files(scene_flow_root)
            self.correspondences[seq] = load_files(correspondence_root)
            for scan_idx in range(1, len(self.pcs[seq])):
                self.dataset.append((seq, scan_idx))


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        seq, index = self.dataset[idx]
        # seqstr = "{0:02d}".format(int(seq))
        # if index in self.skip_frames[seqstr]:
        #     if index + 1 > len(self.pcs[seq]) - 1:
        #         index -= 1
        #     else:
        #         index += 1
        pc1_path = self.pcs[seq][index]
        pc1 = np.fromfile(pc1_path, dtype=np.float32)
        pc1 = pc1.reshape(-1, 4)
        pc1 = pc1[:, :3]
        pc2_path = self.pcs[seq][index-1]
        pc2 = np.fromfile(pc2_path, dtype=np.float32)
        pc2 = pc2.reshape(-1, 4)
        pc2 = pc2[:, :3]

        # load mos and instance groundtruth
        label_path = self.labels[seq][index]
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels = copy.deepcopy(semantic_labels)
        for k, v in self.config["moving_learning_map"].items():
            mos_labels[semantic_labels == k] = v
        # load scene flow groundtruth
        sf_path = self.scene_flows[seq][index]
        scene_flow = np.load(sf_path)
        # residual_path = self.residual_flows[seq][index]
        # residual_flow = np.load(residual_path)
        # load correspondences
        correspondence_path = self.correspondences[seq][index]
        correspondence = np.load(correspondence_path)

        loc1 = np.logical_and(abs(pc1[:,0]) < 50, abs(pc1[:,1]) < 50)
        # loc2 = np.logical_and(abs(pc2[:,0]) < 50, abs(pc2[:,1]) < 50)
        # cor1 = (correspondence != -1)
        # pc1 = pc1[loc1 & cor1, :]
        pc1 = pc1[loc1, :]
        # pc2 = pc2[loc2, :]
        correspondence = correspondence[loc1]
        # correspondence = correspondence[loc1 & cor1]
        # instance_labels = instance_labels[loc1]
        mos_labels = mos_labels[loc1]
        # mos_labels = mos_labels[loc1 & cor1]
        # scene_flow = scene_flow[loc1 & cor1, :]
        # residual_flow = residual_flow[loc1 & cor1, :]

        if pc1.shape[0] > self.num_points:
            sample_index = np.random.choice(pc1.shape[0], self.num_points, replace=False)
        else:
            sample_index = np.concatenate((np.arange(pc1.shape[0]), np.random.choice(pc1.shape[0], self.num_points - pc1.shape[0], replace=True)),axis=-1)
        pc1 = pc1[sample_index, :]
        # if pc2.shape[0] > self.num_points:
        #     sample_index_ = np.random.choice(pc2.shape[0], self.num_points, replace=False)
        # else:
        #     sample_index_ = np.concatenate((np.arange(pc2.shape[0]), np.random.choice(pc2.shape[0], self.num_points - pc2.shape[0], replace=True)),axis=-1)
        # pc2 = pc2[sample_index_, :]
        pc2 = pc2[correspondence[sample_index], :]
        pc1 = torch.tensor(pc1, dtype=torch.float)
        pc2 = torch.tensor(pc2, dtype=torch.float)

        # instance_labels = instance_labels[sample_index]
        # instance_labels = torch.Tensor(instance_labels.astype(np.float32)).long()
        scene_flow = scene_flow[sample_index, :]
        scene_flow = torch.tensor(scene_flow, dtype=torch.float)
        # residual_flow = residual_flow[sample_index, :]
        # residual_flow = torch.tensor(residual_flow, dtype=torch.float)
        mos_labels = mos_labels[sample_index]
        mos_labels = torch.Tensor(mos_labels.astype(np.float32)).long()
        # pc1_pose = self.poses[seq][index]
        # pc2_pose = self.poses[seq][index-1]
        # transform_pc1_to_pc2 = np.linalg.inv(pc2_pose).dot(pc1_pose)
        # transform = torch.tensor(transform_pc1_to_pc2,dtype=torch.float)

        return pc1, pc2, mos_labels


class KITTIEvalDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.num_points = config.num_points
        self.sequences = [8]
        if yaml.__version__ >= '5.1':
            self.config = yaml.load(open(config.config_path), Loader=yaml.FullLoader)
        else:
            self.config = yaml.load(open(config.config_path))
        # self.mean = config.mean
        # self.std = config.std
        self.dataset = []
        self.pcs = {}
        self.poses = {}
        self.labels = {}
        self.scene_flows = {}
        self.residual_flows = {}
        self.correspondences = {}
        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))
            velodyne_root = os.path.join(config.dataset_dir, seqstr, "non_ground_velodyne")
            calib_file = os.path.join(config.dataset_dir, seqstr, "calib.txt")
            calib = load_calib(calib_file)
            poses_file = os.path.join(config.dataset_dir, seqstr, "ICP_POSES.txt")
            label_root = os.path.join(config.dataset_dir, seqstr, "non_ground_labels")
            scene_flow_root = os.path.join(config.sf_dir, seqstr, "scene_flow_gt")
            correspondence_root = os.path.join(config.sf_dir, seqstr, "correspondence_gt")
            self.pcs[seq] = load_oss_pcs(seq, velodyne_root)
            self.poses[seq] = load_lidar_poses(poses_file, calib)
            self.labels[seq] = load_oss_labels(seq, label_root)
            self.scene_flows[seq] = load_files(scene_flow_root)
            self.correspondences[seq] = load_files(correspondence_root)
            for scan_idx in range(1, len(self.pcs[seq])):
                self.dataset.append((seq, scan_idx))


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        seq, index = self.dataset[idx]
        # FLOT & RigidFlow: PC2 is a newer scan than PC1, however, here PC2 is an older scan than PC1
        # Also, the groungtruth flow is from newer to older(PC1 -> PC2)
        pc1_path = self.pcs[seq][index]
        pc1 = np.fromfile(pc1_path, dtype=np.float32)
        pc1 = pc1.reshape(-1, 4)
        pc1 = pc1[:, :3]
        pc2_path = self.pcs[seq][index-1]
        pc2 = np.fromfile(pc2_path, dtype=np.float32)
        pc2 = pc2.reshape(-1, 4)
        pc2 = pc2[:, :3]

        # load mos and instance groundtruth
        label_path = self.labels[seq][index]
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels = copy.deepcopy(semantic_labels)
        for k, v in self.config["moving_learning_map"].items():
            mos_labels[semantic_labels == k] = v
        # load scene flow groundtruth
        sf_path = self.scene_flows[seq][index]
        scene_flow = np.load(sf_path)
        # load residual flow groundtruth
        # rf_path = self.residual_flows[seq][index]
        # residual_flow = np.load(rf_path)

        # load correspondences
        correspondence_path = self.correspondences[seq][index]
        correspondence = np.load(correspondence_path)

        loc1 = np.logical_and(abs(pc1[:,0]) < 50, abs(pc1[:,1]) < 50)
        # loc2 = np.logical_and(abs(pc2[:,0]) < 50, abs(pc2[:,1]) < 50)
        # cor1 = (correspondence != -1)
        # pc1 = pc1[loc1 & cor1, :]
        pc1 = pc1[loc1, :]
        # pc2 = pc2[loc2, :]
        correspondence = correspondence[loc1]
        # correspondence = correspondence[loc1 & cor1]
        # instance_labels = instance_labels[loc1]
        mos_labels = mos_labels[loc1]
        # mos_labels = mos_labels[loc1 & cor1]
        # scene_flow = scene_flow[loc1 & cor1, :]
        # residual_flow = residual_flow[loc1 & cor1, :]

        if pc1.shape[0] > self.num_points:
            sample_index = np.random.choice(pc1.shape[0], self.num_points, replace=False)
        else:
            sample_index = np.concatenate((np.arange(pc1.shape[0]), np.random.choice(pc1.shape[0], self.num_points - pc1.shape[0], replace=True)),axis=-1)
        pc1 = pc1[sample_index, :]
        # if pc2.shape[0] > self.num_points:
        #     sample_index_ = np.random.choice(pc2.shape[0], self.num_points, replace=False)
        # else:
        #     sample_index_ = np.concatenate((np.arange(pc2.shape[0]), np.random.choice(pc2.shape[0], self.num_points - pc2.shape[0], replace=True)),axis=-1)
        # pc2 = pc2[sample_index_, :]
        pc2 = pc2[correspondence[sample_index], :]
        pc1 = torch.tensor(pc1, dtype=torch.float)
        pc2 = torch.tensor(pc2, dtype=torch.float)

        # instance_labels = instance_labels[sample_index]
        # instance_labels = torch.Tensor(instance_labels.astype(np.float32)).long()
        scene_flow = scene_flow[sample_index, :]
        scene_flow = torch.tensor(scene_flow, dtype=torch.float)
        # residual_flow = residual_flow[sample_index, :]
        # residual_flow = torch.tensor(residual_flow, dtype=torch.float)

        mos_labels = mos_labels[sample_index]
        mos_labels = torch.Tensor(mos_labels.astype(np.float32)).long()
        # pc1_pose = self.poses[seq][index]
        # pc2_pose = self.poses[seq][index-1]
        # transform_pc1_to_pc2 = np.linalg.inv(pc2_pose).dot(pc1_pose)
        # transform = torch.tensor(transform_pc1_to_pc2,dtype=torch.float)

        return pc1, pc2, mos_labels


class HAOMOEvalDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.num_points = config.num_points
        self.sequences = list(range(134))
        self.skip_sequences = ['002', '003', '004', '010', '012', '013', '014', '015', '016', '017', '018', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '040', '041', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '057', '062', '063', '064', '070', '071', '072', '075', '076', '080', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '100', '101', '102', '104', '105', '106', '107', '109', '110', '111', '112', '116', '117', '119', '120', '121', '122', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133']
        if yaml.__version__ >= '5.1':
            self.config = yaml.load(open(config.config_path), Loader=yaml.FullLoader)
        else:
            self.config = yaml.load(open(config.config_path))
        # self.mean = config.mean
        # self.std = config.std
        self.dataset = []
        self.pcs = {}
        self.poses = {}
        self.labels = {}
        self.scene_flows = {}
        self.correspondences = {}
        for seq in self.sequences:
            seqstr = "{0:03d}".format(int(seq))
            if seqstr in self.skip_sequences:
                continue
            velodyne_root = os.path.join(config.dataset_dir, seqstr, "non_ground_velodyne")
            calib_file = os.path.join(config.dataset_dir, seqstr, "calib.txt")
            calib = load_calib(calib_file)
            poses_file = os.path.join(config.dataset_dir, seqstr, "poses.txt")
            label_root = os.path.join(config.dataset_dir, seqstr, "non_ground_labels")
            scene_flow_root = os.path.join(config.sf_dir, seqstr, "scene_flow_gt")
            correspondence_root = os.path.join(config.dataset_dir, seqstr, "correspondence_gt")
            self.pcs[seq] = load_files(velodyne_root)
            self.poses[seq] = load_lidar_poses(poses_file, calib)
            self.labels[seq] = load_files(label_root)
            self.scene_flows[seq] = load_files(scene_flow_root)
            # self.correspondences[seq] = load_files(correspondence_root)
            for scan_idx in range(1, len(self.pcs[seq])):
                self.dataset.append((seq, scan_idx))


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        seq, index = self.dataset[idx]
        # FLOT & RigidFlow: PC2 is a newer scan than PC1, however, here PC2 is an older scan than PC1
        # Also, the groungtruth flow is from newer to older(PC1 -> PC2)
        pc1_path = self.pcs[seq][index]
        pc1 = np.fromfile(pc1_path, dtype=np.float32)
        pc1 = pc1.reshape(-1, 3)
        pc2_path = self.pcs[seq][index-1]
        pc2 = np.fromfile(pc2_path, dtype=np.float32)
        pc2 = pc2.reshape(-1, 3)

        # load mos and instance groundtruth
        label_path = self.labels[seq][index]
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels = copy.deepcopy(semantic_labels)
        for k, v in self.config["moving_learning_map"].items():
            mos_labels[semantic_labels == k] = v
        # load scene flow groundtruth
        sf_path = self.scene_flows[seq][index]
        scene_flow = np.load(sf_path)

        # load correspondences
        # correspondence_path = self.correspondences[seq][index]
        # correspondence = np.load(correspondence_path)

        loc1 = np.logical_and(abs(pc1[:,0]) < 50, abs(pc1[:,1]) < 50)
        loc2 = np.logical_and(abs(pc2[:,0]) < 50, abs(pc2[:,1]) < 50)
        # cor1 = (correspondence != -1)
        # pc1 = pc1[loc1 & cor1, :]
        pc1 = pc1[loc1, :]
        pc2 = pc2[loc2, :]
        # correspondence = correspondence[loc1]
        # correspondence = correspondence[loc1 & cor1]
        # instance_labels = instance_labels[loc1]
        mos_labels = mos_labels[loc1]
        scene_flow = scene_flow[loc1, :]

        if pc1.shape[0] > self.num_points:
            sample_index = np.random.choice(pc1.shape[0], self.num_points, replace=False)
        else:
            sample_index = np.concatenate((np.arange(pc1.shape[0]), np.random.choice(pc1.shape[0], self.num_points - pc1.shape[0], replace=True)),axis=-1)
        pc1 = pc1[sample_index, :]
        if pc2.shape[0] > self.num_points:
            sample_index_ = np.random.choice(pc2.shape[0], self.num_points, replace=False)
        else:
            sample_index_ = np.concatenate((np.arange(pc2.shape[0]), np.random.choice(pc2.shape[0], self.num_points - pc2.shape[0], replace=True)),axis=-1)
        pc2 = pc2[sample_index_, :]
        # pc2 = pc2[correspondence[sample_index], :]
        pc1 = torch.tensor(pc1, dtype=torch.float)
        pc2 = torch.tensor(pc2, dtype=torch.float)

        # instance_labels = instance_labels[sample_index]
        # instance_labels = torch.Tensor(instance_labels.astype(np.float32)).long()
        scene_flow = scene_flow[sample_index, :]
        scene_flow = torch.tensor(scene_flow, dtype=torch.float)
        # residual_flow = residual_flow[sample_index, :]
        # residual_flow = torch.tensor(residual_flow, dtype=torch.float)

        mos_labels = mos_labels[sample_index]
        mos_labels = torch.Tensor(mos_labels.astype(np.float32)).long()
        # pc1_pose = self.poses[seq][index]
        # pc2_pose = self.poses[seq][index-1]
        # transform_pc1_to_pc2 = np.linalg.inv(pc2_pose).dot(pc1_pose)
        # transform = torch.tensor(transform_pc1_to_pc2,dtype=torch.float)

        return pc1, pc2, mos_labels