import os
import time
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def load_files(folder):
    """Load all files in a folder and sort."""
    root = os.path.realpath(os.path.expanduser(folder))
    all_paths = sorted(os.listdir(root))
    all_paths = [os.path.join(root, path) for path in all_paths]
    return all_paths


def load_oss_pcs(seq, folder):
    """Load all files in a folder and sort."""
    seq_num = [4541,1101,4661,801,271,2761,1101,1101,4071,1591,1201]
    num = seq_num[seq]
    all_paths = list(range(num))
    all_paths = [os.path.join(folder, "{0:06d}".format(path)+".bin") for path in all_paths]
    return all_paths


def load_oss_labels(seq, folder):
    """Load all files in a folder and sort."""
    seq_num = [4541,1101,4661,801,271,2761,1101,1101,4071,1591,1201]
    num = seq_num[seq]
    all_paths = list(range(num))
    all_paths = [os.path.join(folder, "{0:06d}".format(path)+".label") for path in all_paths]
    return all_paths


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (np.ndarray): [4x4] transformation matrix
    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = np.arctan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = np.arcsin(rotmatrix[0, 2])
    yaw = np.arctan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[0, 3]
    y = rotmatrix[1, 3]
    z = rotmatrix[2, 3]

    return np.array([x, y, z, roll, pitch, yaw])