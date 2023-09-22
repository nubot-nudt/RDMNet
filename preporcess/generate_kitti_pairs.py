from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable

import os, glob
import torch.multiprocessing
from tqdm import tqdm
import time
from utils.utils_common import to_o3d_pcd
import open3d as o3d

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')


from torch.utils.data import Dataset

import PCLKeypoint




class SparseDataset(Dataset):
    """Sparse correspondences dataset.  
    Reads images from files and creates pairs. It generates keypoints, 
    descriptors and ground truth matches which will be used in training."""

    def __init__(self, opt):
        # self.DATA_FILES = [0,1,2,3,4,5,6,7,8,9,10]

        self.DATA_FILES = [8,9,10]
        
        self.train_path = opt.data.dataset_root

        self.IS_ODOMETRY = True
        self.kitti_cache={}

    def __len__(self):
        return len(self.dataset)
    
    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.train_path + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]
    
    def get_velo2cam(self, drive):
        data_path = self.train_path + '/calib/sequences/%02d/calib.txt' % drive

        with open(data_path, 'r') as f:
            for line in f.readlines():
                _, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    calib = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
                calib = np.reshape(calib, (3, 4))    
                calib = np.vstack([calib, [0, 0, 0, 1]]).T
        return calib
        
    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.train_path + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)
    
    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def prepare_kitti_pairs(self, thres):
        '''
        '''
        print('process thres=%d'%thres)
        for seq in self.DATA_FILES:
  
            fnames = glob.glob(self.train_path + '/sequences/%02d/velodyne/*.bin' % seq)
            assert len(fnames) > 0, f"Make sure that the path {self.train_path} has data {seq}"

            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(seq, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            velo2cam = self.get_velo2cam(seq)
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1)) 

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > thres
            # more_than_10 = pdist > 20
            curr_time = inames[0]

            icp_dir = os.path.join(self.train_path,'icp%d'%thres)
            if not os.path.exists(icp_dir):
                os.makedirs(icp_dir)
            
            f = open(os.path.join(icp_dir, '%02d'%seq),'a')

            while curr_time in inames:
                # dis=5+np.random.rand(1)*10
                # more_than_10 = pdist > dis
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:

                    velo2cam = self.get_velo2cam(seq)
                    all_odometry = self.get_video_odometry(seq, [curr_time, next_time])
                    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]

                    fname0 = self._get_velodyne_fn(seq, curr_time)
                    fname1 = self._get_velodyne_fn(seq, next_time)

                    # XYZ and reflectance
                    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
                    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

                    xyz0 = xyzr0[:, :3]
                    xyz1 = xyzr1[:, :3]

                    # pose1 = torch.tensor(positions[0], dtype=torch.float, device='cpu')
                    # pose2 = torch.tensor(positions[1], dtype=torch.float, device='cpu')
                    # T_cam0_velo = torch.tensor(velo2cam.T, dtype=torch.float, device='cpu')
                    # T_gt = torch.einsum('ab,bc,cd,de->ae', torch.inverse(T_cam0_velo), torch.inverse(pose2), pose1, T_cam0_velo)
                    # xyz0_tensor = np.concatenate([xyz0,np.ones([xyz0.shape[0],1])],axis=1)
                    # xyz0_tensor = torch.tensor(xyz0_tensor, dtype=torch.float, device='cpu')
                    # xyz0_tensor_t = torch.einsum('ki,ij->jk', T_gt, xyz0_tensor.T)[:,:3]
                    
                    M = (velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                            @ np.linalg.inv(velo2cam)).T
                    xyz0_t = self.apply_transform(xyz0, M)

                    pcd0 = to_o3d_pcd(xyz0_t)
                    pcd1 = to_o3d_pcd(xyz1)
                    

                    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.5, np.eye(4),
                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
                    # pcd0.transform(reg.transformation)
                    # M3 = torch.einsum('ki,ij->kj', torch.tensor(reg.transformation, dtype=torch.float, device='cpu'), T_gt).numpy()
                    M2 = reg.transformation @  M

                    # xyz0_t = self.apply_transform(xyz0, M2)
                    # pcd01 = to_o3d_pcd(xyz0_t)
                    # pcd1 = to_o3d_pcd(xyz1)
                    # o3d.visualization.draw_geometries([pcd01.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # xyz0_t2 = self.apply_transform(xyz0, M @ reg.transformation)
                    # pcd02 = to_o3d_pcd(xyz0_t2)
                    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # o3d.visualization.draw_geometries([pcd01.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # o3d.visualization.draw_geometries([pcd02.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])

                    # with open(os.path.join(self.train_path,'icp3','%02d'% seq),'a') as f:
                    #     f.write(f'{curr_time} {next_time} {M2[0]:.6f} \n')
                    # f.close()

                    M2=M2.reshape(-1)[:12]
                    f.write(f'{curr_time} {next_time} {M2[0]:.6f} {M2[1]:.6f} {M2[2]:.6f} {M2[3]:.6f} {M2[4]:.6f} {M2[5]:.6f} {M2[6]:.6f} {M2[7]:.6f} {M2[8]:.6f} {M2[9]:.6f} {M2[10]:.6f} {M2[11]:.6f} \n')

                    curr_time = next_time + 1
                    print(curr_time)
                    
            f.close()




from experiments.config import make_cfg

if __name__ == '__main__':

    cfg = make_cfg()

    dataset = SparseDataset(cfg)


    dataset.prepare_kitti_pairs(10)

    # for thres in [5,6,7,8,9,11,12,13,14,15]:
    #     train_set.prepare_kitti_pairs(thres)
    