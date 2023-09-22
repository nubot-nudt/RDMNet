import sys
import json
import argparse
import glob
import os.path as osp
import os 
import time
from warnings import catch_warnings

import numpy as np
import torch
from tqdm import tqdm

from config import make_cfg
from geotransformer.engine import Logger
from geotransformer.modules.registration import weighted_procrustes
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences, registration_with_ransac_from_featurematch, make_mesh_corr_lines, make_open3d_point_cloud, make_open3d_axes
from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
    evaluate_overlap
)

import open3d as o3d
from geotransformer.utils.pointcloud import (
    apply_transform,
    inverse_transform,
)
from utils.utils_common import to_o3d_pcd
from geotransformer.modules.ops import point_to_node_partition
import matplotlib.pyplot as plt

import math

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd', 'ransac_featurematch'], default='lgr', help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser


def load_kitti_gt_txt(file):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []

    with open(file, 'r') as f:
    # with open(osp.join(txt_root, '%04d'%seq), 'r') as f:
    # with open(osp.join(txt_root, seq), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):

            line_splitted = line_str.split()
            trans = np.array([float(x) for x in line_splitted[:]])
            trans = np.reshape(trans, (3, 4))    
            trans = np.vstack([trans, [0, 0, 0, 1]])

            pose = trans
            # data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
            # if seq==8 and anc_idx==15:
            #     continue
            dataset.append(pose)
    # dataset.pop(0)
    return dataset


def make_dataset_kitti(txt_path, seq_list):


        dataset = []
        for seq in seq_list:
            dataset += (load_kitti_gt_txt(txt_path, seq))
           
        return dataset

def umeyama_alignment(x: np.ndarray, y: np.ndarray,
                      with_scale: bool = False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    # if x.shape != y.shape:
    #     raise GeometryException("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    # if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
    #     raise GeometryException("Degenerate covariance rank, "
    #                             "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

def cal_recall(rot_error,trans_error,r_thres,t_thres):

    rot_flag = rot_error < r_thres
    trans_flag = trans_error < t_thres
    recall = (rot_flag & trans_flag).sum() / len(rot_flag)

    rot_error = rot_error[rot_flag]
    trans_error = trans_error[trans_flag]

    errors = dict()
    errors['rot_mean'] = round(np.mean(rot_error), 3)
    errors['rot_median'] = round(np.median(rot_error), 3)
    errors['trans_rmse'] = round(np.mean(trans_error), 3)
    errors['trans_rmedse'] = round(np.median(trans_error), 3)
    errors['rot_std'] = round(np.std(rot_error), 3)
    errors['trans_std'] = round(np.std(trans_error), 3)
    return recall, errors
    predator_recalls.append(predator_recall)

def eval_absolute_error(traj,gt_traj,gt_traj_inv):


    r,t,s = umeyama_alignment(traj[:,:3,3].transpose((1,0)),gt_traj[:,:3,3].transpose((1,0)))
    T = np.hstack((r,np.reshape(t,(3,1))))
    T = np.vstack([T, [0,0,0,1]])

    traj_aligned = np.matmul(T,traj)
    # traj_aligned=traj
    
    traj_error = np.abs(np.matmul(gt_traj_inv,traj_aligned)[:,:3,3])
    mean = round(np.mean(traj_error),3)
    std = round(np.std(traj_error),3)
    median = round(np.median(traj_error),3)
    mse = np.sum(traj_error**2)/len(traj_error)
    rmse = round(np.sqrt(mse),3)

    rotation_error = (np.matmul(gt_traj_inv,traj_aligned)[:,:3,:3])
    tr = rotation_error[:, 0, 0] + rotation_error[:, 1, 1] + rotation_error[:, 2, 2]
    rads = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180

    r_mean = round(np.mean(degrees),2)
    r_std = round(np.std(degrees),2) 
    r_mse = np.sum(degrees**2)/len(degrees)
    r_rmse = round(np.sqrt(mse),2)

    

    errors = dict()
    
    
    errors['r_rmse'] = r_rmse
    errors['r_mean'] = r_mean
    errors['r_std'] = r_std
    # errors['median'] = median
    errors['rmse'] = rmse*100
    errors['mean'] = mean*100
    errors['std'] = std*100
    return errors, traj_aligned
    

def Error_R(r1, r2):
    '''
    Calculate isotropic rotation degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.transpose(0, 2, 1)
    r1r2 = np.matmul(r2_inv, r1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180
    return degrees


def Error_t(t1, t2):
    '''
    calculate translation mse error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    '''
    assert t1.shape == t2.shape
    error_t = np.sqrt(np.sum((t1 - t2) ** 2, axis=1))
    return error_t


def eval_traj():

    dataset='kitti'
    seq=10
    # dataset='apollo'
    # seq=1
    # dataset='kitti360'
    # seq=0



    file = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/traj','%s_%d_traj_compare.npz'%(dataset,seq))
    # file = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/traj','%s_%s_traj_compare.npz'%(dataset,seq))
    data = np.load(
            file
        )
    
    traj = data['traj']
    gt_traj=data['gt_traj']
    predator_traj = data['predator_traj']
    nge_traj = data['nge_traj']
    cofi_traj = data['cofi_traj']
    geo_traj = data['geo_traj']
    # icp_traj = data['icp_traj'][:,:3,3]

    gt_traj_inv = np.linalg.inv(gt_traj)
    
    errors,predator_traj = eval_absolute_error(predator_traj,gt_traj,gt_traj_inv)
    print('predator ', errors)
    errors,cofi_traj = eval_absolute_error(cofi_traj,gt_traj,gt_traj_inv)
    print('cofi ', errors)
    errors,nge_traj = eval_absolute_error(nge_traj,gt_traj,gt_traj_inv)
    print('nge ', errors)
    errors,geo_traj = eval_absolute_error(geo_traj,gt_traj,gt_traj_inv)
    print('geo ', errors)
    errors,traj = eval_absolute_error(traj,gt_traj,gt_traj_inv)
    print('rmdnet ', errors)


    # traj = data['traj'][:,:3,3]
    # gt_traj=data['gt_traj'][:,:3,3]
    predator_traj = data['predator_traj'][:,:3,3]
    # nge_traj = data['nge_traj'][:,:3,3]
    # cofi_traj = data['cofi_traj'][:,:3,3]
    # geo_traj = data['geo_traj'][:,:3,3]
    # # icp_traj = data['icp_traj'][:,:3,3]

    traj = traj[:,:3,3]
    gt_traj=data['gt_traj'][:,:3,3]
    # predator_traj = predator_traj[:,:3,3]
    nge_traj = nge_traj[:,:3,3]
    cofi_traj = cofi_traj[:,:3,3]
    geo_traj = geo_traj[:,:3,3]
    
   
    


    lw=2
    fz=12

    plt.clf()
# 
    # plt.subplot(1,2,1)
    plt.plot(nge_traj[:,0], nge_traj[:,1], "b", linewidth=lw,label='ngenet')
    plt.plot(cofi_traj[:,0], cofi_traj[:,1], "g", linewidth=lw,label='cofinet')
    plt.plot(predator_traj[:,0], predator_traj[:,1], "y", linewidth=lw,label='predator')
    plt.plot(geo_traj[:,0], geo_traj[:,1], "m", linewidth=lw,label='geotransformer')
    # plt.plot(icp_traj[:,0], icp_traj[:,1], "k", linewidth=lw,label='icp')
    plt.plot(gt_traj[:,0], gt_traj[:,1], "k", linewidth=lw,label='Ground truth')
    plt.plot(traj[:,0], traj[:,1], "r", linewidth=lw,label='rdmnet')
    # plt.title("sequence 8")
# 
    # plt.subplot(1,2,2)
    # plt.plot(traj2[:,0], traj2[:,1], "r", linewidth=1.0,label='rdmnet')
    # plt.plot(nge_traj2[:,0], nge_traj2[:,1], "b", linewidth=1.0,label='ngenet')
    # plt.plot(cofi_traj2[:,0], cofi_traj2[:,1], "g", linewidth=1.0,label='cofinet')
    # plt.plot(predator_traj2[:,0], predator_traj2[:,1], "y", linewidth=1.0,label='predator')
    # plt.plot(geo_traj2[:,0], geo_traj2[:,1], "m", linewidth=1.0,label='geotransformer')
    # # plt.plot(icp_traj2[:,0], icp_traj2[:,1], "k", linewidth=1.0,label='icp')
    # plt.plot(gt_traj2[:,0], gt_traj2[:,1], "k", linewidth=1.0,label='Ground truth')
    # plt.title("sequence 10")

    # plt.legend(loc=0)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.xlabel('x',fontsize=fz)
    plt.ylabel('y',fontsize=fz)
    plt.axis('equal')
    plt.show()


def eval_pose():
    dataset='self'
    pose_file = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/pose',f'{dataset}_pose_compare.npz')
    data = np.load(
            pose_file
        )


    predator_T = data['predator_T']
    cofi_T=data['cofi_T']
    nge_T = data['nge_T']
    geo_T = data['geo_T']
    ours_T = data['ours_T']
    gt_T = data['gt_T']

    predator_rot_error = Error_R(predator_T[:, :3, :3], gt_T[:, :3, :3])
    predator_trans_error = Error_t(predator_T[:, :3, 3], gt_T[:, :3, 3])
    cofi_rot_error = Error_R(cofi_T[:, :3, :3], gt_T[:, :3, :3])
    cofi_trans_error = Error_t(cofi_T[:, :3, 3], gt_T[:, :3, 3])
    nge_rot_error = Error_R(nge_T[:, :3, :3], gt_T[:, :3, :3])
    nge_trans_error = Error_t(nge_T[:, :3, 3], gt_T[:, :3, 3])
    geo_rot_error = Error_R(geo_T[:, :3, :3], gt_T[:, :3, :3])
    geo_trans_error = Error_t(geo_T[:, :3, 3], gt_T[:, :3, 3])
    ours_rot_error = Error_R(ours_T[:, :3, :3], gt_T[:, :3, :3])
    ours_trans_error = Error_t(ours_T[:, :3, 3], gt_T[:, :3, 3])

    recall, errors = cal_recall(predator_rot_error,predator_trans_error,5,2)
    print('predator ', errors, recall)
    recall, errors = cal_recall(cofi_rot_error,cofi_trans_error,5,2)
    print('cofinet ', errors, recall)
    recall, errors = cal_recall(nge_rot_error,nge_trans_error,5,2)
    print('nge ', errors, recall)
    recall, errors = cal_recall(geo_rot_error,geo_trans_error,5,2)
    print('geo ', errors, recall)
    recall, errors = cal_recall(ours_rot_error,ours_trans_error,5,2)
    print('ours ', errors, recall)


    rot_threshold_range=[0,5]
    trans_threshold_range=[0,0.6]
    rot_interval=0.02
    trans_interval=0.005

    predator_recalls, cofi_recalls, nge_recalls, geo_recalls, ours_recalls = [],[],[],[],[]
    thres=[]
    for rot_threshold in np.arange(rot_threshold_range[0], rot_threshold_range[1], rot_interval):
        trans_threshold = 2
        predator_recall, _ = cal_recall(predator_rot_error,predator_trans_error,rot_threshold,trans_threshold)
        predator_recalls.append(predator_recall)

        cofi_recall, _ = cal_recall(cofi_rot_error,cofi_trans_error,rot_threshold,trans_threshold)
        cofi_recalls.append(cofi_recall)

        nge_recall, _ = cal_recall(nge_rot_error,nge_trans_error,rot_threshold,trans_threshold)
        nge_recalls.append(nge_recall)

        geo_recall, _ = cal_recall(geo_rot_error,geo_trans_error,rot_threshold,trans_threshold)
        geo_recalls.append(geo_recall)

        ours_recall, _ = cal_recall(ours_rot_error,ours_trans_error,rot_threshold,trans_threshold)
        ours_recalls.append(ours_recall)
        thres.append(rot_threshold)

    predator_recalls, cofi_recalls, nge_recalls, geo_recalls, ours_recalls = np.asarray(predator_recalls), np.asarray(cofi_recalls), np.asarray(nge_recalls), np.asarray(geo_recalls), np.asarray(ours_recalls)
    thres=np.asarray(thres)


    fz=24
    lw=2
    plt.clf()

    plt.plot(thres[::-1], nge_recalls[::-1], "b", linewidth=lw,label='ngenet')
    plt.plot(thres[::-1], cofi_recalls[::-1], "g", linewidth=lw,label='cofinet')
    plt.plot(thres[::-1], predator_recalls[::-1], "y", linewidth=lw,label='predator')
    plt.plot(thres[::-1], geo_recalls[::-1], "m", linewidth=lw,label='geotransformer')
    # plt.plot(icp_traj[:,0], icp_traj[:,1], "w", linewidth=lw,label='icp')
    plt.plot(thres[::-1], ours_recalls[::-1], "r", linewidth=lw,label='ours')

    # plt.legend(loc=0)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.xlabel('rotation threshold (°)',fontsize=fz)
    plt.ylabel('Registration Recall',fontsize=fz)
    plt.gca().invert_xaxis()
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.show()
        
    predator_recalls, cofi_recalls, nge_recalls, geo_recalls, ours_recalls = [],[],[],[],[]
    thres=[]
    for trans_threshold in np.arange(trans_threshold_range[0], trans_threshold_range[1], trans_interval):
        rot_threshold = 5
        predator_recall, _ = cal_recall(predator_rot_error,predator_trans_error,rot_threshold,trans_threshold)
        predator_recalls.append(predator_recall)

        cofi_recall, _ = cal_recall(cofi_rot_error,cofi_trans_error,rot_threshold,trans_threshold)
        cofi_recalls.append(cofi_recall)

        nge_recall, _ = cal_recall(nge_rot_error,nge_trans_error,rot_threshold,trans_threshold)
        nge_recalls.append(nge_recall)

        geo_recall, _ = cal_recall(geo_rot_error,geo_trans_error,rot_threshold,trans_threshold)
        geo_recalls.append(geo_recall)

        ours_recall, _ = cal_recall(ours_rot_error,ours_trans_error,rot_threshold,trans_threshold)
        ours_recalls.append(ours_recall)
        thres.append(trans_threshold)

    predator_recalls, cofi_recalls, nge_recalls, geo_recalls, ours_recalls = np.asarray(predator_recalls), np.asarray(cofi_recalls), np.asarray(nge_recalls), np.asarray(geo_recalls), np.asarray(ours_recalls)
    thres=np.asarray(thres)

    plt.clf()

    plt.plot(thres[::-1], nge_recalls[::-1], "b", linewidth=lw,label='ngenet')
    plt.plot(thres[::-1], cofi_recalls[::-1], "g", linewidth=lw,label='cofinet')
    plt.plot(thres[::-1], predator_recalls[::-1], "y", linewidth=lw,label='predator')
    plt.plot(thres[::-1], geo_recalls[::-1], "m", linewidth=lw,label='geotransformer')
    # plt.plot(icp_traj[:,0], icp_traj[:,1], "w", linewidth=lw,label='icp')
    plt.plot(thres[::-1], ours_recalls[::-1], "r", linewidth=lw,label='ours')

    # plt.legend(loc=0)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.xlabel('translation threshold (m)',fontsize=fz)
    plt.ylabel('Registration Recall',fontsize=fz)
    plt.gca().invert_xaxis()
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.show()

       

def main():
    # eval_traj()
    eval_pose()


if __name__ == '__main__':
    main()
