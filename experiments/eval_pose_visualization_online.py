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
import shutil


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd', 'ransac_featurematch'], default='lgr', help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser


def load_kitti_gt_txt(txt_root, seq):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []

    with open(osp.join(txt_root, '%02d'%seq), 'r') as f:
    # with open(osp.join(txt_root, '%04d'%seq), 'r') as f:
    # with open(osp.join(txt_root, seq), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):

            line_splitted = line_str.split()
            anc_idx = int(line_splitted[0])
            pos_idx = int(line_splitted[1])
            trans = np.array([float(x) for x in line_splitted[2:]])
            trans = np.reshape(trans, (3, 4))    
            trans = np.vstack([trans, [0, 0, 0, 1]])

            data = {'seq_id': seq, 'frame0':  pos_idx, 'frame1': anc_idx, 'transform': trans}
            # data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
            if seq==8 and anc_idx==15:
                continue
            dataset.append(data)
    # dataset.pop(0)
    return dataset


def make_dataset_kitti(txt_path, seq_list):


        dataset = []
        for seq in seq_list:
            dataset += (load_kitti_gt_txt(txt_path, seq))
           
        return dataset

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

def cal_recall(rot_error,trans_error,r_thres,t_thres):

    rot_flag = rot_error < r_thres
    trans_flag = trans_error < t_thres
    recall = (rot_flag & trans_flag).sum() / len(rot_flag)
    return recall
    predator_recalls.append(predator_recall)

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

def eval_absolute_error(traj,gt_traj,gt_traj_inv):        


    r,t,s = umeyama_alignment(traj[:,:3,3].transpose((1,0)),gt_traj[:,:3,3].transpose((1,0)))
    T = np.hstack((r,np.reshape(t,(3,1))))
    T = np.vstack([T, [0,0,0,1]])

    traj_aligned = np.matmul(T,traj)
    # traj_aligned=trajd
    
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
    r_std = round(np.std(degrees),3) 
    r_mse = np.sum(degrees**2)/len(degrees)
    r_rmse = round(np.sqrt(mse),2)

    

    errors = dict()
    
    
    # errors['r_std'] = r_std
    errors['r_rmse'] = r_rmse
    errors['r_mean'] = r_mean
    # errors['median'] = median
    # errors['std'] = std
    errors['rmse'] = rmse*100
    errors['mean'] = mean*100
    return errors

def eval_one_epoch(args, cfg):
    

    
    dataset='self'
    feature_base_root = '/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/match'



    # if dataset=='self':
    #     features_root = cfg.feature_dir+'_self'
    #     eval_seq = [1]
    # if dataset=='kitti':
    #     features_root = cfg.feature_dir+''
    #     eval_seq = [8,9,10]
    # if dataset=='kitti360':
    #     features_root = cfg.feature_dir+'_kitti360'
    #     eval_seq = [0,2,3,4,5,6,7,9,10]
    # if dataset=='apollo':
    #     features_root = cfg.feature_dir+'_apollo'
    #     eval_seq = [1,2,3,4]
    # if dataset=='mulran':
    #     features_root = cfg.feature_dir+'_mulran'
    #     eval_seq = ['sejong01','kaist01','riveside01']
    #     # features_root = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti','geotrans',dataset)

    cfg.feature_dir = feature_base_root + '/rdmnet/'
    if dataset=='self':
        features_root = cfg.feature_dir+'self'
        eval_seq = [1]
        datasets = make_dataset_kitti('/mnt/Mount/Dataset/new/icp10',eval_seq)
    if dataset=='kitti':
        features_root = cfg.feature_dir+'kitti'
        eval_seq = [8,9,10]
        # eval_seq = [9]
        # datasets = make_dataset_kitti('/mnt/Mount/Dataset/KITTI_odometry/raw10_for_odom',eval_seq)
        datasets = make_dataset_kitti('/mnt/Mount/Dataset/KITTI_odometry/icp10',eval_seq)
    if dataset=='kitti360':
        features_root = cfg.feature_dir+'kitti360'
        # eval_seq = [0,2,3,4,5,6,7,9,10]
        eval_seq = [0]
        datasets = make_dataset_kitti('/mnt/Mount3/Dataset/KITTI-360/raw10_for_odom',eval_seq)
    if dataset=='apollo':
        features_root = cfg.feature_dir+'apollo'
        # eval_seq = [1,2,3,4]
        eval_seq = [1]
        datasets = make_dataset_kitti('/mnt/Mount3/Dataset/apollo/raw10_for_odom',eval_seq)
    if dataset=='mulran':
        features_root = cfg.feature_dir+'mulran'
        eval_seq = ['sejong01','kaist01','riveside01']
        datasets = make_dataset_kitti('/mnt/Mount3/Dataset/mulran_process/icp10',eval_seq)



    predator_T=[]
    cofi_T=[]
    nge_T=[]
    geo_T=[]
    ours_T=[]
    gt_T=[]

    for seq in eval_seq:

        traj = []
        gt_traj = []
        cur_pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        gt_cur_pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        predator_traj=[]
        predator_cur_pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        cofi_traj=[]
        cofi_cur_pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        nge_traj=[]
        nge_cur_pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        geo_traj=[]
        geo_cur_pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        icp_traj=[]
        icp_cur_pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        

        # file_names = sorted(
        #     glob.glob(osp.join(features_root, f'{seq}_*.npz')),
        #     key=lambda x: [i for i in osp.splitext(osp.basename(x))[0].split('_')],
        # )
        # file_names = sorted(
        #     glob.glob(osp.join(features_root, f'{seq}_*.npz')),
        #     key=lambda x: [int(i) for i in osp.splitext(osp.basename(x))[0].split('_')],
        # )
        # num_test_pairs = len(file_names)
        # start_idx=0
        # pbar = tqdm(enumerate(file_names[start_idx:], start=start_idx), total=num_test_pairs)
        # for i, file_name in pbar:
        #     try:
        #         seq_id, src_frame, ref_frame = [int(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]
        #     except:
        #         seq_id, src_frame, ref_frame = [(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]
        
        num_test_pairs = len(datasets)
        start_idx=0
        # pbar = tqdm(enumerate(file_names[start_idx:], start=start_idx), total=num_test_pairs)

        for i in range(num_test_pairs):

            metadata = datasets[i]
            seq_id = metadata['seq_id']
            ref_frame = metadata['frame0']
            src_frame = metadata['frame1']
            file_name = osp.join(features_root, f'{seq_id}_{src_frame}_{ref_frame}.npz')
            # sch delete bad data
            if seq_id==8 and src_frame==15:
                continue
                
            # if seq_id==9:
            #     continue
            
            # seq_id, src_frame, ref_frame=8,79,90
            message = f'idx:{i}, seq:{seq_id}, src:{src_frame}, ref:{ref_frame}'
            # print(message)
            # pbar.set_description(message)
            
            # if seq_id!=8 or src_frame!=2486:
            #     continue

            data_dict = np.load(file_name)


            transform = datasets[i]['transform']
            

            # transform=(data_dict['transform'])

            try:
                est_transform=data_dict['est_transform1']
            except:
                ref_corr_points=(data_dict['ref_corr_points'])
                src_corr_points=(data_dict['src_corr_points'])
                est_transform = registration_with_ransac_from_correspondences(
                        src_corr_points,
                        ref_corr_points,
                        distance_threshold=cfg.ransac.distance_threshold,
                        ransac_n=cfg.ransac.num_points,
                        num_iterations=cfg.ransac.num_iterations,
                    )

            # ref_file = osp.join('/mnt/Mount/Dataset/KITTI_odometry/sequences','%02d'%seq_id, 'velodyne/%06d.bin' % (ref_frame))
            # src_file = osp.join('/mnt/Mount/Dataset/KITTI_odometry/sequences','%02d'%seq_id, 'velodyne/%06d.bin' % (src_frame))
            # ref_file = osp.join('/mnt/Mount3/Dataset/apollo/kitti_format/MapData/ColumbiaPark/2018-09-21','%02d'%seq_id, 'velodyne/%06d.bin' % (ref_frame))
            # src_file = osp.join('/mnt/Mount3/Dataset/apollo/kitti_format/MapData/ColumbiaPark/2018-09-21','%02d'%seq_id, 'velodyne/%06d.bin' % (src_frame))
            # ref_points_raw = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            # src_points_raw = np.fromfile(src_file, dtype=np.float32).reshape(-1, 4)[:, :3]

            # # pcd0 = to_o3d_pcd(src_points_raw)
            # # pcd1 = to_o3d_pcd(ref_points_raw)
            # # reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
            # #                                         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            # #                                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
            # # icp_trans = reg.transformation
            # # icp_trans = np.linalg.inv(icp_trans)
            # # icp_cur_pose=np.matmul(icp_cur_pose,icp_trans)
            # # icp_traj.append(icp_cur_pose)



            gt_pose = np.linalg.inv(transform)
            # print(src_frame)
            # print(transform)
            gt_cur_pose=np.matmul(gt_cur_pose,gt_pose)
            gt_traj.append(gt_cur_pose)
            gt_T.append(transform)


            pose = np.linalg.inv(est_transform)
            cur_pose=np.matmul(cur_pose,pose)
            traj.append(cur_pose)
            ours_T.append(est_transform)


            data_dict = np.load(osp.join(feature_base_root,'predator',dataset,f'{seq_id}_{src_frame}_{ref_frame}.npz'))
            # data_dict = np.load(osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/predator',dataset,f'{i}.npz'))
            est_transform=data_dict['ts_est']
            pose = np.linalg.inv(est_transform)
            predator_cur_pose=np.matmul(predator_cur_pose,pose)
            predator_traj.append(predator_cur_pose)
            predator_T.append(est_transform)

            data_dict = np.load(osp.join(feature_base_root,'ngenet',dataset,f'{seq_id}_{src_frame}_{ref_frame}.npz'))
            # data_dict = np.load(osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/ngenet',dataset,f'{i}.npz'))
            est_transform=data_dict['ts_est']
            pose = np.linalg.inv(est_transform)
            nge_cur_pose=np.matmul(nge_cur_pose,pose)
            nge_traj.append(nge_cur_pose)
            nge_T.append(est_transform)


            data_dict = np.load(osp.join(feature_base_root,'cofinet',dataset,f'{seq_id}_{src_frame}_{ref_frame}.npz'))
            est_transform=data_dict['ts_est']
            pose = np.linalg.inv(est_transform)
            cofi_cur_pose=np.matmul(cofi_cur_pose,pose)
            cofi_traj.append(cofi_cur_pose)
            cofi_T.append(est_transform)

            # old_file=osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/geotrans','apollo',f'{seq_id}_{src_frame}_{ref_frame}.npz')
            # os.remove(old_file)
            # new_file=osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/geotrans',dataset,f'{seq_id}_{src_frame}_{ref_frame}.npz')
            # shutil.copyfile(old_file,new_file)
            data_dict = np.load(osp.join(feature_base_root,'geotrans',dataset,f'{seq_id}_{src_frame}_{ref_frame}.npz'))
            est_transform=data_dict['est_transform']
            pose = np.linalg.inv(est_transform)
            geo_cur_pose=np.matmul(geo_cur_pose,pose)
            geo_traj.append(geo_cur_pose)
            geo_T.append(est_transform)

            # print(cur_pose-gt_cur_pose)

        ##### plot traj curve ####
        # plot_traj(traj,predator_traj,nge_traj,cofi_traj,geo_traj,gt_traj,dataset,seq,True)
        # # plot_traj2(gt_traj,dataset,seq_id)

    #### plot recall curve ####
    plot_recall(ours_T,predator_T,nge_T,cofi_T,geo_T,gt_T,dataset,False)
    


def plot_recall(ours_T,predator_T,nge_T,cofi_T,geo_T,gt_T,dataset,save=True):
    #### plot recall curve ####
    predator_T, cofi_T, nge_T, geo_T, ours_T, gt_T = np.asarray(predator_T), np.asarray(cofi_T), np.asarray(nge_T), np.asarray(geo_T), np.asarray(ours_T), np.asarray(gt_T)

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

    # if save:
    #     #### save pose ####
    #     pose_file = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/pose',f'{dataset}_pose_compare.npz')
    #     np.savez_compressed(
    #             pose_file,
    #             predator_T=predator_T,
    #             cofi_T=cofi_T,
    #             nge_T=nge_T,
    #             geo_T=geo_T,
    #             ours_T=ours_T,
    #             gt_T=gt_T,
    #         )
    
    rot_threshold_range=[0,5]
    trans_threshold_range=[0,0.6]
    rot_interval=0.02
    trans_interval=0.005

    predator_recalls, cofi_recalls, nge_recalls, geo_recalls, ours_recalls = [],[],[],[],[]
    thres=[]
    for rot_threshold in np.arange(rot_threshold_range[0], rot_threshold_range[1], rot_interval):
        trans_threshold = 2
        predator_recall = cal_recall(predator_rot_error,predator_trans_error,rot_threshold,trans_threshold)
        predator_recalls.append(predator_recall)

        cofi_recall = cal_recall(cofi_rot_error,cofi_trans_error,rot_threshold,trans_threshold)
        cofi_recalls.append(cofi_recall)

        nge_recall = cal_recall(nge_rot_error,nge_trans_error,rot_threshold,trans_threshold)
        nge_recalls.append(nge_recall)

        geo_recall = cal_recall(geo_rot_error,geo_trans_error,rot_threshold,trans_threshold)
        geo_recalls.append(geo_recall)

        ours_recall = cal_recall(ours_rot_error,ours_trans_error,rot_threshold,trans_threshold)
        ours_recalls.append(ours_recall)
        thres.append(rot_threshold)

    predator_recalls, cofi_recalls, nge_recalls, geo_recalls, ours_recalls = np.asarray(predator_recalls), np.asarray(cofi_recalls), np.asarray(nge_recalls), np.asarray(geo_recalls), np.asarray(ours_recalls)
    thres=np.asarray(thres)



    fz=12
    plt.clf()

    plt.plot(thres[::-1], ours_recalls[::-1], "r", linewidth=1.0,label='ours')
    plt.plot(thres[::-1], nge_recalls[::-1], "b", linewidth=1.0,label='ngenet')
    plt.plot(thres[::-1], cofi_recalls[::-1], "g", linewidth=1.0,label='cofinet')
    plt.plot(thres[::-1], predator_recalls[::-1], "y", linewidth=1.0,label='predator')
    plt.plot(thres[::-1], geo_recalls[::-1], "m", linewidth=1.0,label='geotransformer')
    # plt.plot(icp_traj[:,0], icp_traj[:,1], "w", linewidth=1.0,label='icp')

    # plt.legend(loc=0)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.xlabel('rotation threshold (°)',fontsize=fz)
    plt.ylabel('Registration Recall',fontsize=fz)
    plt.gca().invert_xaxis()
    plt.show()
        
    predator_recalls, cofi_recalls, nge_recalls, geo_recalls, ours_recalls = [],[],[],[],[]
    thres=[]
    for trans_threshold in np.arange(trans_threshold_range[0], trans_threshold_range[1], trans_interval):
        rot_threshold = 5
        predator_recall = cal_recall(predator_rot_error,predator_trans_error,rot_threshold,trans_threshold)
        predator_recalls.append(predator_recall)

        cofi_recall = cal_recall(cofi_rot_error,cofi_trans_error,rot_threshold,trans_threshold)
        cofi_recalls.append(cofi_recall)

        nge_recall = cal_recall(nge_rot_error,nge_trans_error,rot_threshold,trans_threshold)
        nge_recalls.append(nge_recall)

        geo_recall = cal_recall(geo_rot_error,geo_trans_error,rot_threshold,trans_threshold)
        geo_recalls.append(geo_recall)

        ours_recall = cal_recall(ours_rot_error,ours_trans_error,rot_threshold,trans_threshold)
        ours_recalls.append(ours_recall)
        thres.append(trans_threshold)

    predator_recalls, cofi_recalls, nge_recalls, geo_recalls, ours_recalls = np.asarray(predator_recalls), np.asarray(cofi_recalls), np.asarray(nge_recalls), np.asarray(geo_recalls), np.asarray(ours_recalls)
    thres=np.asarray(thres)

    plt.clf()

    plt.plot(thres[::-1], ours_recalls[::-1], "r", linewidth=1.0,label='ours')
    plt.plot(thres[::-1], nge_recalls[::-1], "b", linewidth=1.0,label='ngenet')
    plt.plot(thres[::-1], cofi_recalls[::-1], "g", linewidth=1.0,label='cofinet')
    plt.plot(thres[::-1], predator_recalls[::-1], "y", linewidth=1.0,label='predator')
    plt.plot(thres[::-1], geo_recalls[::-1], "m", linewidth=1.0,label='geotransformer')
    # plt.plot(icp_traj[:,0], icp_traj[:,1], "w", linewidth=1.0,label='icp')

    # plt.legend(loc=0)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.xlabel('translation threshold (m)',fontsize=fz)
    plt.ylabel('Registration Recall',fontsize=fz)
    plt.gca().invert_xaxis()
    plt.show()

    input("This will overwrite your backuped pose files, press enter to save the new files")
    # if save:
    #### save pose ####
    pose_file = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/pose',f'{dataset}_pose_compare.npz')
    np.savez_compressed(
            pose_file,
            predator_T=predator_T,
            cofi_T=cofi_T,
            nge_T=nge_T,
            geo_T=geo_T,
            ours_T=ours_T,
            gt_T=gt_T,
        )

def plot_traj(traj,predator_traj,nge_traj,cofi_traj,geo_traj,gt_traj,dataset,seq_id,save=True):
    ##### plot traj curve ####
    traj = np.asarray(traj)
    gt_traj = np.asarray(gt_traj)
    predator_traj = np.asarray(predator_traj)
    nge_traj = np.asarray(nge_traj)
    cofi_traj = np.asarray(cofi_traj)
    geo_traj = np.asarray(geo_traj)
    # icp_traj = np.asarray(icp_traj)

    if save:
        #### save traj ####
        traj_file = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/traj',f'{dataset}_{seq_id}_traj_compare.npz')
        np.savez_compressed(
                traj_file,
                traj=traj,
                gt_traj=gt_traj,
                predator_traj=predator_traj,
                nge_traj=nge_traj,
                cofi_traj=cofi_traj,
                geo_traj=geo_traj,
                # icp_traj=icp_traj,
            )
    gt_traj_inv = np.linalg.inv(gt_traj)
    
    errors = eval_absolute_error(traj,gt_traj,gt_traj_inv)
    print('rmdnet ', errors)
    errors = eval_absolute_error(predator_traj,gt_traj,gt_traj_inv)
    print('predator ', errors)
    errors = eval_absolute_error(nge_traj,gt_traj,gt_traj_inv)
    print('nge ', errors)
    errors = eval_absolute_error(cofi_traj,gt_traj,gt_traj_inv)
    print('cofi ', errors)
    errors = eval_absolute_error(geo_traj,gt_traj,gt_traj_inv)
    print('geo ', errors)

    traj = traj[:,:3,3]
    gt_traj=gt_traj[:,:3,3]
    predator_traj = predator_traj[:,:3,3]
    nge_traj = nge_traj[:,:3,3]
    cofi_traj = cofi_traj[:,:3,3]
    geo_traj = geo_traj[:,:3,3]
    # icp_traj = icp_traj[:,:3,3]


    # plt.clf()

    # plt.plot(gt_traj[:,0], gt_traj[:,1], "k", linewidth=1.0,label='Ground truth')
    # plt.plot(traj[:,0], traj[:,1], "r", linewidth=1.0,label='ours')
    # plt.plot(nge_traj[:,0], nge_traj[:,1], "b", linewidth=1.0,label='ngenet')
    # plt.plot(cofi_traj[:,0], cofi_traj[:,1], "g", linewidth=1.0,label='cofinet')
    # plt.plot(predator_traj[:,0], predator_traj[:,1], "y", linewidth=1.0,label='predator')
    # plt.plot(geo_traj[:,0], geo_traj[:,1], "m", linewidth=1.0,label='geotransformer')
    # # plt.plot(icp_traj[:,0], icp_traj[:,1], "w", linewidth=1.0,label='icp')

    # plt.legend(loc=0)

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')
    # plt.show()

def plot_traj2(gt_traj_r,dataset,seq_id,save=True):
    ##### plot traj curve ####
    gt_traj_r = np.asarray(gt_traj_r)
    
    pairs_info = []
    pose_file = os.path.join('/mnt/Mount/Dataset/KITTI_odometry', 'poses', f'{seq_id:02}.txt')
    poses = np.genfromtxt(pose_file) # (n, 12)
    poses = np.array([np.vstack([pose.reshape(3, 4), [0, 0, 0, 1]]) for pose in poses]) # (n, 4, 4)
    gt_traj=poses[:,:3,3]
    gt_traj[:,1]=gt_traj[:,2]

   

    # gt_traj=gt_traj[:,:3,3]


    plt.clf()

    plt.plot(gt_traj[:,0], gt_traj[:,1], "k", linewidth=1.0,label='Ground truth')
    plt.plot(gt_traj_r[:,0], gt_traj_r[:,1], "r", linewidth=1.0,label='ours')
    # plt.plot(nge_traj[:,0], nge_traj[:,1], "b", linewidth=1.0,label='ngenet')
    # plt.plot(cofi_traj[:,0], cofi_traj[:,1], "g", linewidth=1.0,label='cofinet')
    # plt.plot(predator_traj[:,0], predator_traj[:,1], "y", linewidth=1.0,label='predator')
    # plt.plot(geo_traj[:,0], geo_traj[:,1], "m", linewidth=1.0,label='geotransformer')
    # plt.plot(icp_traj[:,0], icp_traj[:,1], "w", linewidth=1.0,label='icp')

    plt.legend(loc=0)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    # log_file = osp.join(cfg.log_dir, 'eval-{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
    # logger = Logger(log_file=log_file)

    # message = 'Command executed: ' + ' '.join(sys.argv)
    # logger.info(message)
    # message = 'Configs:\n' + json.dumps(cfg, indent=4)
    # logger.info(message)

    eval_one_epoch(args, cfg)


if __name__ == '__main__':
    main()
