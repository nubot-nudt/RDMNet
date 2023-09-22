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

def eval_mulran_one_epoch(args, cfg, logger):
    features_root = cfg.feature_dir

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter('precision')
    coarse_matching_meter.register_meter('PMR>0')
    coarse_matching_meter.register_meter('PMR>=0.1')
    coarse_matching_meter.register_meter('PMR>=0.3')
    coarse_matching_meter.register_meter('PMR>=0.5')

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter('recall')
    fine_matching_meter.register_meter('inlier_ratio')
    fine_matching_meter.register_meter('overlap')

    registration_meter = SummaryBoard()
    registration_meter.register_meter('recall')
    registration_meter.register_meter('rre')
    registration_meter.register_meter('rte')
    registration_meter.register_meter('x')
    registration_meter.register_meter('y')
    registration_meter.register_meter('z')

    overlap_meter = SummaryBoard()
    overlap_meter.register_meter('n2p_p_mean')
    overlap_meter.register_meter('n2p_n_mean')
    overlap_meter.register_meter('n2p_p_std')
    overlap_meter.register_meter('n2p_n_std')

    fail_case = []

    file_names = glob.glob(osp.join(features_root, '*.npz'))
    num_test_pairs = len(file_names)
    for i, file_name in tqdm(enumerate(file_names), total=num_test_pairs):
        seq_id, src_frame, ref_frame = [x for x in osp.splitext(osp.basename(file_name))[0].split('_')]
        src_frame=int(src_frame)
        ref_frame=int(ref_frame)
        # sch delete bad data

        # if seq_id!='riveside01':
        #     continue
        
        # if seq_id!=8 or src_frame!=2486:
        #     continue

        data_dict = np.load(file_name)

        ref_nodes = data_dict['ref_points_c']
        src_nodes = data_dict['src_points_c']
        ref_node_corr_indices = data_dict['ref_node_corr_indices']
        src_node_corr_indices = data_dict['src_node_corr_indices']

        ref_corr_points = data_dict['ref_corr_points']
        src_corr_points = data_dict['src_corr_points']
        corr_scores = data_dict['corr_scores']

        gt_node_corr_indices = data_dict['gt_node_corr_indices']
        gt_transform = data_dict['transform']

        ref_points=data_dict['ref_points'],
        src_points=data_dict['src_points'],

        

        if args.num_corr is not None and corr_scores.shape[0] > args.num_corr:
            sel_indices = np.argsort(-corr_scores)[: args.num_corr]
            ref_corr_points = ref_corr_points[sel_indices]
            src_corr_points = src_corr_points[sel_indices]
            corr_scores = corr_scores[sel_indices]

        message = '{}/{}, seq_id: {}, id0: {}, id1: {}'.format(i + 1, num_test_pairs, seq_id, src_frame, ref_frame)

        # 1. evaluate correspondences
        # 1.1 evaluate coarse correspondences
        coarse_matching_result_dict = evaluate_sparse_correspondences(
            ref_nodes,
            src_nodes,
            ref_node_corr_indices,
            src_node_corr_indices,
            gt_node_corr_indices,
        )

        coarse_precision = coarse_matching_result_dict['precision']

        coarse_matching_meter.update('precision', coarse_precision)
        coarse_matching_meter.update('PMR>0', float(coarse_precision > 0))
        coarse_matching_meter.update('PMR>=0.1', float(coarse_precision >= 0.1))
        coarse_matching_meter.update('PMR>=0.3', float(coarse_precision >= 0.3))
        coarse_matching_meter.update('PMR>=0.5', float(coarse_precision >= 0.5))

        # 1.2 evaluate fine correspondences
        fine_matching_result_dict = evaluate_correspondences(
            ref_corr_points,
            src_corr_points,
            gt_transform,
            positive_radius=cfg.eval.acceptance_radius,
        )
        

        inlier_ratio = fine_matching_result_dict['inlier_ratio']
        overlap = fine_matching_result_dict['overlap']

        fine_matching_meter.update('inlier_ratio', inlier_ratio)
        fine_matching_meter.update('overlap', overlap)
        fine_matching_meter.update('recall', float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))

        message += ', c_PIR: {:.3f}'.format(coarse_precision)
        message += ', f_IR: {:.3f}'.format(inlier_ratio)
        message += ', f_OV: {:.3f}'.format(overlap)
        message += ', f_RS: {:.3f}'.format(fine_matching_result_dict['residual'])
        message += ', f_NU: {}'.format(fine_matching_result_dict['num_corr'])

        # 2. evaluate registration
        if args.method == 'lgr':
            est_transform = data_dict['estimated_transform']
        elif args.method == 'ransac':
            est_transform = registration_with_ransac_from_correspondences(
                src_corr_points,
                ref_corr_points,
                distance_threshold=cfg.ransac.distance_threshold,
                ransac_n=cfg.ransac.num_points,
                num_iterations=cfg.ransac.num_iterations,
            )
        elif args.method == 'svd':
            with torch.no_grad():
                ref_corr_points = torch.from_numpy(ref_corr_points).cuda()
                src_corr_points = torch.from_numpy(src_corr_points).cuda()
                corr_scores = torch.from_numpy(corr_scores).cuda()
                est_transform = weighted_procrustes(
                    src_corr_points, ref_corr_points, corr_scores, return_transform=True
                )
                est_transform = est_transform.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupported registration method: {args.method}.')

        rre, rte, rx, ry, rz = compute_registration_error(gt_transform, est_transform)
        from utils.visualization import vis_shifte_node
        # vis_shifte_node(torch.tensor(src_nodes), torch.tensor(ref_nodes), torch.tensor(src_nodes), torch.tensor(ref_nodes), torch.tensor(src_points), torch.tensor(ref_points), torch.tensor(est_transform, dtype=torch.float))

        accepted = rre < cfg.eval.rre_threshold and rte < cfg.eval.rte_threshold
        if accepted:
            registration_meter.update('rre', rre)
            registration_meter.update('rte', rte)
            registration_meter.update('x',rx)
            registration_meter.update('y',ry)
            registration_meter.update('z',rz)
        else:
            fail_case.append([seq_id,src_frame, ref_frame])
        registration_meter.update('recall', float(accepted))
        message += ', r_RRE: {:.3f}'.format(rre)
        message += ', r_RTE: {:.3f}'.format(rte)

        if args.verbose:
            logger.info(message)

        # 3. evaluate overlap
        # ref_n2n_scores_c = data_dict['ref_n2n_scores_c']
        # src_n2n_scores_c = data_dict['src_n2n_scores_c']
        # ref_n2p_scores_c = data_dict['ref_n2p_scores_c']
        # src_n2p_scores_c = data_dict['src_n2p_scores_c']
        # ref_points_f = data_dict['ref_points_f']
        # src_points_f = data_dict['src_points_f']

        # overlap_result_dict = evaluate_overlap(
        #     ref_n2n_scores_c,
        #     src_n2n_scores_c,
        #     ref_n2p_scores_c,
        #     src_n2p_scores_c,
        #     ref_points_f,
        #     src_points_f,
        #     ref_nodes,
        #     src_nodes,
        #     gt_transform,
        #     cfg.Vote.overlap_thres,
        #     cfg.Vote.n2p_overlap_thres,
        # )

        # n2p_p_mean = overlap_result_dict['n2p_p_mean']
        # n2p_n_mean = overlap_result_dict['n2p_n_mean']
        # n2p_p_std = overlap_result_dict['n2p_p_std']
        # n2p_n_std = overlap_result_dict['n2p_n_std']

        
        # overlap_meter.update('n2p_p_mean', n2p_p_mean)
        # overlap_meter.update('n2p_n_mean', n2p_n_mean)
        # overlap_meter.update('n2p_p_std', n2p_p_std)
        # overlap_meter.update('n2p_n_std', n2p_n_std)

    if args.test_epoch is not None:
        logger.critical(f'Epoch {args.test_epoch}, method {args.method}')

    # 1. print correspondence evaluation results
    message = '  Coarse Matching'
    message += ', PIR: {:.3f}'.format(coarse_matching_meter.mean('precision'))
    message += ', PMR>0: {:.3f}'.format(coarse_matching_meter.mean('PMR>0'))
    message += ', PMR>=0.1: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.1'))
    message += ', PMR>=0.3: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.3'))
    message += ', PMR>=0.5: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.5'))
    logger.critical(message)

    message = '  Fine Matching'
    message += ', FMR: {:.4f}'.format(fine_matching_meter.mean('recall'))
    message += ', IR: {:.3f}'.format(fine_matching_meter.mean('inlier_ratio'))
    message += ', OV: {:.3f}'.format(fine_matching_meter.mean('overlap'))
    message += ', std: {:.3f}'.format(fine_matching_meter.std('recall'))
    logger.critical(message)

    # 2. print registration evaluation results
    message = '  Registration'
    message += ', RR: {:.4f}'.format(registration_meter.mean("recall"))
    message += ', RRE: {:.3f}'.format(registration_meter.mean("rre"))
    message += ', RTE: {:.3f}'.format(registration_meter.mean("rte"))
    message += ', Rx: {:.3f}'.format(registration_meter.mean("x"))
    message += ', Ry: {:.3f}'.format(registration_meter.mean("y"))
    message += ', Rz: {:.3f}'.format(registration_meter.mean("z"))
    logger.critical(message)

    # message = '  Fail case'
    # logger.critical(fail_case)

    # 3. print overlap results
    # message = '  Overlap'
    # message += ', n2p_p_mean: {:.3f}'.format(overlap_meter.mean("n2p_p_mean"))
    # message += ', n2p_n_mean: {:.3f}'.format(overlap_meter.mean("n2p_n_mean"))
    # message += ', n2p_p_std: {:.3f}'.format(overlap_meter.mean("n2p_p_std"))
    # message += ', n2p_n_std: {:.3f}'.format(overlap_meter.mean("n2p_n_std"))
    # logger.critical(message)

def eval_one_epoch(args, cfg):
    

    
    dataset='kitti'
    # features_root = cfg.feature_dir+'_self'
    # features_root = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti','geotrans',dataset)

    # cfg.feature_dir = '/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/rdmnet/'
    cfg.feature_dir = '/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/geotrans/'
    if dataset=='self':
        features_root = cfg.feature_dir+'self'
        eval_seq = [1]
        datasets = make_dataset_kitti('/mnt/Mount/Dataset/new/icp10',eval_seq)
    if dataset=='kitti':
        features_root = cfg.feature_dir+'kitti'
        # eval_seq = [8,9,10]
        eval_seq = [8]
        datasets = make_dataset_kitti('/mnt/Mount/Dataset/KITTI_odometry/raw10_for_odom',eval_seq)
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

    # eval_seq = [1]

    # eval_seq= ['kaist01','riveside01','sejong01']

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
        # # file_names = sorted(
        # #     glob.glob(osp.join(features_root, f'*.npz')),
        # #     key=lambda x: [int(i) for i in osp.splitext(osp.basename(x))[0].split('_')],
        # # )
        # num_test_pairs = len(file_names)
        # start_idx=0
        # pbar = tqdm(enumerate(file_names[start_idx:], start=start_idx), total=num_test_pairs)
        # for i, file_name in pbar:
        #     try:
        #         seq_id, src_frame, ref_frame = [int(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]
        #     except:
        #         seq_id, src_frame, ref_frame = [(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]

        #     # sch delete bad data
        #     if seq_id==8 and src_frame==15:
        #         continue
                
        #     # if seq_id==9:
        #     #     continue
            
        #     # seq_id, src_frame, ref_frame=8,79,90
        #     message = f'idx:{i}, seq:{seq_id}, src:{src_frame}, ref:{ref_frame}'
        #     # print(message)
        #     pbar.set_description(message)
            


        #     data_dict = np.load(file_name)
        num_test_pairs = len(datasets)
        start_idx=0
        # pbar = tqdm(enumerate(file_names[start_idx:], start=start_idx), total=num_test_pairs)
        # pa=[[236,1643],[234,1648],[231,1642],[227,1635],[236,1636],[236,1652],[228,1643],[245,1643],[250,1643],[255,1643],[220,1643],[260,1643],[255,1648],[255,1638]]
        # pa=[[236,1643],[250,1643],[255,1643],[220,1643],[255,1648],[255,1638]]
        pa=[[255,1638]]

        for i in range(num_test_pairs):

            metadata = datasets[i]
            # seq_id = metadata['seq_id']
            # ref_frame = metadata['frame0']
            # src_frame = metadata['frame1']

            seq_id = 8
            src_frame, ref_frame  = pa[i]
            file_name = osp.join(features_root, f'{seq_id}_{src_frame}_{ref_frame}.npz')
            
            # idx = [int(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]
            
            # seq_id, src_frame, ref_frame=8,79,90
            message = 'idx:%d, seq:%d, src:%d, ref:%d'%(i, seq_id, src_frame, ref_frame)
            # pbar.set_description(message)
            
            # if seq_id!=8 or src_frame!=2486:
            #     continue

            data_dict = np.load(file_name)

           
            ref_corr_points=(data_dict['ref_corr_points'])
            src_corr_points=(data_dict['src_corr_points'])

            transform=(data_dict['transform'])

            # try:
            #     ref_corr_points = data_dict['ref_corr_points']
            # except:
            #     print(1)
            # src_corr_points = data_dict['src_corr_points']
            # corr_scores = data_dict['corr_scores']

            # gt_node_corr_indices = data_dict['gt_node_corr_indices']

            # ref_points=data_dict['ref_points'],
            # src_points=data_dict['src_points'],

            # ref_file = osp.join('/mnt/Mount/Dataset/new/raw','%02d'%seq_id, '%d.pcd' % (ref_frame))
            # src_file = osp.join('/mnt/Mount/Dataset/new/raw','%02d'%seq_id, '%d.pcd' % (src_frame))

            ref_file = osp.join('/mnt/Mount/Dataset/KITTI_odometry/sequences','%02d'%seq_id, 'velodyne/%06d.bin' % (ref_frame))
            src_file = osp.join('/mnt/Mount/Dataset/KITTI_odometry/sequences','%02d'%seq_id, 'velodyne/%06d.bin' % (src_frame))
            # ref_file = osp.join('/mnt/Mount3/Dataset/apollo/kitti_format/MapData/ColumbiaPark/2018-09-21','%02d'%seq_id, 'velodyne/%06d.bin' % (ref_frame))
            # src_file = osp.join('/mnt/Mount3/Dataset/apollo/kitti_format/MapData/ColumbiaPark/2018-09-21','%02d'%seq_id, 'velodyne/%06d.bin' % (src_frame))
            ref_points_raw = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            src_points_raw = np.fromfile(src_file, dtype=np.float32).reshape(-1, 4)[:, :3]

            # pcd = o3d.io.read_point_cloud(ref_file)
            # o3d.visualization.draw_geometries([pcd])

        

            # ref_nodes = data_dict['ref_points_c']
            # src_nodes = data_dict['src_points_c']
            # ori_ref_points_c = data_dict['ori_ref_points_c']
            # ori_src_points_c = data_dict['ori_src_points_c']
            # shifted_ref_points_c = data_dict['shifted_ref_points_c']
            # shifted_src_points_c = data_dict['shifted_src_points_c']
            # ref_node_corr_indices = data_dict['ref_node_corr_indices']
            # src_node_corr_indices = data_dict['src_node_corr_indices']
            # vis_shifte_node(
            #     ori_src_points_c, 
            #     ori_ref_points_c, 
            #     shifted_src_points_c, 
            #     shifted_ref_points_c, 
            #     src_points_raw, 
            #     ref_points_raw,
            #     src_nodes,
            #     ref_nodes, 
            #     # transform,
            #     color=[1,0.5,0],
            #     src_node_color = [0.2, 1, 0.2],
            #     ref_node_color = [0.2, 0.2, 1],
            #     src_point_color = [0.1, 0.6, 0.1],
            #     ref_point_color = [0.1, 0.1, 0.6],
            #     )
            

 
            from utils.visualization import visualization
            # visualization(
            #     data_dict,
            #     transform,
            #     src_points_raw,
            #     ref_points_raw,
            #     ref_point_color=[0.1, 0.1, 0.6],
            #     src_point_color=[0.1, 0.6, 0.1],
            #     offsets=(0, 0, -30),
            #     find_true=True
            # )

            est_transform = registration_with_ransac_from_correspondences(
                    src_corr_points,
                    ref_corr_points,
                    distance_threshold=cfg.ransac.distance_threshold,
                    ransac_n=cfg.ransac.num_points,
                    num_iterations=cfg.ransac.num_iterations,
                )
            
            RRE=Error_R(est_transform[:3,:3].reshape((1,3,3)),transform[:3,:3].reshape((1,3,3)))
            RTE=Error_t(est_transform[:3,3].reshape((1,3)),transform[:3,3].reshape((1,3)))
            print(RRE,RTE)

            # src_points_raw_t = apply_transform(src_points_raw, est_transform)
            # src_pcd = make_open3d_point_cloud(src_points_raw_t)
            # ref_pcd = make_open3d_point_cloud(ref_points_raw)
            # o3d.visualization.draw_geometries([src_pcd.paint_uniform_color(np.asarray([0.1,0.1,0.6])), ref_pcd.paint_uniform_color(np.asarray([0.1,0.6,0.1]))])

            # src_points_raw_t = apply_transform(src_points_raw, transform)
            # src_pcd = make_open3d_point_cloud(src_points_raw_t)
            # ref_pcd = make_open3d_point_cloud(ref_points_raw)
            # o3d.visualization.draw_geometries([src_pcd.paint_uniform_color(np.asarray([0,0,1])), ref_pcd.paint_uniform_color(np.asarray([0,1,0]))])

            

            # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])


            # pcd0 = to_o3d_pcd(src_points_raw)
            # pcd1 = to_o3d_pcd(ref_points_raw)
            # reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
            #                                         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #                                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
            # icp_trans = reg.transformation
            # icp_trans = np.linalg.inv(icp_trans)
            # icp_cur_pose=np.matmul(icp_cur_pose,icp_trans)
            # icp_traj.append(icp_cur_pose)



        
        

def vis_shifte_node(
        src_node, 
        ref_node, 
        shifted_src_node, 
        shifted_ref_node, 
        src_points, ref_points, 
        final_src_node, 
        final_ref_node, 
        transform=None,
        ref_overlap=None, 
        src_overlap=None,
        color=[0.7,0.7,0],
        src_node_color = [0.8, 0, 0],
        ref_node_color = [0, 0, 0.8],
        ref_point_color = [0.4,0.4,0.8],
        src_point_color = [0.8,0.4,0.4],
        ):
    # translation=[0,200,0]
    # translation2=[0,400,0]
    translation=[0,0,0]
    translation2=[0,0,0]
    # devise_AB=[0,-200,0]

    devise_AB=[0,0,0]
    devise_BA=[0,0,-30]
    ruler = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    ruler = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    # ruler.translate([150,0,0])
    ruler.paint_uniform_color([0, 0, 1])
    # ruler=[]

    radius = 0.2
    # radius = 0.01
    if transform is not None:
        ref_node = apply_transform(ref_node, inverse_transform(transform))
        shifted_ref_node = apply_transform(shifted_ref_node, inverse_transform(transform))
        ref_points = apply_transform(ref_points, inverse_transform(transform))
        final_ref_node = apply_transform(final_ref_node, inverse_transform(transform))
        # translation=apply_transform(translation, transform)
        # translation2=apply_transform(translation2, transform)
    box_list1 = []
    box_list2 = []
    box_list3 = []
    if ref_overlap is not None:
        ref_color = ref_overlap.detach().cpu().numpy()
        src_color = src_overlap.detach().cpu().numpy()
    else:
        ref_color = torch.ones(shifted_ref_node.shape[0])*0.4
        src_color = torch.ones(shifted_src_node.shape[0])*0.4

    
    for i in range(src_node.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(src_node[i].reshape([3, 1])).translate(devise_BA)
        # mesh_box.translate(translation)
        # mesh_box.paint_uniform_color(src_node_color)
        mesh_box.paint_uniform_color([1,0,0])

        mesh_box2 = o3d.geometry.TriangleMesh.create_sphere(radius=radius*0.5)
        mesh_box2.translate(shifted_src_node[i].reshape([3, 1])).translate(devise_BA)
        # mesh_box2.translate(translation+translation2)
        mesh_box2.paint_uniform_color([src_color[i], 0, 0])

        mesh_box3 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box3.translate(src_node[i].reshape([3, 1])).translate(devise_BA)
        mesh_box3.translate(translation)
        # mesh_box3.paint_uniform_color(src_node_color)
        mesh_box3.paint_uniform_color([1,0,0])

        mesh_box4 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box4.translate(shifted_src_node[i].reshape([3, 1])).translate(devise_BA)
        mesh_box4.translate(translation2)
        # mesh_box4.paint_uniform_color(src_node_color)
        mesh_box4.paint_uniform_color([1,0,0])

        box_list1.append(mesh_box)
        # box_list1.append(mesh_box2)
        box_list2.append(mesh_box3)
        box_list3.append(mesh_box4)
    
    for i in range(ref_node.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(ref_node[i].reshape([3, 1])).translate(devise_AB)
        # mesh_box.translate(translation)
        # mesh_box.paint_uniform_color(ref_node_color)
        mesh_box.paint_uniform_color([1,0,0])

        mesh_box2 = o3d.geometry.TriangleMesh.create_sphere(radius=radius*0.5)
        mesh_box2.translate(shifted_ref_node[i].reshape([3, 1])).translate(devise_AB)
        # mesh_box2.translate(translation+translation2)
        mesh_box2.paint_uniform_color([0, 0, ref_color[i]])

        mesh_box3 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box3.translate(ref_node[i].reshape([3, 1])).translate(devise_AB)
        mesh_box3.translate(translation)
        # mesh_box3.paint_uniform_color(ref_node_color)
        mesh_box3.paint_uniform_color([1,0,0])

        mesh_box4 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box4.translate(shifted_ref_node[i].reshape([3, 1])).translate(devise_AB)
        mesh_box4.translate(translation2)
        # mesh_box4.paint_uniform_color(ref_node_color)
        mesh_box4.paint_uniform_color([1,0,0])

        box_list1.append(mesh_box)
        # box_list1.append(mesh_box2)
        box_list2.append(mesh_box3)
        box_list3.append(mesh_box4)

    src_corr_lines = make_mesh_corr_lines(src_node+(devise_BA), shifted_src_node+(devise_BA),color, radius=0.1)
    ref_corr_lines = make_mesh_corr_lines(ref_node+(devise_AB), shifted_ref_node+(devise_AB),color,radius=0.1)

    box_list4=[]

    for i in range(final_ref_node.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(final_ref_node[i].reshape([3, 1])).translate(devise_AB)
        # mesh_box.translate(translation)
        # mesh_box.paint_uniform_color(ref_node_color)
        mesh_box.paint_uniform_color([1,0,0])

        box_list4.append(mesh_box)
    for i in range(final_src_node.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(final_src_node[i].reshape([3, 1])).translate(devise_BA)
        # mesh_box.translate(translation)
        # mesh_box.paint_uniform_color(src_node_color)
        mesh_box.paint_uniform_color([1,0,0])

        box_list4.append(mesh_box)


    pcd0 = to_o3d_pcd(src_points).translate(devise_BA)
    pcd1 = to_o3d_pcd(ref_points).translate(devise_AB)
    # pcd01 = to_o3d_pcd(src_points.view(-1,3)).translate(translation).translate(devise_BA)
    # pcd11 = to_o3d_pcd(ref_points.view(-1,3)).translate(translation).translate(devise_AB)
    # pcd02 = to_o3d_pcd(src_points.view(-1,3)).translate(translation2).translate(devise_BA)
    # pcd12 = to_o3d_pcd(ref_points.view(-1,3)).translate(translation2).translate(devise_AB)

    # _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
    #     ref_points, shifted_ref_node, 256
    # )


    #######draw point-to-node######
    # box_list5=[]
    # for i in range(final_src_node.shape[0]):
    #     mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    #     mesh_box.translate(final_src_node[i].reshape([3, 1])).translate(devise_BA)
    #     # mesh_box.translate(translation)
    #     mesh_box.paint_uniform_color([1,0,0])

    #     box_list5.append(mesh_box)
    # _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
    #     torch.from_numpy(src_points).cuda(), torch.from_numpy(final_src_node).cuda(), 5000
    # )
    # src_node_knn_indices[~src_node_knn_masks]=0
    # draw_point_to_node2(src_points,final_src_node,src_node_knn_indices.cpu().detach().numpy(),node_colors=None, box=box_list5)

    # box_list5=[]
    # for i in range(src_node.shape[0]):
    #     mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    #     mesh_box.translate(src_node[i].reshape([3, 1])).translate(devise_BA)
    #     # mesh_box.translate(translation)
    #     mesh_box.paint_uniform_color([1,0,0])

    #     box_list5.append(mesh_box)
    # _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
    #     torch.from_numpy(src_points).cuda(), torch.from_numpy(src_node).cuda(), 5000
    # )
    # src_node_knn_indices[~src_node_knn_masks]=0
    # draw_point_to_node2(src_points,src_node,src_node_knn_indices.cpu().detach().numpy(),node_colors=None, box=box_list5)

    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color)]+box_list4)
    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color),*src_corr_lines]+box_list1)





    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color)+pcd1.paint_uniform_color(ref_point_color),*src_corr_lines,*ref_corr_lines]+box_list1)

    # o3d.visualization.draw_geometries([pcd01.paint_uniform_color(src_point_color)+pcd11.paint_uniform_color(ref_point_color)]+box_list2)
    o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color)+pcd1.paint_uniform_color(ref_point_color)]+box_list1)
    o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color)+pcd1.paint_uniform_color(ref_point_color),*src_corr_lines,*ref_corr_lines]+box_list1)
    # o3d.visualization.draw_geometries([pcd02.paint_uniform_color(src_point_color)+pcd12.paint_uniform_color(ref_point_color)]+box_list3)
    o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color)+pcd1.paint_uniform_color(ref_point_color)]+box_list4)
    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])+pcd01+pcd11+pcd02+pcd12,src_corr_lines,ref_corr_lines]+box_list1+box_list2+[ruler])

    # # o3d.visualization.draw([pcd0, pcd1, src_corr_lines, ref_corr_lines, box_list1, box_list2])

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
