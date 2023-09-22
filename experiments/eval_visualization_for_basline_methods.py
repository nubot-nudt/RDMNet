import sys
import json
import argparse
import glob
import os.path as osp
import time
from warnings import catch_warnings
import math
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

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd', 'ransac_featurematch'], default='lgr', help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser

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

def load_kitti_gt_txt(txt_root, seq):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []

    # with open(osp.join(txt_root, '%02d'%seq), 'r') as f:
    # with open(osp.join(txt_root, '%04d'%seq), 'r') as f:
    with open(osp.join(txt_root, '%s'%seq), 'r') as f:
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

def eval_one_epoch(args, cfg):

    dataset='mulran'
    # method='cofinet'
    

    fail_case = []

    if dataset=='self':
        # features_root = cfg.feature_dir+'self'
        eval_seq = [1]
        datasets = make_dataset_kitti('/mnt/Mount/Dataset/new/icp10',eval_seq)
    if dataset=='kitti':
        # features_root = cfg.feature_dir+'kitti'
        eval_seq = [8,9,10]
        # eval_seq = [9]
        datasets = make_dataset_kitti('/mnt/Mount/Dataset/KITTI_odometry/raw10_for_odom',eval_seq)
    if dataset=='kitti360':
        # features_root = cfg.feature_dir+'kitti360'
        # eval_seq = [0,2,3,4,5,6,7,9,10]
        eval_seq = [0]
        datasets = make_dataset_kitti('/mnt/Mount3/Dataset/KITTI-360/raw10_for_odom',eval_seq)
    if dataset=='apollo':
        # features_root = cfg.feature_dir+'apollo'
        # eval_seq = [1,2,3,4]
        eval_seq = [1]
        datasets = make_dataset_kitti('/mnt/Mount3/Dataset/apollo/raw10_for_odom',eval_seq)
    if dataset=='mulran':
        # features_root = cfg.feature_dir+'mulran'
        eval_seq = ['sejong01','kaist01','riveside01']
        datasets = make_dataset_kitti('/mnt/Mount3/Dataset/mulran_process/icp10',eval_seq)

    

    # file_names = sorted(
    #     glob.glob(osp.join(features_root, '*.npz')),
    #     key=lambda x: [int(i) for i in osp.splitext(osp.basename(x))[0].split('_')],
    # )
    num_test_pairs = len(datasets)
    start_idx=0
    # pbar = tqdm(enumerate(file_names[start_idx:], start=start_idx), total=num_test_pairs)
    for i in range(num_test_pairs):

        metadata = datasets[i]
        # seq_id = metadata['seq_id']
        # ref_frame = metadata['frame0']
        # src_frame = metadata['frame1']

        # seq_id = 1
        # src_frame = 379
        # ref_frame = 387

        seq_id = 'sejong01'
        src_frame = 1561008401876652944
        ref_frame = 1561008402576260795

        

        if dataset=='kitti':
            ref_file = osp.join('/mnt/Mount/Dataset/KITTI_odometry/sequences','%02d'%seq_id, 'velodyne/%06d.bin' % (ref_frame))
            src_file = osp.join('/mnt/Mount/Dataset/KITTI_odometry/sequences','%02d'%seq_id, 'velodyne/%06d.bin' % (src_frame))
            ref_points_raw = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            src_points_raw = np.fromfile(src_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        elif dataset=='apollo':
            ref_file = osp.join('/mnt/Mount3/Dataset/apollo/kitti_format/MapData/ColumbiaPark/2018-09-21','%02d'%seq_id, 'velodyne/%06d.bin' % (ref_frame))
            src_file = osp.join('/mnt/Mount3/Dataset/apollo/kitti_format/MapData/ColumbiaPark/2018-09-21','%02d'%seq_id, 'velodyne/%06d.bin' % (src_frame))
            ref_points_raw = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            src_points_raw = np.fromfile(src_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        elif dataset=='mulran':
            ref_file = osp.join('/mnt/Mount/Dataset/mulran', seq_id, 'sensor_data/Ouster/%d.bin' % (ref_frame))
            src_file = osp.join('/mnt/Mount/Dataset/mulran', seq_id, 'sensor_data/Ouster/%d.bin' % (src_frame))
            ref_points_raw = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            src_points_raw = np.fromfile(src_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        elif dataset=='self':
            ref_file = osp.join('/mnt/Mount/Dataset/new/raw','%02d'%seq_id, '%d.pcd' % (ref_frame))
            src_file = osp.join('/mnt/Mount/Dataset/new/raw','%02d'%seq_id, '%d.pcd' % (src_frame))
            ref_points_raw = o3d.io.read_point_cloud(ref_file)
            src_points_raw = o3d.io.read_point_cloud(src_file)
            ref_points_raw=np.asarray(ref_points_raw.points)[:, :3]
            src_points_raw=np.asarray(src_points_raw.points)[:, :3]


        features_root = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti','cofinet',dataset)
        file_name = osp.join(features_root, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        data_dict = np.load(file_name)
        src_pcd=data_dict['src_pcd']
        tgt_pcd=data_dict['tgt_pcd']
        trans=data_dict['trans']
        est_transform=data_dict['ts_est']
        corres=data_dict['corres']
        corres=data_dict['corres']
        corres=data_dict['corres']
        src_corr_points=src_pcd[corres[:,0]]
        ref_corr_points=tgt_pcd[corres[:,1]]
        true_mask = find_true_false(src_corr_points,ref_corr_points, trans)
        print(true_mask.sum(),'/',src_corr_points.shape[0],'=',true_mask.sum()/src_corr_points.shape[0])
        cof_RRE=Error_R(est_transform[:3,:3].reshape((1,3,3)),trans[:3,:3].reshape((1,3,3)))
        cof_RTE=Error_t(est_transform[:3,3].reshape((1,3)),trans[:3,3].reshape((1,3)))
        print(cof_RRE, cof_RTE)
        # draw_point_correspondences(
        #     ref_points_raw,
        #     src_points_raw,
        #     src_corr_points,
        #     ref_corr_points,
        #     true_mask=true_mask
        # )
        # src_points_raw_t = apply_transform(src_points_raw, est_transform)
        # src_pcd = make_open3d_point_cloud(src_points_raw_t)
        # ref_pcd = make_open3d_point_cloud(ref_points_raw)
        # o3d.visualization.draw_geometries([src_pcd.paint_uniform_color(np.asarray([0,0,1])), ref_pcd.paint_uniform_color(np.asarray([0,1,0]))])


        features_root = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti','geotrans',dataset)
        file_name = osp.join(features_root, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        data_dict = np.load(file_name)
        ref_corr_points=(data_dict['ref_corr_points'])
        src_corr_points=(data_dict['src_corr_points'])
        est_transform=data_dict['est_transform']
        true_mask = find_true_false(src_corr_points,ref_corr_points, trans)
        print(true_mask.sum(),'/',src_corr_points.shape[0],'=',true_mask.sum()/src_corr_points.shape[0])
        geo_RRE=Error_R(est_transform[:3,:3].reshape((1,3,3)),trans[:3,:3].reshape((1,3,3)))
        geo_RTE=Error_t(est_transform[:3,3].reshape((1,3)),trans[:3,3].reshape((1,3)))
        print(geo_RRE, geo_RTE)
        # draw_point_correspondences(
        #     ref_points_raw,
        #     src_points_raw,
        #     src_corr_points,
        #     ref_corr_points,
        #     true_mask=true_mask
        # )
        # src_points_raw_t = apply_transform(src_points_raw, est_transform)
        # src_pcd = make_open3d_point_cloud(src_points_raw_t)
        # ref_pcd = make_open3d_point_cloud(ref_points_raw)
        # o3d.visualization.draw_geometries([src_pcd.paint_uniform_color(np.asarray([0,0,1])), ref_pcd.paint_uniform_color(np.asarray([0,1,0]))])


        features_root = osp.join('/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti','rdmnet',dataset)
        file_name = osp.join(features_root, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        data_dict = np.load(file_name)
        ref_corr_points=(data_dict['ref_corr_points'])
        src_corr_points=(data_dict['src_corr_points'])
        est_transform=data_dict['est_transform']
        true_mask = find_true_false(src_corr_points,ref_corr_points, trans)
        print(true_mask.sum(),'/',src_corr_points.shape[0],'=',true_mask.sum()/src_corr_points.shape[0])
        rdm_RRE=Error_R(est_transform[:3,:3].reshape((1,3,3)),trans[:3,:3].reshape((1,3,3)))
        rdm_RTE=Error_t(est_transform[:3,3].reshape((1,3)),trans[:3,3].reshape((1,3)))
        print(rdm_RRE, rdm_RTE)
        # draw_point_correspondences(
        #     ref_points_raw,
        #     src_points_raw,
        #     src_corr_points,
        #     ref_corr_points,
        #     true_mask=true_mask
        # )
        # src_points_raw_t = apply_transform(src_points_raw, est_transform)
        # src_pcd = make_open3d_point_cloud(src_points_raw_t)
        # ref_pcd = make_open3d_point_cloud(ref_points_raw)
        # o3d.visualization.draw_geometries([src_pcd.paint_uniform_color(np.asarray([0,0,1])), ref_pcd.paint_uniform_color(np.asarray([0,1,0]))])


        # if (cof_RRE>2.5 or cof_RTE>1) and (geo_RRE>2.5 or geo_RTE>1) and  (rdm_RRE<2.5 and rdm_RTE<1) and true_mask.sum()/src_corr_points.shape[0]>0.4:
        # if (cof_RRE>2.5 or cof_RTE>1) and (geo_RRE>2.5 or geo_RTE>1) and  (rdm_RRE<2.5 and rdm_RTE<1) :
        #     print(f'{seq_id}_{src_frame}_{ref_frame}')


        # draw_point_correspondences(
        #     ref_points_raw,
        #     src_points_raw,
        #     src_corr_points,
        #     ref_corr_points,
        #     true_mask=true_mask
        # )
        # src_points_raw_t = apply_transform(src_points_raw, est_transform)
        # src_pcd = make_open3d_point_cloud(src_points_raw_t)
        # ref_pcd = make_open3d_point_cloud(ref_points_raw)
        # o3d.visualization.draw_geometries([src_pcd.paint_uniform_color(np.asarray([0,0,1])), ref_pcd.paint_uniform_color(np.asarray([0,1,0]))])

        

        # overlaps=data_dict['overlaps'],

def find_true_false(src_corr_points, ref_corr_points, transform, node_corr_indices=None, thres=1):
    # src_node = apply_transform(src_node, transform)
    # src_points = apply_transform(src_points, transform)
    src_corr_points = apply_transform(src_corr_points, transform)
    if node_corr_indices is None:
        # src_corr_points = apply_transform(src_corr_points, transform)
        # true = torch.norm(src_corr_points-ref_corr_points,dim=-1)<thres
        corr_distances = np.linalg.norm(ref_corr_points - src_corr_points, axis=1)
        true = (corr_distances<thres)
    else:
        true = np.linalg.norm(src_corr_points[node_corr_indices[:,1]]-ref_corr_points[node_corr_indices[:,0]],axis=-1)<thres
    return true
    
from geotransformer.utils.open3d import (
    make_open3d_point_cloud,
    make_open3d_axes,
    make_open3d_corr_lines,
    make_open3d_corr_lines2,
    make_mesh_corr_lines
)       

def draw_point_correspondences(
    ref_points,
    src_points,
    src_corr_points,
    ref_corr_points,
    ref_point_colors=[0,0,1],
    src_point_colors=[0,0.6,0],
    offsets=(0, 0, -30),
    true_mask=None
):

    src_points = src_points + offsets
    src_corr_points = src_corr_points + offsets

    # if ref_node_colors is None:
    #     ref_node_colors = np.random.rand(*ref_nodes.shape)
    #     if ref_point_to_node is not None:
    #         ref_point_colors=np.zeros_like(ref_points)
    #         for i in range(ref_nodes.shape[0]):
    #             ref_point_colors[ref_point_to_node[i,:]] = ref_node_colors[i]
    #     ref_node_colors = np.ones_like(ref_nodes) * np.array([[1, 0, 0]])

    # if src_node_colors is None:
    #     src_node_colors = np.random.rand(*src_nodes.shape)
    #     if ref_point_to_node is not None:
    #         src_point_colors=np.zeros_like(src_points)
    #         for i in range(src_nodes.shape[0]):
    #             src_point_colors[src_point_to_node[i,:]] = src_node_colors[i]
    #     src_node_colors = np.ones_like(src_nodes) * np.array([[1, 0, 0]])\
    
    ref_pcd = make_open3d_point_cloud(ref_points)
    src_pcd = make_open3d_point_cloud(src_points)
    axes = make_open3d_axes(scale=0.1)

    if true_mask is not None:
        # true_mask=true_mask.cpu().numpy()
        ture_src_corr_points = src_corr_points[true_mask]
        ture_ref_corr_points = ref_corr_points[true_mask]

        false_src_corr_points = src_corr_points[~true_mask]
        false_ref_corr_points = ref_corr_points[~true_mask]

        t_idx=np.arange(ture_src_corr_points.shape[0]).reshape(-1,1)
        t_point_correspondences=np.concatenate([t_idx,t_idx],axis=1)

        n_idx=np.arange(false_src_corr_points.shape[0]).reshape(-1,1)
        n_point_correspondences=np.concatenate([n_idx,n_idx],axis=1)

    
        t_corr_lines = make_open3d_corr_lines2(ture_ref_corr_points, ture_src_corr_points, t_point_correspondences, 'true')
        n_corr_lines = make_open3d_corr_lines2(false_ref_corr_points, false_src_corr_points, n_point_correspondences, 'false')


        # corr_lines = make_mesh_corr_lines(torch.tensor(ref_corr_points), torch.tensor(src_corr_points),[0,1,0],0.1)
        # o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors),  axes])
        o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors), t_corr_lines, n_corr_lines, axes])

    else:
        idx=np.arange(ref_corr_points.shape[0]).reshape(-1,1)
        point_correspondences=np.concatenate([idx,idx],axis=1)

    
        corr_lines = make_open3d_corr_lines2(ref_corr_points, src_corr_points, point_correspondences)
        # corr_lines = make_mesh_corr_lines(torch.tensor(ref_corr_points), torch.tensor(src_corr_points),[0,1,0],0.1)
        # o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors),  axes])
        o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors), corr_lines, axes])

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
