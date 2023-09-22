import random, re
import torch
import numpy as np
import time
import open3d

def square_distance(src, tgt, normalize=False):
    '''
    Calculate Euclide distance between every two points
    :param src: source point cloud in shape [B, N, C]
    :param tgt: target point cloud in shape [B, M, C]
    :param normalize: whether to normalize calculated distances
    :return:
    '''

    B, N, _ = src.shape
    _, M, _ = tgt.shape
    dist = -2. * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    if normalize:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
        dist += torch.sum(tgt ** 2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist
    
def point2node(nodes, points):
    '''
    Assign each point to a certain node according to nearest neighbor search
    :param nodes: [M, 3]
    :param points: [N, 3]
    :return: idx [N], indicating the id of node that each point belongs to
    '''
    M, _ = nodes.size()
    N, _ = points.size()
    dist = square_distance(points.unsqueeze(0), nodes.unsqueeze(0))[0]

    idx = dist.topk(k=1, dim=-1, largest=False)[1] #[B, N, 1], ignore the smallest element as it's the query itself

    idx = idx.squeeze(-1)
    return idx


def point2node_correspondences(src_nodes, src_points, tgt_nodes, tgt_points, point_correspondences, thres=0., device='cpu'):
    '''
    Based on point correspondences & point2node relationships, calculate node correspondences
    :param src_nodes: Nodes of source point cloud
    :param src_points: Points of source point cloud
    :param tgt_nodes: Nodes of target point cloud
    :param tgt_points: Points of target point cloud
    :param point_correspondences: Ground truth point correspondences
    :return: node_corr_mask: Overlap ratios between nodes
             node_corr: Node correspondences sampled for training
    '''
    #####################################
    # calc visible ratio for each node
    src_visible, tgt_visible = point_correspondences[:, 0], point_correspondences[:, 1]

    src_vis, tgt_vis = torch.zeros((src_points.shape[0])).to(device), torch.zeros((tgt_points.shape[0])).to(device)

    src_vis[src_visible] = 1.
    tgt_vis[tgt_visible] = 1.

    src_vis = src_vis.nonzero().squeeze(1)
    tgt_vis = tgt_vis.nonzero().squeeze(1)

    src_vis_num = torch.zeros((src_nodes.shape[0])).to(device)
    src_tot_num = torch.ones((src_nodes.shape[0])).to(device)

    src_idx = point2node(src_nodes, src_points)
    idx, cts = torch.unique(src_idx, return_counts=True)
    src_tot_num[idx] = cts.float()

    src_idx_ = src_idx[src_vis]
    idx_, cts_ = torch.unique(src_idx_, return_counts=True)
    src_vis_num[idx_] = cts_.float()

    src_node_vis = src_vis_num / src_tot_num

    tgt_vis_num = torch.zeros((tgt_nodes.shape[0])).to(device)
    tgt_tot_num = torch.ones((tgt_nodes.shape[0])).to(device)


    tgt_idx = point2node(tgt_nodes, tgt_points)
    idx, cts = torch.unique(tgt_idx, return_counts=True)
    tgt_tot_num[idx] = cts.float()

    tgt_idx_ = tgt_idx[tgt_vis]
    idx_, cts_ = torch.unique(tgt_idx_, return_counts=True)
    tgt_vis_num[idx_] = cts_.float()

    tgt_node_vis = tgt_vis_num / tgt_tot_num

    src_corr = point_correspondences[:, 0]  # [K]
    tgt_corr = point_correspondences[:, 1]  # [K]

    src_node_corr = torch.gather(src_idx, 0, src_corr)
    tgt_node_corr = torch.gather(tgt_idx, 0, tgt_corr)

    index = src_node_corr * tgt_idx.shape[0] + tgt_node_corr

    index, counts = torch.unique(index, return_counts=True)


    src_node_corr = index // tgt_idx.shape[0]
    tgt_node_corr = index % tgt_idx.shape[0]

    node_correspondences = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32).to(device)

    node_corr_mask = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32).to(device)
    node_correspondences[src_node_corr, tgt_node_corr] = counts.float()
    node_correspondences = node_correspondences[:-1, :-1]

    node_corr_sum_row = torch.sum(node_correspondences, dim=1, keepdim=True)
    node_corr_sum_col = torch.sum(node_correspondences, dim=0, keepdim=True)

    node_corr_norm_row = (node_correspondences / (node_corr_sum_row + 1e-10)) * src_node_vis.unsqueeze(1).expand(src_nodes.shape[0], tgt_nodes.shape[0])

    node_corr_norm_col = (node_correspondences / (node_corr_sum_col + 1e-10)) * tgt_node_vis.unsqueeze(0).expand(src_nodes.shape[0], tgt_nodes.shape[0])

    node_corr_mask[:-1, :-1] = torch.min(node_corr_norm_row, node_corr_norm_col)
    ############################################################
    # Binary masks
    #node_corr_mask[:-1, :-1] = (node_corr_mask[:-1, :-1] > 0.01)
    #node_corr_mask[-1, :-1] = torch.clamp(1. - torch.sum(node_corr_mask[:-1, :-1], dim=0), min=0.)
    #node_corr_mask[:-1, -1] = torch.clamp(1. - torch.sum(node_corr_mask[:-1, :-1], dim=1), min=0.)
    re_mask = (node_corr_mask[:-1, :-1] > thres)
    node_corr_mask[:-1, :-1] = node_corr_mask[:-1, :-1] * re_mask

    #####################################################
    # Soft weighted mask, best Performance
    node_corr_mask[:-1, -1] = 1. - src_node_vis
    node_corr_mask[-1, :-1] = 1. - tgt_node_vis
    #####################################################

    node_corr = node_corr_mask[:-1, :-1].nonzero()
    return node_corr_mask, node_corr
