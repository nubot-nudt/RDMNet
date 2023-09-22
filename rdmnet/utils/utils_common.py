# import logging
# import os
# import pickle
# import random
# import shutil
# import subprocess

import numpy as np
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
import open3d as o3d

from pathlib import Path
from utils.logger import Logger
import os
from tensorboardX import SummaryWriter
import time

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def config(cfgs, mode):
    model_cfg = cfgs.MODEL.GAT

    # 创建模型输出路径
    if cfgs.MODEL.Vote.use_vote:
        vote = 'usevote'
    else:
        vote = None
    model_name = 'layer{}.{}.{}.{}' .format(
        cfgs.MODEL.KPConv.num_layers, model_cfg.posencoder, cfgs.LOSS.method, vote)
    model_out_path = '{}/{}/kpconv/{}' .format(cfgs.DATA.model_out_path, cfgs.DATA.dataset, model_name)

    ################## 
    # log
    log_path = '{}/logs'.format(model_out_path)
    log_path = Path(log_path)
    log_path.mkdir(exist_ok=True, parents=True)
    log_file = os.path.join(log_path, '{}-{}.log'.format(mode, time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file, local_rank=cfgs.train.local_rank)

    ################## 
    # event
    event_path = '{}/events'.format(model_out_path)
    event_path = Path(event_path)
    event_path.mkdir(exist_ok=True, parents=True)
    event = SummaryWriter(event_path)



    # if cfgs.LOSS.method == 'fe_loss':
    #     Loss = fe_loss(cfgs.LOSS)
    # elif cfgs.LOSS.method == 'distribution_overlap':
    #     Loss = distribution_overlap(cfgs.LOSS)
    # elif cfgs.LOSS.method == 'fe_loss2':
    #     Loss = fe_loss2(cfgs.LOSS)
    # elif cfgs.LOSS.method == 'fe_loss3':
    #     Loss = fe_loss3(cfgs.LOSS)
    Loss=None

    return model_out_path, logger, event, Loss
