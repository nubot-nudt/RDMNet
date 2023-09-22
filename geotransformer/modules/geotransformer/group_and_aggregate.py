import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device
from geotransformer.modules.ops import radius_search
from geotransformer.modules.ops import index_select

class Group_and_Aggregate(nn.Module):
    def __init__(self, radius, neighbor_limit):
        super(Group_and_Aggregate, self).__init__()
        self.radius = radius
        self.neighbor_limit = neighbor_limit

        shared_mlp = []
        shared_mlp.extend([
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        ])
        self.aggregation_layer = nn.Sequential(*shared_mlp)

    def forward(self, shifted_src_points_c, src_masks, src_feats_c):
        r"""
        """
        device = src_feats_c.device
        q_nodes = shifted_src_points_c[src_masks]
        s_nodes = shifted_src_points_c
        neighbor_indices = radius_search(q_nodes.cpu(),s_nodes.cpu(),torch.LongTensor([q_nodes.shape[0]]),torch.LongTensor([s_nodes.shape[0]]),self.radius,self.neighbor_limit)
        neighbor_indices[neighbor_indices==src_feats_c.shape[0]] = 0
        group_feature = index_select(src_feats_c.cpu(),neighbor_indices,0)
        group_feature = group_feature.transpose(0,2).transpose(1,2)
        new_features = F.max_pool2d(
                        group_feature, kernel_size=[1, group_feature.size(2)]
                    )  # (B, mlp[-1], npoint, 1)

        new_features = new_features.transpose(0,1).squeeze(-1).to(device)
        new_features = self.aggregation_layer(new_features)

        return new_features
