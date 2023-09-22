import torch
from torch import nn

from geotransformer.modules.ops.radius_search import radius_search

class NMS(nn.Module):
    def __init__(self, cfgs, neighbor_limits):
        super().__init__()
        self.NMS_radius = cfgs.NMS_radius
        # self.pdis = nn.PairwiseDistance(p=2)
        self.neighbor_limits = neighbor_limits[-1]

    @torch.no_grad()
    def forward(self, nodes_dict, length_dict=None, overlap_score=None, features=None):
        '''
        :param nodes: list of node sets, each of shape Mx3
        :param length_dict: list of node length
        :param overlap_score: M or None
        :param features: Mx256 or None
        :return: stacks of masks for each node set
        '''
        if length_dict is None:
            length_dict=[nodes_dict.shape[0]]
        node_knn_indices = radius_search(
            nodes_dict.cpu(),
            nodes_dict.cpu(),
            length_dict.cpu(),
            length_dict.cpu(),
            self.NMS_radius,
            self.neighbor_limits,
        )

        length_increment = 0
        masks = torch.zeros(node_knn_indices.shape[0]+1, dtype=torch.bool).cuda()
        nms_length =[]
        for i in range(node_knn_indices.shape[0]):
            if masks[node_knn_indices[i]].sum()==0:
                masks[i]=True

        return masks[:-1]

    
class Vote_layer(nn.Module):
    """ Light voting module with limitation"""
    def __init__(self, cfgs, r):
        super(Vote_layer, self).__init__()

        mlp_list = cfgs.MLPS
        max_translate_range = cfgs.MAX_TRANSLATE_RANGE
        pre_channel = cfgs.input_feats_dim

        if len(mlp_list) > 0:
            shared_mlps = []
            for i in range(len(mlp_list)):
                
                shared_mlps.extend([
                    nn.Linear(pre_channel, mlp_list[i]),
                    nn.LayerNorm(mlp_list[i]),
                    nn.ReLU()
                ])
                pre_channel = mlp_list[i]
            self.mlp_modules = nn.Sequential(*shared_mlps)
        else:
            self.mlp_modules = None

        self.ctr_reg = nn.Linear(pre_channel, 3+cfgs.input_feats_dim)
        self.max_offset_limit = torch.tensor(max_translate_range).float()/r if max_translate_range is not None else None

        out_proj = []
        out_proj.extend([
            nn.LayerNorm(cfgs.input_feats_dim),
            # nn.Linear(cfgs.input_feats_dim, cfgs.input_feats_dim)
        ])
        self.out_proj = nn.Sequential(*out_proj)

       

    def forward(self, xyz, features, aug_rotation=None):
        if len(xyz.shape)<3:
            xyz = xyz.unsqueeze(0)
        if len(features.shape)<3:
            features = features.unsqueeze(0)

        xyz_select = xyz
        features_select = features

        if self.mlp_modules is not None: 
            new_features = self.mlp_modules(features_select) #([4, 256, 256]) ->([4, 128, 256])            
        else:
            new_features = features_select
        
        ctr_offsets = self.ctr_reg(new_features) #[4, 128, 256]) -> ([4, 3, 256])

        ctr_offsets = ctr_offsets#([4, 256, 3+input_feats_dim])
        feat_offets = ctr_offsets[..., 3:]
        ctr_offsets = ctr_offsets[..., :3]
        
        if self.max_offset_limit is not None:
            # if aug_rotation is not None:
            #    self.max_offset_limit = torch.matmul(self.max_offset_limit, aug_rotation.detach().cpu().T)
           
            max_offset_limit = self.max_offset_limit.view(1, 1, 3)            
            max_offset_limit = self.max_offset_limit.repeat((xyz_select.shape[0], xyz_select.shape[1], 1)).to(xyz_select.device) #([4, 256, 3])
      
            limited_ctr_offsets = torch.where(ctr_offsets > max_offset_limit, max_offset_limit, ctr_offsets)
            min_offset_limit = -1 * max_offset_limit
            limited_ctr_offsets = torch.where(limited_ctr_offsets < min_offset_limit, min_offset_limit, limited_ctr_offsets)
            vote_xyz = xyz_select + limited_ctr_offsets
        else:
            vote_xyz = xyz_select + ctr_offsets

        new_features = self.out_proj(features_select + feat_offets)

        # new_features = features_select + feat_offets

        # return vote_xyz.squeeze(0), new_features.squeeze(0), xyz_select, ctr_offsets
        return vote_xyz.squeeze(0), new_features.squeeze(0)
