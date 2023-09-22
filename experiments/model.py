import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences, get_node_correspondences_disance, get_node_overlap
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    # GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)


from rdmnet.thdroformer import ThDRoFormer
from rdmnet.vote import Vote_layer, NMS
from rdmnet.utils.visualization import vis_shifte_node, visualization, vis_node_grouping

from backbone import Encoder as Encoder5 
from backbone import Decoder as Decoder3

from time import time

class RDMNet(nn.Module):
    def __init__(self, cfg):
        super(RDMNet, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.corres_radius = cfg.model.ground_truth_corres_radius
        self.n2p_score_threshold = cfg.model.n2p_score_threshold
        self.p2p_score_threshold = cfg.model.p2p_score_threshold


        self.encoder = Encoder5(
            cfg.backbone.input_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )
        self.decoder = Decoder3(
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.group_norm,
        )
        self.transformer = ThDRoFormer(
            cfg.thdroformer.input_dim,
            cfg.thdroformer.output_dim,
            cfg.thdroformer.hidden_dim,
            cfg.thdroformer.num_heads,
            cfg.thdroformer.num_layers,
        )
        
        

        self.use_vote=cfg.Vote.inference_use_vote and cfg.Vote.model_use_vote
        if cfg.Vote.model_use_vote:
            cfg.Vote.input_feats_dim = cfg.thdroformer.output_dim
            self.vote = Vote_layer(cfg.Vote, 1)
            self.nms = NMS(cfg.Vote, cfg.neighbor_limits)

            self.proj_n2n_score = nn.Linear(cfg.thdroformer.output_dim,1)
            self.sigmoid = nn.Sigmoid()

            self.transformer2 = ThDRoFormer(
                cfg.thdroformer.input_dim2,
                cfg.thdroformer.output_dim,
                cfg.thdroformer.hidden_dim,
                cfg.thdroformer.num_heads,
                cfg.thdroformer.num_layers2,
                cfg.thdroformer.k2
            )
        self.proj_n2p_score = nn.Linear(cfg.thdroformer.output_dim,1)
        self.sigmoid2 = nn.Sigmoid()

        

            

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization, cfg.model.n2p_score_threshold
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )
        

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)
        
        self.vis = cfg.test.vis


    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        # ref_length_f = data_dict['lengths'][0][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        # points_f = data_dict['points'][0].detach()
        points = data_dict['points'][0].detach()

        testing = data_dict['testing']

        ori_ref_points_c = points_c[:ref_length_c]
        ori_src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ori_ref_points_c'] = ori_ref_points_c
        output_dict['ori_src_points_c'] = ori_src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # # for visualization
        # ref_points_raw = data_dict['ref_points_raw']
        # src_points_raw = data_dict['src_points_raw']

        # #########################################
        # #  backbone
        feats_list = self.encoder(feats, data_dict)

        feats_c = feats_list[-1]
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]

        # 3. Transformer
        ref_feats_c, src_feats_c = self.transformer(
            ori_ref_points_c.unsqueeze(0),
            ori_src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_n2p_f = self.proj_n2p_score(ref_feats_c)
        src_n2p_f = self.proj_n2p_score(src_feats_c)
        ref_n2p_scores_c = torch.clamp(self.sigmoid2(ref_n2p_f.view(-1)),min=0,max=1)
        src_n2p_scores_c = torch.clamp(self.sigmoid2(src_n2p_f.view(-1)),min=0,max=1)

        ref_feats_o_c = torch.cat([ref_feats_c,ref_n2p_f], dim=2)
        src_feats_o_c = torch.cat([src_feats_c,src_n2p_f], dim=2)
        feats_list[-1] = torch.cat([ref_feats_o_c.squeeze(0),src_feats_o_c.squeeze(0)],dim=0)
        feats_list = self.decoder(feats_list, data_dict)
        feats_f = feats_list[0][:,:-1]

        ref_p2p_f = feats_list[0][:,-1][:ref_length_f]
        src_p2p_f = feats_list[0][:,-1][ref_length_f:]
        ref_p2p_scores_c = torch.clamp(self.sigmoid2(ref_p2p_f.view(-1)),min=0,max=1)
        src_p2p_scores_c = torch.clamp(self.sigmoid2(src_p2p_f.view(-1)),min=0,max=1)
        output_dict['ref_n2p_scores_c'] = ref_n2p_scores_c
        output_dict['src_n2p_scores_c'] = src_n2p_scores_c
        output_dict['ref_p2p_scores_c'] = ref_p2p_scores_c
        output_dict['src_p2p_scores_c'] = src_p2p_scores_c

        if self.use_vote:
        # if False:
            if not testing:
                # for loss calculating
                mask = get_node_correspondences_disance(
                    ori_ref_points_c,
                    ori_src_points_c,
                    transform,
                    self.corres_radius
                )
                output_dict['mask'] = mask
            
            #####################################
            # 3.2. vote layer
            feats_c = torch.cat([ref_feats_c.squeeze(0), src_feats_c.squeeze(0)], dim=0)
            # shifted_src_points_c, src_feats_c = self.vote(src_points_c, src_feats_c)
            # shifted_ref_points_c, ref_feats_c = self.vote(ref_points_c, ref_feats_c)
            shifted_points_c, feats_c = self.vote(points_c, feats_c)



            shifted_ref_points_c = shifted_points_c[:ref_length_c]
            shifted_src_points_c = shifted_points_c[ref_length_c:]
            ref_feats_c = feats_c[:ref_length_c]
            src_feats_c = feats_c[ref_length_c:]
            output_dict['shifted_src_points_c'] = shifted_src_points_c
            output_dict['shifted_ref_points_c'] = shifted_ref_points_c


            ref_n2n_scores_c = self.proj_n2n_score(ref_feats_c)
            src_n2n_scores_c = self.proj_n2n_score(src_feats_c)
            ref_n2n_scores_c = torch.clamp(self.sigmoid(ref_n2n_scores_c.view(-1)),min=0,max=1)
            src_n2n_scores_c = torch.clamp(self.sigmoid(src_n2n_scores_c.view(-1)),min=0,max=1)

            if not testing:
                # for loss calculating
                output_dict['ref_n2n_scores_c'] = ref_n2n_scores_c
                output_dict['src_n2n_scores_c'] = src_n2n_scores_c

            masks = self.nms(shifted_points_c, data_dict['lengths'][-1])
            ref_masks = masks[:data_dict['lengths'][-1][0]].cuda()
            src_masks = masks[data_dict['lengths'][-1][0]:].cuda()
            
            '''visualize node voting'''
            if self.vis and testing:
                vis_shifte_node(ori_src_points_c, ori_ref_points_c, shifted_src_points_c, shifted_ref_points_c, src_points_f, ref_points_f, data_dict['transform'],src_masks,ref_masks,
                    color=[1,0.5,0],
                    src_node_color = [0.2, 1, 0.2],
                    ref_node_color = [0.2, 0.2, 1],
                    src_point_color = [0.1, 0.6, 0.1],
                    ref_point_color = [0.1, 0.1, 0.6],
                    )

            src_points_c = shifted_src_points_c[src_masks]
            ref_points_c = shifted_ref_points_c[ref_masks]
            src_feats_c = src_feats_c[src_masks]
            ref_feats_c = ref_feats_c[ref_masks]

            if testing:
                src_n2p_scores_c = src_n2p_scores_c[src_masks]
                ref_n2p_scores_c = ref_n2p_scores_c[ref_masks]
                src_n2n_scores_c = src_n2n_scores_c[src_masks]
                ref_n2n_scores_c = ref_n2n_scores_c[ref_masks]
                output_dict['src_n2p_scores_c'] = src_n2p_scores_c
                output_dict['ref_n2p_scores_c'] = ref_n2p_scores_c
                output_dict['src_n2n_scores_c'] = src_n2n_scores_c
                output_dict['ref_n2n_scores_c'] = ref_n2n_scores_c

            ref_feats_c = ref_feats_c.unsqueeze(0)
            src_feats_c = src_feats_c.unsqueeze(0)

            ref_feats_c, src_feats_c = self.transformer2(
                ref_points_c.unsqueeze(0),
                src_points_c.unsqueeze(0),
                ref_feats_c,
                src_feats_c,
            )

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
     
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        '''visualize point_to_node grouping '''
        if self.vis and testing:
            vis_node_grouping(src_points_c, ori_src_points_c, src_points_f)

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        
        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                # ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks, ref_n2p_scores_c, src_n2p_scores_c
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks


        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        if not self.training:
            with torch.no_grad():
                if not self.fine_matching.use_dustbin:
                    matching_scores = matching_scores[:, :-1, :-1]

                ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                    ref_node_corr_knn_points,
                    src_node_corr_knn_points,
                    ref_node_corr_knn_masks,
                    src_node_corr_knn_masks,
                    matching_scores,
                    node_corr_scores,
                )

                output_dict['ref_corr_points'] = ref_corr_points
                output_dict['src_corr_points'] = src_corr_points
                output_dict['corr_scores'] = corr_scores
                output_dict['estimated_transform'] = estimated_transform

        '''visualize point correspondences'''
        if self.vis and testing:
            visualization(
                output_dict,
                transform,
                src_node_knn_indices,
                src_node_knn_masks,
                ref_node_knn_indices,
                ref_node_knn_masks,

                src_node_color = [0.2, 1, 0.2],
                ref_node_color = [0.2, 0.2, 1],
                src_point_color = [0.1, 0.6, 0.1],
                ref_point_color = [0.1, 0.1, 0.6],
                offsets=(0, 0, -30),
                find_true=True
            )

        return output_dict


def create_model(cfg):
    model = RDMNet(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
