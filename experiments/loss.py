from numpy import zeros_like
import torch
import torch.nn as nn

from geotransformer.modules.ops import apply_transform, pairwise_distance
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.utils.registration import get_correspondences

class SingleSideChamferLoss_Brute(nn.Module):
    def __init__(self):
        super(SingleSideChamferLoss_Brute, self).__init__()

    def forward(self, output_dict):
        '''
        :param pc_src_input: Bx3xM Variable in GPU
        :param pc_dst_input: Bx3xN Variable in GPU
        :return:
        '''

        ref_node = output_dict['shifted_ref_points_c']
        src_node = output_dict['shifted_src_points_c']
        ref_points_f = output_dict['ref_points_f']
        src_points_f = output_dict['src_points_f']

        ref_on_pc_dsit_mat = torch.sqrt(pairwise_distance(ref_node, ref_points_f, normalized=False))
        ref_min_dist, _ = torch.min(ref_on_pc_dsit_mat, dim=1, keepdim=False)  # BxM

        src_on_pc_dsit_mat = torch.sqrt(pairwise_distance(src_node, src_points_f, normalized=False))
        src_min_dist, _ = torch.min(src_on_pc_dsit_mat, dim=1, keepdim=False)  # BxM

        loss = (ref_min_dist.mean()+src_min_dist.mean())/2

        return loss

class VoteLoss(nn.Module):
    def __init__(self, cfg):
        super(VoteLoss, self).__init__()
        self.n2n_overlap_threshold = cfg.n2n_overlap_threshold
        self.n2p_overlap_threshold = cfg.n2p_overlap_threshold
        self.p2p_overlap_threshold = cfg.p2p_overlap_threshold
        self.BCE_loss = nn.BCELoss(reduction='none')

        self.NMS_radius = cfg.NMS_radius
    
    def get_weighted_bce_loss(self, prediction, gt):

        class_loss = self.BCE_loss(prediction, gt) 

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0) 
        w_positive = 1 - w_negative  
        
        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        return w_class_loss

    def forward(self, output_dict, data_dict):

        ref_node = output_dict['shifted_ref_points_c']
        src_node = output_dict['shifted_src_points_c']
        # ref_node2 = output_dict['shifted_ref_points_c2']
        # src_node2 = output_dict['shifted_src_points_c2']
        transform = data_dict['transform']
        n2n_ref_scores_c = output_dict['ref_n2n_scores_c']
        n2n_src_scores_c = output_dict['src_n2n_scores_c']
        device = n2n_src_scores_c.device

        src_node = apply_transform(src_node, transform)
        # src_node2 = apply_transform(src_node2, transform)
        dist_mat = torch.sqrt(pairwise_distance(ref_node, src_node, normalized=False))
        # dist_mat2 = torch.sqrt(pairwise_distance(ref_node2, src_node2, normalized=False))
        # points_dist_mat = torch.sqrt(pairwise_distance(ref_points_f, src_points_f, normalized=False))

        ###################
        # chamfer loss
        mask = output_dict['mask']

        src_dst_min_dist, _ = torch.min(dist_mat, dim=1, keepdim=False)  # BxM
        ref_mask = mask.sum(1)>0
        forward_loss = src_dst_min_dist[ref_mask].mean()
        dst_src_min_dist, _ = torch.min(dist_mat, dim=0, keepdim=False)  # BxN
        src_mask = mask.sum(0)>0
        backward_loss = dst_src_min_dist[src_mask].mean()

        chamfer_pure = forward_loss + backward_loss

        ###################
        # n2n overlap loss
        corr_indices = get_correspondences(ref_node.detach().cpu().numpy(), src_node.detach().cpu().numpy(), matching_radius = self.n2n_overlap_threshold)
        if len(corr_indices)==0:
            src_gt = torch.zeros(src_node.shape[0], device=device)
            ref_gt = torch.zeros(ref_node.shape[0], device=device)
        else:
            src_idx = corr_indices[:,1]
            ref_idx = corr_indices[:,0]

            src_gt = torch.zeros(src_node.shape[0], device=device)
            src_gt[src_idx]=1.
            ref_gt = torch.zeros(ref_node.shape[0], device=device)
            ref_gt[ref_idx]=1.

        gt_labels = torch.cat((src_gt, ref_gt))
        scores_overlap = torch.cat((n2n_src_scores_c, n2n_ref_scores_c))
        n2n_overlap_loss = self.get_weighted_bce_loss(scores_overlap, gt_labels)


        return chamfer_pure, n2n_overlap_loss

class OverlapLoss(nn.Module):
    def __init__(self, cfg):
        super(OverlapLoss, self).__init__()
        self.n2p_overlap_threshold = cfg.n2p_overlap_threshold
        self.p2p_overlap_threshold = cfg.p2p_overlap_threshold
        self.BCE_loss = nn.BCELoss(reduction='none')
    
    def get_weighted_bce_loss(self, prediction, gt):

        class_loss = self.BCE_loss(prediction, gt) 

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0) 
        w_positive = 1 - w_negative  
        
        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        return w_class_loss

    def forward(self, output_dict, data_dict):

        ref_points_c = output_dict['ori_ref_points_c']
        src_points_c = output_dict['ori_src_points_c']
        ref_points_f = output_dict['ref_points_f']
        src_points_f = output_dict['src_points_f']
        transform = data_dict['transform']
        p2p_ref_scores_c = output_dict['ref_p2p_scores_c']
        p2p_src_scores_c = output_dict['src_p2p_scores_c']
        n2p_ref_scores_c = output_dict['ref_n2p_scores_c']
        n2p_src_scores_c = output_dict['src_n2p_scores_c']
        device = n2p_src_scores_c.device

        src_points_f = apply_transform(src_points_f, transform)
        src_points_c = apply_transform(src_points_c, transform)

        ###################
        # p2p overlap loss
        corr_indices = get_correspondences(ref_points_f.detach().cpu().numpy(), src_points_f.detach().cpu().numpy(), matching_radius = self.p2p_overlap_threshold)
        src_idx = corr_indices[:,1]
        ref_idx = corr_indices[:,0]

        src_gt = torch.zeros(src_points_f.shape[0], device=device)
        src_gt[src_idx]=1.
        ref_gt = torch.zeros(ref_points_f.shape[0], device=device)
        ref_gt[ref_idx]=1.

        gt_labels = torch.cat((src_gt, ref_gt))
        scores_overlap = torch.cat((p2p_src_scores_c, p2p_ref_scores_c))
        p2p_overlap_loss = self.get_weighted_bce_loss(scores_overlap, gt_labels)

        ###################
        # n2p overlap loss
        dsit_mat = torch.sqrt(pairwise_distance(ref_points_c.double(), src_points_f.double(), normalized=False))
        ref_src_n2p_min_dist, _ = torch.min(dsit_mat, dim=1, keepdim=False)
        ref_gt = torch.zeros(ref_src_n2p_min_dist.shape[0], device=device)
        ref_gt[ref_src_n2p_min_dist<self.n2p_overlap_threshold] = 1
        
        dsit_mat = torch.sqrt(pairwise_distance(src_points_c.double(), ref_points_f.double(), normalized=False))
        src_ref_n2p_min_dist, _ = torch.min(dsit_mat, dim=1, keepdim=False)
        src_gt = torch.zeros(src_ref_n2p_min_dist.shape[0], device=device)
        src_gt[src_ref_n2p_min_dist<self.n2p_overlap_threshold] = 1

        scores_overlap = torch.cat((n2p_src_scores_c, n2p_ref_scores_c))
        gt_labels = torch.cat((src_gt, ref_gt))
        n2p_overlap_loss = self.get_weighted_bce_loss(scores_overlap, gt_labels)

        return n2p_overlap_loss, p2p_overlap_loss

class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class gap_loss(nn.Module):
    def __init__(self, cfg):
        super(gap_loss, self).__init__()
        self.triplet_loss_gamma = cfg.gap_loss.triplet_loss_gamma
        self.positive_radius = cfg.gap_loss.positive_radius
        # self.lamda = lamda

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        b, n, m = matching_scores.size()

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))

        ref_dist, ref_mindis_indic = dists.min(-1)
        src_dist, src_mindis_indic = dists.min(-2)
        ref_idx = torch.arange(dists.shape[1]).view(1, n-1).expand(b, n-1)
        src_idx = torch.arange(dists.shape[2]).view(1, m-1).expand(b, m-1)
        ref_batch_idx = torch.arange(dists.shape[0]).view(b, 1).expand(b, n-1)
        src_batch_idx = torch.arange(dists.shape[0]).view(b, 1).expand(b, m-1)

        ref_gt_corr_map = torch.zeros_like(dists, dtype=bool)
        ref_gt_corr_mask = torch.lt(ref_dist, self.positive_radius ** 2)
        ref_gt_corr_map[ref_batch_idx[ref_gt_corr_mask], ref_idx[ref_gt_corr_mask], ref_mindis_indic[ref_gt_corr_mask]] = True
        ref_gt_corr_map = torch.logical_and(ref_gt_corr_map, gt_masks)
        slack_row_labels = torch.eq(ref_gt_corr_map.sum(2), 0)
        ref_labels = torch.zeros_like(matching_scores[:,:-1,:], dtype=torch.bool)
        ref_labels[:, :, :-1] = ref_gt_corr_map
        ref_labels[:, :, -1] = slack_row_labels

        src_gt_corr_map = torch.zeros_like(dists, dtype=bool)
        src_gt_corr_mask = torch.lt(src_dist, self.positive_radius ** 2)
        src_gt_corr_map[src_batch_idx[src_gt_corr_mask], src_mindis_indic[src_gt_corr_mask], src_idx[src_gt_corr_mask]] = True
        src_gt_corr_map = torch.logical_and(src_gt_corr_map, gt_masks)
        slack_col_labels = torch.eq(src_gt_corr_map.sum(1), 0)
        src_labels = torch.zeros_like(matching_scores[:,:,:-1], dtype=torch.bool)
        src_labels[:, :-1, :] = src_gt_corr_map
        src_labels[:, -1, :] = slack_col_labels

        
        '''pc0 -> pc1'''   
        score_anc_pos = -matching_scores[:,:-1,:][ref_labels].view(b,n-1,1)
        score_anc_neg = -matching_scores[:,:-1,:][~ref_labels].view(b,n-1,m-1)
        gap = score_anc_pos - score_anc_neg # score gap between true and false
        gap = gap[~(score_anc_pos.squeeze(-1)==1e12)]

        '''pc1 -> pc0'''
        score_anc_pos2 = -matching_scores[:,:,:-1][src_labels].view(b,1,m-1)
        score_anc_neg2 = -matching_scores[:,:,:-1][~src_labels].view(b,n-1,m-1)
        gap2 = score_anc_pos2 - score_anc_neg2 # score gap between true and false
        gap2 = gap2.transpose(1,2)[~(score_anc_pos2.squeeze(-2)==1e12)] 

        '''gap loss'''
        # before_clamp_loss = margin[non_match0[:,:,None].repeat(1,1,m) == False].view(b,-1,m) # margin removing the non-matches
        active_num = torch.sum((gap + self.triplet_loss_gamma > 0).float(), dim=1, keepdim=False)
        active_num[active_num==0]=1
        gap_loss = torch.clamp(gap + self.triplet_loss_gamma, min=0)
        gap_loss = torch.mean(torch.log(torch.sum(gap_loss, dim=1)+1))
        # gap_loss[gap<-1e10] = -1e12
        # gap_loss = torch.clamp(torch.logsumexp(gap_loss,dim=1),min=0).mean()
        
        # before_clamp_loss2 = margin2[non_match1[:,None].repeat(1,n,1) == False].view(b,n,-1) # margin removing the non-matches
        active_num2 = torch.sum((gap2 > -self.triplet_loss_gamma).float(), dim=1, keepdim=False)
        active_num2[active_num2==0]=1
        gap_loss2 = torch.clamp(gap2 + self.triplet_loss_gamma, min=0)
        gap_loss2 = torch.mean(torch.log(torch.sum(gap_loss2, dim=1)+1))
        # gap_loss2[gap2<-1e10] = -1e12
        # gap_loss2 = torch.clamp(torch.logsumexp(gap_loss2,dim=1),min=0).mean()

        gap_loss = (gap_loss+gap_loss2)/2

        return gap_loss

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.overlap = OverlapLoss(cfg.Vote)
        self.gap_loss = gap_loss(cfg)
        self.vote_loss = VoteLoss(cfg.Vote)
        self.node_on_pc_loss = SingleSideChamferLoss_Brute()
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_vote_loss = cfg.loss.weight_vote_loss
        self.weight_gap_loss = cfg.loss.weight_gap_loss

    def forward(self, output_dict, data_dict):
        loss_all={}
        '''coarse match loss'''
        coarse_loss = self.coarse_loss(output_dict)

        '''fine match loss'''
        gap_loss = self.gap_loss(output_dict, data_dict)
        n_loss, p_loss = self.overlap(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_gap_loss * gap_loss + n_loss + p_loss
        loss_all['c_loss'] = coarse_loss
        loss_all['g_loss'] = gap_loss
        loss_all['n_loss'] = n_loss
        loss_all['p_loss'] = p_loss

        '''superpoint detection loss'''
        vote_loss, nn_overlap_loss = self.vote_loss(output_dict, data_dict)
        node_on_pc_loss = self.node_on_pc_loss(output_dict)

        loss = loss + (vote_loss+node_on_pc_loss)*self.weight_vote_loss + nn_overlap_loss

        loss_all['v_loss'] = vote_loss
        loss_all['nn_loss'] = nn_overlap_loss
        loss_all['d_loss'] = node_on_pc_loss
        
        loss_all['loss'] = loss

        return loss_all




class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.rre_threshold = cfg.eval.rre_threshold
        self.rte_threshold = cfg.eval.rte_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(size=(ref_length_c, src_length_c)).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        rre, rte = isotropic_transform_error(transform, est_transform)
        recall = torch.logical_and(torch.lt(rre, self.rre_threshold), torch.lt(rte, self.rte_threshold)).float()
        return rre, rte, recall

    def forward(self, output_dict, data_dict):
        result={}
        c_precision = self.evaluate_coarse(output_dict)
        result['PIR'] = c_precision
        if data_dict['evaling']:
            f_precision = self.evaluate_fine(output_dict, data_dict)
            rre, rte, recall = self.evaluate_registration(output_dict, data_dict)
            result['IR'] = f_precision
            result['RRE'] = rre
            result['RTE'] = rte
            result['RR'] = recall
        return result
