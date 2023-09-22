import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport

class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True, n2p_score_threshold=None):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization
        self.n2p_score_threshold = n2p_score_threshold

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None, ref_n2p_scores_c=None, src_n2p_scores_c=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        # select top-k proposals
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        
        ######################
        if ref_n2p_scores_c is not None:
            if self.n2p_score_threshold is None:
                raise ValueError(' "n2p_scores_c" is given but "n2p_score_threshold" is not set. ')
            ref_n2p_scores_c = ref_n2p_scores_c[ref_indices]
            src_n2p_scores_c = src_n2p_scores_c[src_indices]
            ref_overlap_mask = ref_n2p_scores_c > self.n2p_score_threshold
            src_overlap_mask = src_n2p_scores_c > self.n2p_score_threshold
            overlap_mask = torch.logical_and(ref_overlap_mask.unsqueeze(1),src_overlap_mask.unsqueeze(0))
            matching_scores[~overlap_mask] = 0

        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        ######################
        # superpoint match using overlap score
        # if ref_n2p_scores_c is not None:
        #     ref_n2p_scores_c = ref_n2p_scores_c[ref_indices]
        #     src_n2p_scores_c = src_n2p_scores_c[src_indices]
        #     # ref_n2p_scores_c = ref_n2p_scores_c/ref_n2p_scores_c.sum()
        #     # src_n2p_scores_c = src_n2p_scores_c/src_n2p_scores_c.sum()
        #     ref_overlap_mask = ref_n2p_scores_c > 0.5
        #     src_overlap_mask = src_n2p_scores_c > 0.5
        #     overlap_mask = torch.logical_and(ref_overlap_mask.unsqueeze(1),src_overlap_mask.unsqueeze(0))
        #     matching_scores[~overlap_mask] = 0
        # max_score = matching_scores.max(-1)
        # ori_src_indice = max_score.indices[(max_score.values>0)]
        # ori_ref_indice = torch.nonzero(max_score.values>0, as_tuple=True)[0]
        # num_correspondences = min(self.num_correspondences, ori_src_indice.shape[0])
        # corr_scores, top_indice = max_score.values[(max_score.values>0)].topk(k=num_correspondences, largest=True)
        # src_corr_indices = ori_src_indice[top_indice]
        # ref_corr_indices = ori_ref_indice[top_indice]

        return ref_corr_indices, src_corr_indices, corr_scores
