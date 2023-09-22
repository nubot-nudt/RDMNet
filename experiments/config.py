import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# random seed
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.snapshot_dir, 'logs')
_C.event_dir = osp.join(_C.snapshot_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)


_C.dataset = 'kitti'    # kitti kitti360 apollo mulran


# data
_C.data = edict()
_C.data.dataset_root = '/mnt/Mount2/Dataset/KITTI_odometry'   # change to your own path
_C.data.dataset_360_root = '/mnt/Mount3/Dataset/KITTI-360'   # change to your own path
_C.data.mulran_root = '/mnt/Mount3/Dataset/mulran_process'   # change to your own path
_C.data.apollo_root = '/mnt/Mount3/Dataset/apollo'   # change to your own path

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.01
_C.train.augmentation_min_scale = 0.8
_C.train.augmentation_max_scale = 1.2
_C.train.augmentation_shift = 2.0
_C.train.augmentation_rotation = 1.0

# test config
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = None
_C.test.vis = True

# eval config
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.6
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rre_threshold = 5.0
_C.eval.rte_threshold = 2.0

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.3
_C.ransac.num_points = 4
_C.ransac.num_iterations = 50000

# optim config
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 4
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 160
_C.optim.grad_acc_steps = 1

'''Model'''
# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 5
_C.backbone.init_voxel_size = 0.3
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 4.25
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.6
_C.model.num_points_in_patch = 128
_C.model.num_sinkhorn_iterations = 100
_C.model.ground_truth_corres_radius = 2.4
_C.model.n2p_score_threshold = 0.1
_C.model.p2p_score_threshold = 0.1


# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

# model - Thdroformer
_C.thdroformer = edict()
_C.thdroformer.input_dim = 2048
_C.thdroformer.hidden_dim = 128
_C.thdroformer.output_dim = 256
_C.thdroformer.num_heads = 4
_C.thdroformer.num_layers = 4
_C.thdroformer.input_dim2 = 256
_C.thdroformer.num_layers2 = 4
_C.thdroformer.k2 = None

# model - Vote layer
_C.Vote = edict()
_C.Vote.model_use_vote = True
_C.Vote.inference_use_vote = True
_C.Vote.MAX_TRANSLATE_RANGE=[3.0, 3.0, 3.0]
_C.Vote.MLPS=[512, 256]
_C.Vote.NMS_radius=2.4
_C.Vote.n2n_overlap_threshold=1.2
_C.Vote.n2p_overlap_threshold=0.6
_C.Vote.p2p_overlap_threshold=0.6

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 2048
_C.geotransformer.hidden_dim = 128
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 4.8
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

'''LGR fine match'''
# model - Fine Matching using gap loss
_C.fine_matching = edict()
_C.fine_matching.acceptance_radius = 0.6
_C.fine_matching.mutual = False
_C.fine_matching.topk = 1
_C.fine_matching.confidence_threshold = 0
_C.fine_matching.use_dustbin = True
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5


'''loss'''
# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 40
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.gap_loss = edict()
_C.gap_loss.positive_radius = 0.6
_C.gap_loss.triplet_loss_gamma = 0.5

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_vote_loss = 1.0
_C.loss.weight_gap_loss = 5.0



def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
