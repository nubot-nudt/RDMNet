import sys
import json
import argparse
import glob
import os.path as osp
import time
from warnings import catch_warnings

import numpy as np
import torch
from tqdm import tqdm

from config import make_cfg
from geotransformer.engine import Logger
from geotransformer.modules.registration import weighted_procrustes
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences, registration_with_ransac_from_featurematch
from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
    evaluate_overlap,
    evaluate_node_overlap
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd', 'ransac_featurematch'], default='lgr', help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser


def eval_one_epoch(args, cfg, logger):
    features_root = cfg.feature_dir + f'{cfg.dataset}'


    node_detect_meter = SummaryBoard()
    node_detect_meter.register_meter('pred_overlap')
    node_detect_meter.register_meter('gt_overlap')
    node_detect_meter.register_meter('pred_true_overlap')
    node_detect_meter.register_meter('gt_max_overlap_for_each_node')

    

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter('precision')
    coarse_matching_meter.register_meter('PMR>0')
    coarse_matching_meter.register_meter('PMR>=0.1')
    coarse_matching_meter.register_meter('PMR>=0.3')
    coarse_matching_meter.register_meter('PMR>=0.5')

    coarse_matching_meter.register_meter('node_OV')

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter('recall')
    fine_matching_meter.register_meter('inlier_ratio')
    fine_matching_meter.register_meter('inlier_ratio_0.3')
    fine_matching_meter.register_meter('inlier_ratio_0.1')
    fine_matching_meter.register_meter('overlap')
    fine_matching_meter.register_meter('num_corr')

    

    registration_meter = SummaryBoard()
    registration_meter.register_meter('recall')
    registration_meter.register_meter('rre')
    registration_meter.register_meter('rte')
    registration_meter.register_meter('x')
    registration_meter.register_meter('y')
    registration_meter.register_meter('z')


    fail_case = []

    file_names = sorted(
        glob.glob(osp.join(features_root, '*.npz')),
        key=lambda x: [int(i) for i in osp.splitext(osp.basename(x))[0].split('_')],
    )
    num_test_pairs = len(file_names)
    pbar = tqdm(enumerate(file_names), total=num_test_pairs)
    for i, file_name in pbar:
        try:
            seq_id, src_frame, ref_frame = [int(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]
        except:
            seq_id, src_frame, ref_frame = [x for x in osp.splitext(osp.basename(file_name))[0].split('_')]
            src_frame=int(src_frame)
            ref_frame=int(ref_frame)
            # print(file_name)

        # delete bad data
        if seq_id==8 and src_frame==15:
            continue
        
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
        gt_node_corr_overlaps = data_dict['gt_node_corr_overlaps']
        gt_transform = data_dict['transform']

        # ref_points=data_dict['ref_points'],
        # src_points=data_dict['src_points'],

        # overlaps=data_dict['overlaps'],


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

        # coarse_matching_meter.update('node_OV', np.array(overlaps).mean())

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
        fine_matching_meter.update('inlier_ratio_0.3', fine_matching_result_dict['inlier_ratio_0.3'])
        fine_matching_meter.update('inlier_ratio_0.1', fine_matching_result_dict['inlier_ratio_0.1'])
        fine_matching_meter.update('overlap', overlap)
        fine_matching_meter.update('recall', float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))
        fine_matching_meter.update('num_corr', corr_scores.shape[0])

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
        elif args.method == 'teaser':
            import teaserpp_python
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.cbar2 = 1
            solver_params.noise_bound = 0.01
            solver_params.estimate_scaling = False
            solver_params.rotation_estimation_algorithm = (
                teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            )
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            solver_params.rotation_cost_threshold = 1e-12

            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(src_corr_points.T, ref_corr_points.T)

            solution = solver.getSolution()

            est_transform = np.zeros((4,4))
            est_transform[:3,:3] = solution.rotation
            est_transform[:3,3] = solution.translation
            est_transform[3,:3] = 1.0
        else:
            raise ValueError(f'Unsupported registration method: {args.method}.')

        rre, rte, rx, ry, rz = compute_registration_error(gt_transform, est_transform)

        accepted = rre < cfg.eval.rre_threshold and rte < cfg.eval.rte_threshold
        # registration_meter.update('rre', rre)
        # registration_meter.update('rte', rte)
        # registration_meter.update('x',rx)
        # registration_meter.update('y',ry)
        # registration_meter.update('z',rz)
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


    if args.test_epoch is not None:
        logger.critical(f'Epoch {args.test_epoch}, method {args.method}')

    # 1. print correspondence evaluation results
    message = '  Node Detection'
    message += ', PRED_OV: {:.3f}'.format(node_detect_meter.mean('pred_overlap'))
    message += ', GT_OV: {:.3f}'.format(node_detect_meter.mean('gt_overlap'))
    message += ', PRED_T_OV: {:.3f}'.format(node_detect_meter.mean('pred_true_overlap'))
    message += ', GT_MAX_OV: {:.3f}'.format(node_detect_meter.mean('gt_max_overlap_for_each_node'))
    
    logger.critical(message)

    # 1. print correspondence evaluation results
    message = '  Coarse Matching'
    message += ', PIR: {:.3f}'.format(coarse_matching_meter.mean('precision'))
    message += ', PMR>0: {:.3f}'.format(coarse_matching_meter.mean('PMR>0'))
    message += ', PMR>=0.1: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.1'))
    message += ', PMR>=0.3: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.3'))
    message += ', PMR>=0.5: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.5'))
    # message += ', node_OV: {:.3f}'.format(coarse_matching_meter.mean('node_OV'))
    
    logger.critical(message)

    message = '  Fine Matching'
    message += ', FMR: {:.4f}'.format(fine_matching_meter.mean('recall'))
    message += ', IR: {:.3f}'.format(fine_matching_meter.mean('inlier_ratio'))
    message += ', IR_0.3: {:.3f}'.format(fine_matching_meter.mean('inlier_ratio_0.3'))
    message += ', IR_0.1: {:.3f}'.format(fine_matching_meter.mean('inlier_ratio_0.1'))
    message += ', num_Corr: {:.3f}'.format(fine_matching_meter.mean('num_corr'))
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


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.log_dir, 'eval-{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
    logger = Logger(log_file=log_file)

    message = 'Command executed: ' + ' '.join(sys.argv)
    logger.info(message)
    message = 'Configs:\n' + json.dumps(cfg, indent=4)
    logger.info(message)
    
    eval_one_epoch(args, cfg, logger)

if __name__ == '__main__':
    main()
