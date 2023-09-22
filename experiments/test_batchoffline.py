import argparse
import os.path as osp
import os
from re import I
import time

import numpy as np

from geotransformer.engine import SingleTester
from geotransformer.utils.common import ensure_dir, get_log_string
from geotransformer.utils.torch import release_cuda

from dataset import test_data_loader
from loss import Evaluator
from model import create_model
from config import make_cfg

class Tester(SingleTester):
    def __init__(self, cfg, local_rank=None, logger=None):
        super().__init__(cfg, local_rank=local_rank, logger=logger)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.distributed, cfg.dataset)
        cfg.neighbor_limits=neighbor_limits
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        data_dict['testing'] = True
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        data_dict['testing'] = True
        data_dict['evaling'] = True
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'seq_id: {seq_id}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']

        # from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
        # est_transform = registration_with_ransac_from_correspondences(
        #             release_cuda(output_dict['src_corr_points']),
        #             release_cuda(output_dict['ref_corr_points']),
        #             distance_threshold=0.3,
        #             ransac_n=4,
        #             num_iterations=50000,
        #         )
        

        file_name = osp.join(self.output_dir, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)

        np.savez_compressed(
            file_name,
            # ref_points=release_cuda(output_dict['ref_points']),
            # src_points=release_cuda(output_dict['src_points']),
            # ref_points_f=release_cuda(output_dict['ref_points_f']),
            # src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            # ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            # src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            # estimated_transform=est_transform,
            transform=release_cuda(data_dict['transform']),
        )


        # file_name = osp.join(self.output_dir, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        # np.savez_compressed(
        #     file_name,
        #     # ref_points=release_cuda(output_dict['ref_points']),
        #     # src_points=release_cuda(output_dict['src_points']),
        #     ref_points_f=release_cuda(output_dict['ref_points_f']),
        #     src_points_f=release_cuda(output_dict['src_points_f']),
        #     ref_points_c=release_cuda(output_dict['ref_points_c']),
        #     src_points_c=release_cuda(output_dict['src_points_c']),
        #     ref_feats_c=release_cuda(output_dict['ref_feats_c']),
        #     src_feats_c=release_cuda(output_dict['src_feats_c']),
        #     ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
        #     src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
        #     ref_corr_points=release_cuda(output_dict['ref_corr_points']),
        #     src_corr_points=release_cuda(output_dict['src_corr_points']),
        #     corr_scores=release_cuda(output_dict['corr_scores']),
        #     gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
        #     gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
        #     estimated_transform=release_cuda(output_dict['estimated_transform']),
        #     transform=release_cuda(data_dict['transform']),
        #     # sch
        #     # overlaps=release_cuda(output_dict['overlaps']),
            
        #     # # sch: for visualization
        #     # ori_ref_points_c=release_cuda(output_dict['ori_ref_points_c']),
        #     # ori_src_points_c=release_cuda(output_dict['ori_src_points_c']),
        #     # shifted_ref_points_c=release_cuda(output_dict['shifted_ref_points_c']),
        #     # shifted_src_points_c=release_cuda(output_dict['shifted_src_points_c']),
        #     # ref_frame=release_cuda(data_dict['ref_frame']),
        #     # src_frame=release_cuda(data_dict['src_frame']),
        # )



        ##################################################
        #### save for evaluation
        # est_transform = registration_with_ransac_from_correspondences(
        #             release_cuda(output_dict['src_corr_points']),
        #             release_cuda(output_dict['ref_corr_points']),
        #             distance_threshold=0.3,
        #             ransac_n=4,
        #             num_iterations=50000,
        #         )

        # ## save rdmnet
        # file_name = f'/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/rdmnet/kitti/{seq_id}_{src_frame}_{ref_frame}.npz'
        # if not osp.exists(f'/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/rdmnet/kitti'):
        #     os.makedirs(f'/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/rdmnet/kitti')
        # np.savez_compressed(
        #     file_name,
        #     ref_corr_points=release_cuda(output_dict['ref_corr_points']),
        #     src_corr_points=release_cuda(output_dict['src_corr_points']),
        #     estimated_transform_lgr=release_cuda(output_dict['estimated_transform']),
        #     transform=release_cuda(data_dict['transform']),
        #     est_transform=est_transform,
        #     ref_points_c=release_cuda(output_dict['ref_points_c']),
        #     src_points_c=release_cuda(output_dict['src_points_c']),
        #     ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
        #     src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
        #     gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
        #     gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
        #     ori_ref_points_c=release_cuda(output_dict['ori_ref_points_c']),
        #     ori_src_points_c=release_cuda(output_dict['ori_src_points_c']),
        #     shifted_ref_points_c=release_cuda(output_dict['shifted_ref_points_c']),
        #     shifted_src_points_c=release_cuda(output_dict['shifted_src_points_c']),
        # )

        ## save geotrans
        # file_name = f'/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/geotrans/kitti/{seq_id}_{src_frame}_{ref_frame}.npz'
        # if not osp.exists(f'/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/geotrans/kitti'):
        #     os.makedirs(f'/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/geotrans/kitti')
        # np.savez_compressed(
        #     file_name,
        #     ref_corr_points=release_cuda(output_dict['ref_corr_points']),
        #     src_corr_points=release_cuda(output_dict['src_corr_points']),
        #     estimated_transform_lgr=release_cuda(output_dict['estimated_transform']),
        #     transform=release_cuda(data_dict['transform']),
        #     est_transform=est_transform,
        #     ref_points_c=release_cuda(output_dict['ref_points_c']),
        #     src_points_c=release_cuda(output_dict['src_points_c']),
        #     ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
        #     src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
        #     gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
        #     gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
        # )

        

        

# def main():
#     cfg = make_cfg()
#     tester = Tester(cfg)

#     for i in range(100,200):
        
#         snapshot = '/home/chenghao/DL_workspace/Registration/GeoTransformer/output/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/snapshots/epoch-{}.pth.tar'.format(i)

#         tester.run(snapshot)

#         cfg.test_epoch=i
#         cfg.method = 'ransac'
#         cfg.verbose = False
#         cfg.num_corr = None
#         from eval import eval_one_epoch
#         eval_one_epoch(cfg, cfg, tester.logger)

# if __name__ == '__main__':
#     main()


def main(local_rank, i, cfg, logger):



    tester = Tester(cfg, local_rank, logger)
    # snapshot = '/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/output/snapshots/epoch-{}.pth.tar'.format(i)
    snapshot = '/mnt/Mount/sch_ws/r_mdgat/checkpoint/kitti/output/6.17kpcon5-4-gap_loss-snapshots-/epoch-173.pth.tar'


    # snapshot = '../../weights/geotransformer-kitti.pth.tar'

    tester.run(snapshot)

    

import torch.multiprocessing as mp
from geotransformer.engine.logger import Logger
from eval import eval_one_epoch

if __name__ == '__main__':

    cfg = make_cfg()

    cfg.robust_test= 0
    cfg.benchmark_distance=None
    # cfg.feature_dir = osp.join(cfg.feature_dir,'%.02f'%cfg.robust_test)
    cfg.subset=None
    cfg.feature_dir = cfg.feature_dir + f'{cfg.dataset}'

    '''for loop closure test. Note! change distance4/overlap0.3 in dataset'''
    # cfg.subset=[0]
    # cfg.feature_dir = '/mnt/Mount/sch_ws/r_mdgat/features/registration_lc/rdmnet/' + 'kitti/' + '%02d/'%cfg.subset[0] + 'distance4'

    
    
    log_file = osp.join(cfg.log_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file, local_rank=0)

    nprocs = 4

    for i in range(190,200,10): 
        if True:
            cfg.nprocs = nprocs
            '''torch.multiprocessing.spawn: directly run the script'''
            mp.spawn(main,
                nprocs=nprocs,
                args=(i, cfg, logger),
                )
        else:
            mp.set_start_method('spawn')
            main_worker(0, 1, cfgs)

        cfg.test_epoch=i
        cfg.method = 'lgr'
        cfg.verbose = False
        cfg.num_corr = None
        eval_one_epoch(cfg, cfg, logger)

        cfg.test_epoch=i
        cfg.method = 'ransac'
        cfg.verbose = False
        cfg.num_corr = None
        eval_one_epoch(cfg, cfg, logger)
        logger.critical('\n')

        # cfg.test_epoch=i
        # cfg.method = 'lgr'
        # cfg.verbose = False
        # cfg.num_corr = 5000

        # eval_one_epoch(cfg, cfg, logger)

        # # cfg.test_epoch=i
        # # cfg.method = 'ransac'
        # # cfg.verbose = False
        # # cfg.num_corr = 5000

        # # eval_one_epoch(cfg, cfg, logger)
        # logger.critical('\n')

        # cfg.test_epoch=i
        # cfg.method = 'lgr'
        # cfg.verbose = False
        # cfg.num_corr = 1000

        # eval_one_epoch(cfg, cfg, logger)

        # # cfg.test_epoch=i
        # # cfg.method = 'ransac'
        # # cfg.verbose = False
        # # cfg.num_corr = 1000

        # # eval_one_epoch(cfg, cfg, logger)
        # logger.critical('\n')
        
        # cfg.test_epoch=i
        # cfg.method = 'lgr'
        # cfg.verbose = False
        # cfg.num_corr = 250

        # eval_one_epoch(cfg, cfg, logger)

        # # cfg.test_epoch=i
        # # cfg.method = 'ransac'
        # # cfg.verbose = False
        # # cfg.num_corr = 250

        # # eval_one_epoch(cfg, cfg, logger)
        # logger.critical('\n')

        
    