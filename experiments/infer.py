import argparse
import os.path as osp
import time

import numpy as np

from geotransformer.engine import SingleTester
from geotransformer.utils.common import ensure_dir, get_log_string
from geotransformer.utils.torch import release_cuda

# from dataset import test_data_loader
from dataset import  infer_data_loader

from loss import Evaluator
from model_infer import create_model
from config import make_cfg


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = infer_data_loader(cfg, self.distributed, cfg.dataset)
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
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'seq_id: {seq_id}, id0: {ref_frame}, id1: {src_frame}'
        # message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']


        f = open(osp.join(self.output_dir, '%02d_pose'%seq_id),'a')

        estimated_transform=release_cuda(output_dict['estimated_transform'])
        M2=estimated_transform.reshape(-1)[:12]

        f.write(f'{ref_frame} {src_frame} {M2[0]:.6f} {M2[1]:.6f} {M2[2]:.6f} {M2[3]:.6f} {M2[4]:.6f} {M2[5]:.6f} {M2[6]:.6f} {M2[7]:.6f} {M2[8]:.6f} {M2[9]:.6f} {M2[10]:.6f} {M2[11]:.6f} \n')

        from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
        est_transform = registration_with_ransac_from_correspondences(
                    release_cuda(output_dict['src_corr_points']),
                    release_cuda(output_dict['ref_corr_points']),
                    distance_threshold=0.3,
                    ransac_n=4,
                    num_iterations=50000,
                )

        file_name = osp.join(self.output_dir, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),

            estimated_transform=release_cuda(output_dict['estimated_transform']),
            estimated_transform_ransac=est_transform,
            
            # for visualization
            # ori_ref_points_c=release_cuda(output_dict['ori_ref_points_c']),
            # ori_src_points_c=release_cuda(output_dict['ori_src_points_c']),
            # shifted_ref_points_c=release_cuda(output_dict['shifted_ref_points_c']),
            # shifted_src_points_c=release_cuda(output_dict['shifted_src_points_c']),
            # ref_frame=release_cuda(data_dict['ref_frame']),
            # src_frame=release_cuda(data_dict['src_frame']),
        )


def main():

    cfg = make_cfg()
    
    cfg.feature_dir = cfg.feature_dir + f'{cfg.dataset}'

    if cfg.dataset=='mulran':
        cfg.Vote.inference_use_vote = False


    snapshots='weights/rdmnet.pth.tar'
    tester = Tester(cfg)
    tester.run(snapshots)


if __name__ == '__main__':
    main()