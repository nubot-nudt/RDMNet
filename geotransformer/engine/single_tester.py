from typing import Dict

import torch
import ipdb
from tqdm import tqdm

from geotransformer.engine.base_tester import BaseTester
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.common import get_log_string
from geotransformer.utils.torch import release_cuda, to_cuda


class SingleTester(BaseTester):
    def __init__(self, cfg, local_rank=None, logger=None, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, local_rank=local_rank, logger=logger, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self, snapshot=None):
        assert self.test_loader is not None
        if self.distributed:
            self.test_loader.sampler.set_epoch(0)
        if snapshot is not None:
            self.load_snapshot(snapshot)
        else:
            self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        if self.local_rank == 0:
            pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        else:
            pbar= enumerate(self.test_loader)
        for iteration, data_dict in pbar:
            # on start
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            # test step
            torch.cuda.synchronize()
            timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            # eval step
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            # result_dict = release_cuda(result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            if self.local_rank == 0:
                pbar.set_description(message)
            torch.cuda.empty_cache()
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)

    def run(self, snapshot=None):
        assert self.test_loader is not None
        if self.distributed:
            self.test_loader.sampler.set_epoch(0)
        if snapshot is not None:
            self.load_snapshot(snapshot)
        else:
            self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        if self.local_rank == 0:
            pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        else:
            pbar= enumerate(self.test_loader)
        for iteration, data_dict in pbar:
            # on start

            

            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            # test step
            torch.cuda.synchronize()
            timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            # eval step
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            # result_dict = release_cuda(result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            if self.local_rank == 0:
                pbar.set_description(message)
            torch.cuda.empty_cache()
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)