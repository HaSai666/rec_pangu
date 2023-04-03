# -*- ecoding: utf-8 -*-
# @ModuleName: benchmark_trainer
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/13 3:55 PM
from typing import Dict, List, Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader
import time
import os
from rec_pangu.trainer import RankTrainer
from rec_pangu.models.ranking import *
from rec_pangu.models.multi_task import *
from loguru import logger


class BenchmarkTrainer:
    """
    BenchmarkTrainer is used to train models and store training logs in a pandas dataframe.
    """

    def __init__(
            self,
            num_task: int = 1,
            model_list: Optional[List[str]] = None,
            benchmark_res_path: Optional[str] = None,
            ckpt_root: str = './benchmark_ckpt'
    ) -> None:
        """
        Args:
            num_task: The number of tasks being trained on.
            model_list: The list of models to train.
            benchmark_res_path: The filepath to store training logs.
            ckpt_root: The directory to store the trained models' checkpoints.
        """
        self.num_task = num_task
        self.model_list = model_list
        self.benchmark_res_df = pd.DataFrame()
        self.benchmark_res_path = benchmark_res_path
        self.ckpt_root = ckpt_root

    def run(
            self,
            train_loader: DataLoader,
            enc_dict: Dict[str, int],
            valid_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
            epoch: int = 10,
            lr: float = 1e-3,
            device: torch.device = torch.device('cpu')
    ) -> None:
        """Train and evaluate models on given data loaders and store logs.

        Args:
            train_loader: The data loader for training data.
            enc_dict: Dictionary of word embeddings.
            valid_loader: The data loader for validation data.
            test_loader: The data loader for test data.
            epoch: Maximum number of epochs to train the model.
            lr: Learning rate to use in training.
            device: Device to run the model on.
        """
        for model_name in self.model_list:
            logger.info(f'Start Training Model: {model_name}')
            model_class = eval(model_name)
            if self.num_task > 1:
                model = model_class(enc_dict=enc_dict, device=device)
            else:
                model = model_class(enc_dict=enc_dict)

            model_trainer = RankTrainer(num_task=self.num_task, model_ckpt_dir=os.path.join(self.ckpt_root, model_name))

            start_time = time.time()
            valid_metric = model_trainer.fit(model, train_loader, valid_loader, epoch=epoch, lr=lr, device=device)
            train_time = time.time() - start_time

            start_time = time.time()
            if test_loader is not None:
                test_metric = model_trainer.evaluate_model(model, test_loader, device=device)
            else:
                test_metric = {}
            test_time = time.time() - start_time
            model_ckpt = os.path.join(self.ckpt_root, model_name)
            model_trainer.save_all(model, enc_dict, model_ckpt)
            log_dict = {
                'model_name': model_name,
                'train_model_time': train_time * 1000,
                'test_model_time': test_time * 1000
            }
            logger.info(f'Model {model_name} Training Log :{log_dict}')
            log_dict.update(valid_metric)
            log_dict.update(test_metric)
            self.benchmark_res_df = self.benchmark_res_df.append(log_dict, ignore_index=True)
            self.benchmark_res_df.to_csv(self.benchmark_res_path, index=False)

# class BenchmarkTrainer:
#     def __init__(self, num_task = 1, model_list = None, benhcmark_res_path = None, ckpt_root='./benchmark_ckpt'):
#         self.num_task = num_task
#         self.model_list = model_list
#         self.benchmark_res_df = pd.DataFrame()
#         self.benhcmark_res_path = benhcmark_res_path
#         self.ckpt_root = ckpt_root
#
#     def run(self,train_loader, enc_dict, valid_loader=None, test_loader=None, epoch=10, lr=1e-3, device=torch.device('cpu')):
#         for model_name in self.model_list:
#             logger.info(f'Start Training Model: {model_name}')
#             model_class = eval(model_name)
#             if self.num_task >1:
#                 model = model_class(enc_dict=enc_dict, device=device)
#             else:
#                 model = model_class(enc_dict=enc_dict)
#
#             model_trainer = RankTrainer(num_task=self.num_task, model_ckpt_dir=os.path.join(self.ckpt_root, model_name))
#
#             start_time = time.time()
#             valid_metric = model_trainer.fit(model, train_loader, valid_loader, epoch=epoch, lr=lr, device=device)
#             train_time = time.time() - start_time
#
#             start_time = time.time()
#             if test_loader!= None:
#                 test_metric = model_trainer.evaluate_model(model,test_loader,device=device)
#             else:
#                 test_metric = {}
#             test_time = time.time() - start_time
#             model_ckpt = os.path.join(self.ckpt_root, model_name)
#             model_trainer.save_all(model, enc_dict, os.path.join(self.ckpt_root,model_name))
#             log_dict = {
#                 'model_name':model_name,
#                 'train_model_time':train_time * 1000,
#                 'test_model_time':test_time * 1000
#             }
#             logger.info(f'Model {model_name} Training Log :{log_dict}')
#             log_dict.update(valid_metric)
#             log_dict.update(test_metric)
#             self.benchmark_res_df = self.benchmark_res_df.append(log_dict, ignore_index=True)
#             self.benchmark_res_df.to_csv(self.benhcmark_res_path, index=False)
#
#
