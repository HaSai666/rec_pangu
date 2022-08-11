# -*- ecoding: utf-8 -*-
# @ModuleName: benchmark_trainer
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/13 3:55 PM
import pandas as pd
import torch
import time
import os
from tqdm import tqdm
from rec_pangu.trainer import RankTraniner
from rec_pangu.models.ranking import *
from rec_pangu.models.multi_task import *
from loguru import logger
from .utils import beautify_json

class BenchmarkTrainer:
    def __init__(self, num_task = 1, model_list = None, benhcmark_res_path = None, ckpt_root='./benchmark_ckpt'):
        self.num_task = num_task
        self.model_list = model_list
        self.benchmark_res_df = pd.DataFrame()
        self.benhcmark_res_path = benhcmark_res_path
        self.ckpt_root = ckpt_root

    def run(self,train_loader, enc_dict, valid_loader=None, test_loader=None, epoch=10, lr=1e-3, device=torch.device('cpu')):
        for model_name in self.model_list:
            logger.info(f'Start Training Model: {model_name}')
            model_class = eval(model_name)
            if self.num_task >1:
                model = model_class(enc_dict=enc_dict, device=device)
            else:
                model = model_class(enc_dict=enc_dict)

            model_trainer = RankTraniner(num_task=self.num_task)

            start_time = time.time()
            valid_metric = model_trainer.fit(model, train_loader, valid_loader, epoch=epoch, lr=lr, device=device)
            train_time = time.time() - start_time

            start_time = time.time()
            test_metric = model_trainer.evaluate_model(model,test_loader,device=device)
            test_time = time.time() - start_time
            model_ckpt = os.path.join(self.ckpt_root, model_name)
            model_trainer.save_model(model, model_ckpt)
            log_dict = {
                'model_name':model_name,
                'train_model_time':train_time * 1000,
                'test_model_time':test_time * 1000
            }
            logger.info(f'Model {model_name} Training Log :{beautify_json(log_dict)}')
            log_dict.update(valid_metric)
            log_dict.update(test_metric)
            self.benchmark_res_df = self.benchmark_res_df.append(log_dict, ignore_index=True)
            self.benchmark_res_df.to_csv(self.benhcmark_res_path, index=False)





