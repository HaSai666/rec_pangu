# -*- ecoding: utf-8 -*-
# @ModuleName: trainer
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import os.path

import torch
from .model_pipeline import train_model, valid_model, test_model

class RankTraniner:
    def __init__(self, num_task = 1):
        self.num_task = num_task
    def fit(self, model, train_loader, valid_loader=None, epoch=10, lr=1e-3, device=torch.device('cpu')):
        # 声明optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),eps=1e-08, weight_decay=0)
        model = model.to(device)
        # 模型训练流程
        for i in range(epoch):
            # 模型训练
            train_metirc = train_model(model, train_loader, optimizer=optimizer, device=device,num_task=self.num_task)
            print("Train Metric:")
            print(train_metirc)
            # 模型验证
            if valid_loader != None:
                valid_metric = valid_model(model, valid_loader, device, num_task=self.num_task)
                print("Valid Metric:")
                print(valid_metric)
        return valid_metric

    def save_model(self, model, model_ckpt_dir):
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict,os.path.join(model_ckpt_dir, 'model.pth'))

    def evaluate_model(self, model, test_loader, device=torch.device('cpu')):
        test_metric = test_model(model, test_loader, device, num_task=self.num_task)
        return test_metric




