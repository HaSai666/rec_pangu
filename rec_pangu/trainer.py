# -*- ecoding: utf-8 -*-
# @ModuleName: trainer
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import os.path

import torch
from .model_pipeline import train_model, valid_model, test_model, train_graph_model, test_graph_model
from .utils import beautify_json
from .dataset import BaseDataset,MultiTaskDataset
from loguru import logger
import torch.utils.data as D

class RankTraniner:
    def __init__(self, num_task = 1):
        self.num_task = num_task
    def fit(self, model, train_loader, valid_loader=None, epoch=10, lr=1e-3, device=torch.device('cpu')):
        # 声明optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),eps=1e-08, weight_decay=0)
        model = model.to(device)
        # 模型训练流程
        logger.info('Model Starting Training ')
        for i in range(epoch):
            # 模型训练
            train_metric = train_model(model, train_loader, optimizer=optimizer, device=device,num_task=self.num_task)
            logger.info(f"Train Metric:{beautify_json(train_metric)}")
            # 模型验证
            if valid_loader != None:
                valid_metric = valid_model(model, valid_loader, device, num_task=self.num_task)
                logger.info(f"Valid Metric:{beautify_json(valid_metric)}")
        return valid_metric

    def save_model(self, model, model_ckpt_dir):
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict,os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Model Saved to {model_ckpt_dir}')

    def save_all(self, model, enc_dict, model_ckpt_dir):
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict(),
                     'enc_dict': enc_dict}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Enc_dict and Model Saved to {model_ckpt_dir}')

    def evaluate_model(self, model, test_loader, device=torch.device('cpu')):
        test_metric = test_model(model, test_loader, device, num_task=self.num_task)
        logger.info(f"Test Metric:{beautify_json(test_metric)}")
        return test_metric

    def predict_dataloader(self, model, test_loader, device=torch.device('cpu')):
        model.eval()
        if self.num_task == 1:
            pred_list = []
            for data in test_loader:
                for key in data.keys():
                    data[key] = data[key].to(device)
                output = model(data)
                pred = output['pred']
                pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            return pred_list
        else:
            multi_task_pred_list = [[] for _ in range(self.num_task)]
            for data in test_loader:
                for key in data.keys():
                    data[key] = data[key].to(device)
                output = model(data)
                for i in range(self.num_task):
                    multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
            return multi_task_pred_list

    def predict_dataframe(self,model, test_df, enc_dict, schema, device=torch.device('cpu')):
        test_dataset = BaseDataset(schema, test_df, enc_dict=enc_dict)
        test_loader = D.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        return self.predict_dataloader(model, test_loader, device=device)

class GraphTrainer:
    def __init__(self):
        logger.info("Graph Trainer")

    def fit(self,model,train_data,epoch,lr):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        for epoch in range(1, epoch+1):
            train_metric = train_graph_model(model,optimizer,train_data)
            logger.info(f"Train Metric:{beautify_json(train_metric)}")

    def save_model(self, model, model_ckpt_dir):
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict,os.path.join(model_ckpt_dir, 'model.pth'))

    def evaluate_model(self, model, test_data):
        test_metric = test_graph_model(model,test_data)
        logger.info(f"Test Metric:{beautify_json(test_metric)}")
        return test_metric