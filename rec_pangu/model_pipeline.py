# -*- ecoding: utf-8 -*-
# @ModuleName: model_pipeline
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import time

from tqdm import tqdm
from sklearn.metrics import roc_auc_score,log_loss
from loguru import logger
from .utils import get_gpu_usage
import torch
import wandb

def train_model(model, train_loader, optimizer, device, metric_list=['roc_auc_score','log_loss'], num_task =1, use_wandb=False,log_rounds=100):
    model.train()
    max_iter = int(train_loader.dataset.__len__() / train_loader.batch_size)
    # scaler = torch.cuda.amp.GradScaler()
    if num_task == 1:
        pred_list = []
        label_list = []
        start_time = time.time()
        for idx,data in enumerate(train_loader):

            for key in data.keys():
                data[key] = data[key].to(device)

            # with torch.cuda.amp.autocast():
            output = model(data)
            pred = output['pred']
            loss = output['loss']

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            model.zero_grad()

            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

            auc = round(roc_auc_score(label_list[-1000:], pred_list[-1000:]), 4)

            if use_wandb:
                wandb.log({'train_loss':loss.item(),
                           'train_auc':auc})

            iter_time = time.time() - start_time
            remaining_time = round(((iter_time / (idx+1)) * (max_iter - idx + 1)) / 60, 2)

            if idx % log_rounds == 0 and device.type != 'cpu':
                logger.info(f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} AUC:{auc} GPU Mem:{get_gpu_usage(device)}')
            elif idx % log_rounds == 0:
                logger.info(f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} AUC:{auc}')
        res_dict = dict()
        for metric in metric_list:
            assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                metric, ['roc_auc_score', 'log_loss'])
            if metric =='log_loss':
                res_dict[f'train_{metric}'] = round(log_loss(label_list,pred_list, eps=1e-7),4)
            else:
                res_dict[f'train_{metric}'] = round(eval(metric)(label_list,pred_list),4)
        return res_dict
    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]
        start_time = time.time()
        for idx,data in enumerate(train_loader):

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            loss = output['loss']

            loss.backward()
            optimizer.step()
            model.zero_grad()
            for i in range(num_task):
                multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
                multi_task_label_list[i].extend(list(data[f'task{i + 1}_label'].squeeze(-1).cpu().detach().numpy()))
            if use_wandb:
                wandb.log({'train_loss':loss.item()})
            iter_time = time.time() - start_time
            remaining_time = round(((iter_time / (idx + 1)) * (max_iter - idx + 1)) / 60, 2)
            if idx % log_rounds ==0 and device.type != 'cpu':
                logger.info(f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} GPU Mem:{get_gpu_usage(device)}')
            elif idx % log_rounds ==0:
                logger.info(f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)}')

        res_dict = dict()
        for i in range(num_task):
            for metric in metric_list:
                assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                    metric, ['roc_auc_score', 'log_loss'])
                if metric == 'log_loss':
                    res_dict[f'train_task{i+1}_{metric}'] = round(log_loss(multi_task_label_list[i], multi_task_pred_list[i], eps=1e-7),4)
                else:
                    res_dict[f'train_task{i+1}_{metric}'] = round(eval(metric)(multi_task_label_list[i], multi_task_pred_list[i]),4)
        return res_dict

def valid_model(model, valid_loader, device, metric_list=['roc_auc_score','log_loss'],num_task =1):
    model.eval()
    if num_task == 1:
        pred_list = []
        label_list = []
        for data in valid_loader:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            pred = output['pred']

            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

        res_dict = dict()
        for metric in metric_list:
            assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                metric, ['roc_auc_score', 'log_loss'])
            if metric =='log_loss':
                res_dict[f'valid_{metric}'] = round(log_loss(label_list,pred_list, eps=1e-7),4)
            else:
                res_dict[f'valid_{metric}'] = round(eval(metric)(label_list,pred_list),4)

        return res_dict
    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]
        for data in valid_loader:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)

            for i in range(num_task):
                multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
                multi_task_label_list[i].extend(list(data[f'task{i + 1}_label'].squeeze(-1).cpu().detach().numpy()))

        res_dict = dict()
        for i in range(num_task):
            for metric in metric_list:
                assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                    metric, ['roc_auc_score', 'log_loss'])
                if metric == 'log_loss':
                    res_dict[f'valid_task{i+1}_{metric}'] = round(log_loss(multi_task_label_list[i], multi_task_pred_list[i], eps=1e-7),4)
                else:
                    res_dict[f'valid_task{i+1}_{metric}'] = round(eval(metric)(multi_task_label_list[i], multi_task_pred_list[i]),4)
        return res_dict

def test_model(model, test_loader, device, metric_list=['roc_auc_score','log_loss'],num_task =1):
    model.eval()
    if num_task == 1:
        pred_list = []
        label_list = []
        for data in test_loader:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            pred = output['pred']

            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

        res_dict = dict()
        for metric in metric_list:
            assert metric in ['roc_auc_score','log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                metric,['roc_auc_score','log_loss'])
            if metric == 'log_loss':
                res_dict[f'test_{metric}'] = round(log_loss(label_list, pred_list, eps=1e-7),4)
            else:
                res_dict[f'test_{metric}'] = round(eval(metric)(label_list, pred_list),4)

        return res_dict
    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]
        for data in test_loader:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)

            for i in range(num_task):
                multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
                multi_task_label_list[i].extend(list(data[f'task{i + 1}_label'].squeeze(-1).cpu().detach().numpy()))

        res_dict = dict()
        for i in range(num_task):
            for metric in metric_list:
                assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                    metric, ['roc_auc_score', 'log_loss'])
                if metric == 'log_loss':
                    res_dict[f'test_task{i + 1}_{metric}'] = round(log_loss(multi_task_label_list[i], multi_task_pred_list[i],
                                                                 eps=1e-7),4)
                else:
                    res_dict[f'test_task{i + 1}_{metric}'] = round(eval(metric)(multi_task_label_list[i], multi_task_pred_list[i]),4)
        return res_dict



def train_graph_model(model,optimizer,train_data):

    model.train()
    optimizer.zero_grad()
    result = model(train_data['train_graph_edge_index'], train_data['train_edge_index'],train_data['train_edge_label'])
    loss = result['loss']
    loss.backward()
    optimizer.step()

    return float(loss)


def test_graph_model(model,test_data):
    model.eval()
    result = model(test_data['test_graph_edge_index'], test_data['test_edge_index'], test_data['test_edge_label'])
    loss = result['loss']

    return float(loss)