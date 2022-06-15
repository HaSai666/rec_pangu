# -*- ecoding: utf-8 -*-
# @ModuleName: model_pipeline
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,log_loss

def train_model(model, train_loader, optimizer, device, metric_list=['roc_auc_score','log_loss'], num_task =1):
    model.train()
    if num_task == 1:
        pred_list = []
        label_list = []
        pbar = tqdm(train_loader)
        for data in pbar:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            pred = output['pred']
            loss = output['loss']

            loss.backward()
            optimizer.step()
            model.zero_grad()

            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())
            pbar.set_description("Loss {}".format(loss))

        res_dict = dict()
        for metric in metric_list:
            assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                metric, ['roc_auc_score', 'log_loss'])
            if metric =='log_loss':
                res_dict[f'train_{metric}'] = log_loss(label_list,pred_list, eps=1e-7)
            else:
                res_dict[f'train_{metric}'] = eval(metric)(label_list,pred_list)
        return res_dict
    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]
        pbar = tqdm(train_loader)
        for data in pbar:

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
            pbar.set_description("Loss {}".format(loss))

        res_dict = dict()
        for i in range(num_task):
            for metric in metric_list:
                assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                    metric, ['roc_auc_score', 'log_loss'])
                if metric == 'log_loss':
                    res_dict[f'train_task{i+1}_{metric}'] = log_loss(multi_task_label_list[i], multi_task_pred_list[i], eps=1e-7)
                else:
                    res_dict[f'train_task{i+1}_{metric}'] = eval(metric)(multi_task_label_list[i], multi_task_pred_list[i])
        return res_dict

def valid_model(model, valid_loader, device, metric_list=['roc_auc_score','log_loss'],num_task =1):
    model.eval()
    if num_task == 1:
        pred_list = []
        label_list = []
        for data in tqdm(valid_loader):

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
                res_dict[f'valid_{metric}'] = log_loss(label_list,pred_list, eps=1e-7)
            else:
                res_dict[f'valid_{metric}'] = eval(metric)(label_list,pred_list)

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
                    res_dict[f'valid_task{i+1}_{metric}'] = log_loss(multi_task_label_list[i], multi_task_pred_list[i], eps=1e-7)
                else:
                    res_dict[f'valid_task{i+1}_{metric}'] = eval(metric)(multi_task_label_list[i], multi_task_pred_list[i])
        return res_dict

def test_model(model, test_loader, device, metric_list=['roc_auc_score','log_loss'],num_task =1):
    model.eval()
    if num_task == 1:
        pred_list = []
        label_list = []
        for data in tqdm(test_loader):

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
                res_dict[f'test_{metric}'] = log_loss(label_list, pred_list, eps=1e-7)
            else:
                res_dict[f'test_{metric}'] = eval(metric)(label_list, pred_list)

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
                    res_dict[f'test_task{i + 1}_{metric}'] = log_loss(multi_task_label_list[i], multi_task_pred_list[i],
                                                                 eps=1e-7)
                else:
                    res_dict[f'test_task{i + 1}_{metric}'] = eval(metric)(multi_task_label_list[i], multi_task_pred_list[i])
        return res_dict
