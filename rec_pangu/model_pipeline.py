# -*- ecoding: utf-8 -*-
# @ModuleName: model_pipeline
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict, List
import torch
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from loguru import logger
from .utils import get_gpu_usage, evaluate_recall, get_recall_predict
import faiss
import wandb


def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                metric_list: List[str] = ['roc_auc_score', 'log_loss'],
                num_task: int = 1,
                use_wandb: bool = False,
                log_rounds: int = 100) -> dict:
    """
    Trains the provided model using the given data and optimizer. Computes evaluation metrics and logs the training progress.

    Args:
        model: The model to be trained
        train_loader: The data loader for the training data
        optimizer: The optimizer to be used during training
        device: The device to run the training on
        metric_list: The metrics to compute during training. Should be a list of any of 'roc_auc_score' or 'log_loss'. Default is ['roc_auc_score', 'log_loss'].
        num_task: The number of tasks being performed. Should be an integer. Default is 1.
        use_wandb: Whether or not to use Weights & Biases for logging. Should be a boolean. Default is False.
        log_rounds: The number of iterations before logging progress. Should be an integer. Default is 100.

    Returns:
        A dictionary containing the computed evaluation metrics
    """
    model.train()
    max_iter = int(train_loader.dataset.__len__() / train_loader.batch_size)
    if num_task == 1:
        pred_list = []
        label_list = []
        start_time = time.time()
        for idx, data in enumerate(train_loader):

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

            auc = round(roc_auc_score(label_list[-1000:], pred_list[-1000:]), 4)

            if use_wandb:
                wandb.log({'train_loss': loss.item(),
                           'train_auc': auc})

            iter_time = time.time() - start_time
            remaining_time = round(((iter_time / (idx + 1)) * (max_iter - idx + 1)) / 60, 2)

            if idx % log_rounds == 0 and device.type != 'cpu':
                logger.info(
                    f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} AUC:{auc} GPU Mem:{get_gpu_usage(device)}')
            elif idx % log_rounds == 0:
                logger.info(
                    f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} AUC:{auc}')
        res_dict = dict()
        for metric in metric_list:
            assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                metric, ['roc_auc_score', 'log_loss'])
            if metric == 'log_loss':
                res_dict[f'train_{metric}'] = round(log_loss(label_list, pred_list, eps=1e-7), 4)
            else:
                res_dict[f'train_{metric}'] = round(eval(metric)(label_list, pred_list), 4)
        return res_dict
    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]
        start_time = time.time()
        for idx, data in enumerate(train_loader):

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
                wandb.log({'train_loss': loss.item()})
            iter_time = time.time() - start_time
            remaining_time = round(((iter_time / (idx + 1)) * (max_iter - idx + 1)) / 60, 2)
            if idx % log_rounds == 0 and device.type != 'cpu':
                logger.info(
                    f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} GPU Mem:{get_gpu_usage(device)}')
            elif idx % log_rounds == 0:
                logger.info(
                    f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)}')

        res_dict = dict()
        for i in range(num_task):
            for metric in metric_list:
                assert metric in ['roc_auc_score', 'log_loss'], 'metric :{} not supported! metric must be in {}'.format(
                    metric, ['roc_auc_score', 'log_loss'])
                if metric == 'log_loss':
                    res_dict[f'train_task{i + 1}_{metric}'] = round(
                        log_loss(multi_task_label_list[i], multi_task_pred_list[i], eps=1e-7), 4)
                else:
                    res_dict[f'train_task{i + 1}_{metric}'] = round(
                        eval(metric)(multi_task_label_list[i], multi_task_pred_list[i]), 4)
        return res_dict


def test_model(model: torch.nn.Module,
               test_loader: torch.utils.data.DataLoader,
               device: torch.device,
               metric_list: List[str] = ['roc_auc_score', 'log_loss'],
               num_task: int = 1) -> dict:
    """
    Evaluate the performance of a given model on the test set.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (DataLoader): PyTorch DataLoader for loading the test dataset.
        device (torch.device): Device where the computation will take place.
        metric_list (List[str]): List of evaluation metrics to be used. Supported metrics are ['roc_auc_score', 'log_loss']. Defaults to ['roc_auc_score','log_loss']
        num_task (int): Number of tasks. Defaults to 1.

    Returns:
        Dict: A dictionary containing the evaluation metrics and their corresponding values.
    """

    # Switch to evaluation mode
    model.eval()

    if num_task == 1:
        pred_list = []
        label_list = []

        # Iterate over the test set
        for data in test_loader:

            # Move the data to the device
            for key in data.keys():
                data[key] = data[key].to(device)

            # Forward pass
            output = model(data)
            pred = output['pred']

            # Append predictions and labels to corresponding lists
            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

        # Initialize the results dictionary
        res_dict = dict()

        # Compute evaluation metric for each supported metric
        for metric in metric_list:
            assert metric in ['roc_auc_score',
                              'log_loss'], f"Unsupported metric: {metric}. Supported metrics are ['roc_auc_score','log_loss']."
            if metric == 'log_loss':
                res_dict[metric] = round(log_loss(label_list, pred_list, eps=1e-7), 4)
            else:
                res_dict[metric] = round(eval(metric)(label_list, pred_list), 4)

        return res_dict

    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]

        # Iterate over the test set
        for data in test_loader:

            # Move the data to the device
            for key in data.keys():
                data[key] = data[key].to(device)

            # Forward pass
            output = model(data)

            # Append predictions and labels to the corresponding task lists
            for i in range(num_task):
                multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
                multi_task_label_list[i].extend(list(data[f'task{i + 1}_label'].squeeze(-1).cpu().detach().numpy()))

        # Initialize the results dictionary
        res_dict = dict()

        # Compute the evaluation metric for each task and each supported metric
        for i in range(num_task):
            for metric in metric_list:
                assert metric in ['roc_auc_score',
                                  'log_loss'], f"Unsupported metric: {metric}. Supported metrics are ['roc_auc_score','log_loss']."
                if metric == 'log_loss':
                    res_dict[f'test_task{i + 1}_{metric}'] = round(
                        log_loss(multi_task_label_list[i], multi_task_pred_list[i], eps=1e-7), 4)
                else:
                    res_dict[f'test_task{i + 1}_{metric}'] = round(
                        eval(metric)(multi_task_label_list[i], multi_task_pred_list[i]), 4)

        return res_dict


def train_sequence_model(model: torch.nn.Module,
                         train_loader: torch.utils.data.DataLoader,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device,
                         use_wandb: bool = False,
                         log_rounds: int = 100):
    """
    Function to train a sequence model.

    Args:
        model: (nn.Module) Sequence model to be trained.
        train_loader: (DataLoader) Loader for the training dataset.
        optimizer: (optim) Optimizer for weight update.
        device: (torch.device) Device on which to run the model.
        use_wandb: (bool) Whether to use Weights & Biases logging.
        log_rounds: (int) Frequency of logging training loss.

    Returns:
        None
    """
    model.train()

    # Calculate the number of iterations required to complete an epoch
    max_iter = train_loader.dataset.__len__() // train_loader.batch_size

    start_time = time.time()

    # Iterate over the dataset batches
    for idx, data in enumerate(train_loader):
        # Move the data to the device
        for key in data.keys():
            data[key] = data[key].to(device)

        # Forward pass
        output = model(data, is_training=True)
        loss = output['loss']

        # Backward pass
        loss.backward()
        optimizer.step()
        model.zero_grad()

        # Log the loss
        if use_wandb:
            wandb.log({'train_loss': loss.item()})

        # Calculate time for iteration and remaining time
        iter_time = time.time() - start_time
        remaining_time = round(((iter_time / (idx + 1)) * (max_iter - idx + 1)) / 60, 2)

        # Log progress
        if idx % log_rounds == 0 and device.type != 'cpu':
            logger.info(
                f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} GPU Mem:{get_gpu_usage(device)}')
        elif idx % log_rounds == 0:
            logger.info(
                f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} ')


def test_sequence_model(model: torch.nn.Module,
                        test_loader: torch.utils.data.DataLoader,
                        device: torch.device,
                        topk_list: List[int] = [20, 50, 100],
                        use_wandb: bool = False) -> dict:
    """
    Test a sequence model's accuracy on a test dataset by returning recall metrics for various top k values.
    Args:
        model (nn.Module): Sequence model being tested.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device where the model is being trained.
        topk_list (list[int]): List of top k values being evaluated. Default: [20, 50, 100].
        use_wandb (bool): Flag to use Weights & Biases for logging. Default: False.

    Returns:
        A dictionary containing recall metrics corresponding to each top k value.
    """
    # Set the model in evaluation mode
    model.eval()

    # Get test ground truth
    test_gd = test_loader.dataset.get_test_gd()

    # Get sequence model's prediction for top N values
    preds = get_recall_predict(model, test_loader, device, topN=200)

    # Calculate recall metrics for each top k value
    metric_dict = {}
    for i, k in enumerate(topk_list):
        temp_metric_dict = evaluate_recall(preds, test_gd, k)
        logger.info(temp_metric_dict)
        metric_dict.update(temp_metric_dict)

    # Log metrics using Weights & Biases
    if use_wandb:
        wandb.log(metric_dict)

    return metric_dict


def train_graph_model(model, train_dataset, optimizer, device, batch_size=1024):
    model.train()
    epoch_loss = 0
    pbar = tqdm(range(train_dataset.__len__() // batch_size))
    for _ in pbar:

        data = train_dataset.sample(batch_size)

        for key in data.keys():
            data[key] = data[key].to(device)

        output = model(data)
        loss = output['loss']

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_loss += loss.item()
        pbar.set_description("Loss {}".format(round(epoch_loss, 4)))
    return epoch_loss


def test_graph_model(model, train_gd, test_gd, hidden_size, topN=50):
    model.eval()
    output = model(None, is_training=False)
    user_embs = output['user_emb'].detach().cpu().numpy()
    item_embs = output['item_emb'].detach().cpu().numpy()

    test_user_list = list(test_gd.keys())

    faiss_index = faiss.IndexFlatIP(hidden_size)
    faiss_index.add(item_embs)

    preds = dict()

    for i in tqdm(range(0, len(test_user_list), 1000)):
        user_ids = test_user_list[i:i + 1000]
        batch_user_emb = user_embs[user_ids, :]
        D, I = faiss_index.search(batch_user_emb, 1000)

        for i, iid_list in enumerate(user_ids):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
            train_items = train_gd.get(user_ids[i], [])
            preds[user_ids[i]] = [x for x in list(I[i, :]) if x not in train_items]
    return evaluate_recall(preds, test_gd, topN=topN)
