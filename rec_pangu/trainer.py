# -*- ecoding: utf-8 -*-
# @ModuleName: trainer
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
"""
Model Trainer
"""
import os
import torch
from .model_pipeline import train_model, test_model, train_graph_model, test_graph_model, train_sequence_model, \
    test_sequence_model
from .utils import beautify_json
from .dataset import BaseDataset, MultiTaskDataset
from loguru import logger
import torch.utils.data as D
import wandb
from typing import List, Optional
import pandas as pd


class RankTrainer:
    """
    A class for training ranking models with single or multiple tasks.

    Attributes:
        num_task (int): The number of tasks for the model.
        wandb_config (dict): Configuration for Weights and Biases integration.
        model_ckpt_dir (str): The directory for saving model checkpoints.
        use_wandb (bool): Whether to use Weights and Biases integration.
    """

    def __init__(self, num_task: int = 1, wandb_config: dict = None, model_ckpt_dir: str = './model_ckpt'):
        """
        Initializes RankTrainer with the given number of tasks, wandb_config, and model_ckpt_dir.

        Args:
            num_task (int, optional): The number of tasks for the model. Defaults to 1.
            wandb_config (dict, optional): Configuration for Weights and Biases integration. Defaults to None.
            model_ckpt_dir (str, optional): The directory for saving model checkpoints. Defaults to './model_ckpt'.
        """
        self.num_task = num_task
        self.wandb_config = wandb_config
        self.model_ckpt_dir = model_ckpt_dir
        self.use_wandb = self.wandb_config is not None
        if self.use_wandb:
            wandb.login(key=self.wandb_config['key'])
            self.wandb_config.pop('key')

    def fit(self, model, train_loader, valid_loader: Optional = None, epoch: int = 10, lr: float = 1e-3,
            device: torch.device = torch.device('cpu'), use_earlystopping: bool = False,
            max_patience: int = 999, monitor_metric: Optional[str] = None):
        """
        Train the model using the given data loaders and hyperparameters.

        Args:
            model (nn.Module): The model to be trained.
            train_loader (DataLoader): The data loader for training data.
            valid_loader (DataLoader, optional): The data loader for validation data. Defaults to None.
            epoch (int, optional): The number of training epochs. Defaults to 10.
            lr (float, optional): The learning rate. Defaults to 1e-3.
            device (torch.device, optional): The device to train the model on. Defaults to 'cpu'.
            use_earlystopping (bool, optional): Whether to use early stopping. Defaults to False.
            max_patience (int, optional): The maximum patience for early stopping. Defaults to 999.
            monitor_metric (str, optional): The metric to monitor for early stopping. Defaults to None.
        """
        if self.use_wandb:
            wandb.init(
                **self.wandb_config
            )
        # Declare the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model = model.to(device)
        # Model training process
        logger.info('Model Starting Training ')
        best_epoch = -1
        best_metric = -1
        for i in range(1, epoch + 1):
            # Model training
            train_metric = train_model(model, train_loader, optimizer=optimizer, device=device, num_task=self.num_task,
                                       use_wandb=self.use_wandb)
            logger.info(f"Train Metric:{train_metric}")
            # Model validation
            if valid_loader is not None:
                valid_metric = test_model(model, valid_loader, device, num_task=self.num_task)
                model_str = f'e_{i}'
                self.save_train_model(model, self.model_ckpt_dir, model_str)
                if self.use_wandb:
                    wandb.log(valid_metric)

                if use_earlystopping:
                    assert monitor_metric in valid_metric.keys(), f'{monitor_metric} not in Valid Metric {valid_metric.keys()}'
                    if valid_metric[monitor_metric] > best_metric:
                        best_epoch = i
                        best_metric = valid_metric[monitor_metric]
                        self.save_train_model(model, self.model_ckpt_dir, 'best')
                    if i - best_epoch >= max_patience:
                        logger.info(f"EarlyStopping at the Epoch {i} Valid Metric:{valid_metric}")
                        break
                logger.info(f"Valid Metric:{valid_metric}")
        if self.use_wandb:
            wandb.finish()
        return valid_metric

    def save_model(self, model, model_ckpt_dir: str):
        """
        Save the model to the specified directory.

        Args:
            model (nn.Module): The model to be saved.
            model_ckpt_dir (str): The directory for saving the model.
        """
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Model Saved to {model_ckpt_dir}')

    def save_all(self, model, enc_dict: dict, model_ckpt_dir: str):
        """
        Save the model and encoding dictionary to the specified directory.

        Args:
            model (nn.Module): The model to be saved.
            enc_dict (dict): The encoding dictionary.
            model_ckpt_dir (str): The directory for saving the model and encoding dictionary.
        """
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict(),
                     'enc_dict': enc_dict}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Enc_dict and Model Saved to {model_ckpt_dir}')

    def save_train_model(self, model, model_ckpt_dir: str, model_str: str):
        """
        Save the model during training to the specified directory with a specified name.

        Args:
            model (nn.Module): The model to be saved.
            model_ckpt_dir (str): The directory for saving the model.
            model_str (str): The string to append to the model file name.
        """
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict, os.path.join(model_ckpt_dir, f'model_{model_str}.pth'))
        logger.info(f'Model Saved to {model_ckpt_dir}')

    def evaluate_model(self, model, test_loader, device: torch.device = torch.device('cpu')):
        """
        Evaluate the model using the given test loader and device.

        Args:
            model (nn.Module): The model to be evaluated.
            test_loader (DataLoader): The data loader for test data.
            device (torch.device, optional): The device to evaluate the model on. Defaults to 'cpu'.

        Returns:
            dict: The evaluation metrics.
        """
        test_metric = test_model(model, test_loader, device, num_task=self.num_task)
        logger.info(f"Test Metric:{beautify_json(test_metric)}")
        return test_metric

    def predict_dataloader(self, model, test_loader, device: torch.device = torch.device('cpu')):
        """
        Make predictions for the data in the given data loader using the model and device.

        Args:
            model (nn.Module): The model to make predictions with.
            test_loader (DataLoader): The data loader for test data.
            device (torch.device, optional): The device to
            make predictions on. Defaults to 'cpu'.

            Returns:
                list: A list of predictions.
            """
        model.eval()
        if self.num_task == 1:
            pred_list = []
            for data in test_loader:
                for key in data.keys():
                    data[key] = data[key].to(device)
                output = model(data, is_training=False)
                pred = output['pred']
                pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            return pred_list
        else:
            multi_task_pred_list = [[] for _ in range(self.num_task)]
            for data in test_loader:
                for key in data.keys():
                    data[key] = data[key].to(device)
                output = model(data, is_training=False)
                for i in range(self.num_task):
                    multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
            return multi_task_pred_list

    def predict_dataframe(self, model, test_df, enc_dict: dict, schema: dict,
                          device: torch.device = torch.device('cpu'), batch_size: int = 1024):
        """
        Make predictions for the data in the given DataFrame using the model, encoding dictionary, and schema.

        Args:
            model (nn.Module): The model to make predictions with.
            test_df (pd.DataFrame): The DataFrame containing the test data.
            enc_dict (dict): The encoding dictionary.
            schema (dict): The schema describing the task type.
            device (torch.device, optional): The device to make predictions on. Defaults to 'cpu'.
            batch_size (int, optional): The batch size to use when creating the data loader. Defaults to 1024.

        Returns:
            list: A list of predictions.
        """
        if schema['task_type'] == 'ranking':
            test_dataset = BaseDataset(schema, test_df, enc_dict=enc_dict)
        elif schema['task_type'] == 'multitask':
            test_dataset = MultiTaskDataset(schema, test_df, enc_dict=enc_dict)
        test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return self.predict_dataloader(model, test_loader, device=device)


class SequenceTrainer:
    """
    Sequence Trainer class for training and evaluating sequence models.
    """

    def __init__(self, wandb_config: Optional[dict] = None,
                 model_ckpt_dir: str = './model_ckpt'):
        """
        Initializes the SequenceTrainer class.

        Args:
            wandb_config (Optional[dict], optional): Wandb configuration dictionary. Defaults to None.
            model_ckpt_dir (str, optional): Directory for saving model checkpoints. Defaults to './model_ckpt'.
        """
        self.wandb_config = wandb_config
        self.log_df = pd.DataFrame()
        self.model_ckpt_dir = model_ckpt_dir
        self.use_wandb = self.wandb_config is not None
        if self.use_wandb:
            wandb.login(key=self.wandb_config['key'])
            self.wandb_config.pop('key')

    def fit(self, model, train_loader, valid_loader: Optional = None, epoch: int = 50, lr: float = 1e-3,
            device: torch.device = torch.device('cpu'), topk_list: Optional[List[int]] = None,
            use_earlystoping: bool = False, max_patience: int = 999, monitor_metric: Optional[str] = None,
            log_rounds: int = 100):
        """
        Fits the model using the given data loaders.

        Args:
            model: The model to train.
            train_loader: DataLoader for training data.
            valid_loader (Optional): DataLoader for validation data. Defaults to None.
            epoch (int, optional): Number of training epochs. Defaults to 50.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            device (torch.device, optional): Device to train the model on. Defaults to torch.device('cpu').
            topk_list (Optional[List[int]], optional): List of top-k values to compute metrics. Defaults to None.
            use_earlystoping (bool, optional): Whether to use early stopping. Defaults to False.
            max_patience (int, optional): Maximum number of epochs without improvement for early stopping. Defaults to 999.
            monitor_metric (Optional[str], optional): Metric to monitor for early stopping. Defaults to None.
            log_rounds (int, optional): Number of training rounds between logging. Defaults to 100.
        """
        if topk_list is None:
            topk_list = [20, 50, 100]

        if self.use_wandb:
            wandb.init(
                **self.wandb_config
            )

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model = model.to(device)

        # Training process
        logger.info('Model Starting Training')
        best_epoch = -1
        best_metric = -1
        best_metric_dict = dict()

        for i in range(1, epoch + 1):
            # Train the model
            train_sequence_model(model, train_loader, optimizer=optimizer, device=device,
                                 use_wandb=self.use_wandb, log_rounds=log_rounds)

            # Validate the model
            if valid_loader is not None:
                valid_metric = test_sequence_model(model=model, test_loader=valid_loader, topk_list=topk_list,
                                                   device=device, use_wandb=self.use_wandb)
                valid_metric['phase'] = 'valid'
                self.log_df = self.log_df.append(valid_metric, ignore_index=True)
                model_str = f'e_{i}'
                self.save_train_model(model, self.model_ckpt_dir, model_str)
                self.log_df.to_csv(os.path.join(self.model_ckpt_dir, 'log.csv'), index=False)
                if self.use_wandb:
                    wandb.log(valid_metric)
                if use_earlystoping:
                    assert monitor_metric in valid_metric.keys(), f'{monitor_metric} not in Valid Metric {valid_metric.keys()}'
                    if valid_metric[monitor_metric] > best_metric:
                        best_epoch = i
                        best_metric = valid_metric[monitor_metric]
                        best_metric_dict = valid_metric
                        self.save_train_model(model, self.model_ckpt_dir, 'best')
                    if i - best_epoch >= max_patience:
                        logger.info(f"EarlyStopping at the Epoch {best_epoch} Valid Metric:{best_metric_dict}")
                        break
                logger.info(f"Valid Metric:{valid_metric}")

    def evaluate_model(self, model, test_loader, device: torch.device = torch.device('cpu'),
                       topk_list: Optional[List[int]] = None) -> dict:
        """
        Evaluates the model using the given test loader.

        Args:
            model: The model to evaluate.
            test_loader: DataLoader for test data.
            device (torch.device, optional): Device to evaluate the model on. Defaults to torch.device('cpu').
            topk_list (Optional[List[int]], optional): List of top-k values to compute metrics. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: Top-K Recommendation Metric.
        """
        if topk_list is None:
            topk_list = [20, 50, 100]
        test_metric = test_sequence_model(model=model, test_loader=test_loader, topk_list=topk_list,
                                          device=device, use_wandb=self.use_wandb)
        test_metric['phase'] = 'test'
        self.log_df = self.log_df.append(test_metric, ignore_index=True)
        self.log_df.to_csv(os.path.join(self.model_ckpt_dir, 'log.csv'), index=False)
        logger.info(f"Test Metric:{test_metric}")
        if self.use_wandb:
            wandb.finish()

        return test_metric

    def save_model(self, model, model_ckpt_dir: str):
        """
        Saves the model to the specified directory.

        Args:
            model: The model to save.
            model_ckpt_dir (str): Directory to save the model.
        """
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Model Saved to {model_ckpt_dir}')

    def save_all(self, model, enc_dict, model_ckpt_dir: str):
        """
        Saves the model and encoder dictionary to the specified directory.

        Args:
            model: The model to save.
            enc_dict: The encoder dictionary to save.
            model_ckpt_dir (str): Directory to save the model and encoder dictionary.
        """
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict(),
                     'enc_dict': enc_dict}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Enc_dict and Model Saved to {model_ckpt_dir}')

    def save_train_model(self, model, model_ckpt_dir: str, model_str: str):
        """
        Saves the model during training to the specified directory.

        Args:
            model: The model to save.
            model_ckpt_dir (str): Directory to save the model.
            model_str (str): String to add to the model file name.
        """
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict, os.path.join(model_ckpt_dir, f'model_{model_str}.pth'))
        logger.info(f'Model Saved to {model_ckpt_dir}')


class GraphTrainer:
    def __init__(self):
        logger.info("Graph Trainer")

    def fit(self, model, train_dataset, epoch, lr, device=torch.device('cpu'), batch_size=1024):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        for i in range(1, epoch + 1):
            epoch_loss = train_graph_model(model=model, train_dataset=train_dataset, optimizer=optimizer, device=device,
                                           batch_size=batch_size)
            logger.info(f"Epoch:{i}/{epoch} Train Loss:{epoch_loss}")

    def save_model(self, model, model_ckpt_dir):
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))

    def evaluate_model(self, model, train_dataset, test_dataset, hidden_size, topN=50):
        train_gd = train_dataset.generate_test_gd()
        test_gd = test_dataset.generate_test_gd()
        test_metric = test_graph_model(model, train_gd=train_gd, test_gd=test_gd, hidden_size=hidden_size, topN=topN)
        logger.info(f"Test Metric:{beautify_json(test_metric)}")
        return test_metric
