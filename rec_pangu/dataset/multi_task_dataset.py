# -*- ecoding: utf-8 -*-
# @ModuleName: multi_task_dataset
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
from .base_dataset import BaseDataset

class MultiTaskDataset(BaseDataset):
    """
    A dataset class for multi-task learning.

    Args:
        config: A dictionary containing the dataset configuration.
        df: A Pandas DataFrame containing the data.
        enc_dict: A dictionary containing the encoding functions for each column.

    Attributes:
        config (dict): The dataset configuration.
        df (DataFrame): The input data.
        enc_dict (dict): The encoding functions for each column.
        dense_cols (list): A list of dense columns.
        sparse_cols (list): A list of sparse columns.
        feature_name (list): A list of feature names.

    Methods:
        __getitem__(index: int) -> dict: Gets the data and labels for a given index.
        __len__() -> int: Gets the length of the dataset.
    """

    def __init__(self, config: dict, df: pd.DataFrame, enc_dict: dict = None) -> None:
        self.config = config
        self.df = df
        self.enc_dict = enc_dict

        # Rename the label columns according to the configuration.
        for idx, col in enumerate(self.config['label_col']):
            self.df = self.df.rename(columns={col: f'task{idx + 1}_label'})

        # Initialize the dense and sparse columns.
        self.dense_cols = list(set(self.config['dense_cols']))
        self.sparse_cols = list(set(self.config['sparse_cols']))

        # Initialize the feature names.
        self.feature_name = self.dense_cols + self.sparse_cols + ['label']

        # If an encoding dictionary is not provided, create one and apply it to the data.
        if self.enc_dict is None:
            self.get_enc_dict()
        self.data()

    def __getitem__(self, index: int) -> dict:
        """
        Gets the data and labels for a given index.

        Args:
            index (int): The index to retrieve.

        Returns:
            A dictionary containing the encoded data and labels.
        """
        data = defaultdict(np.array)
        for col in self.dense_cols:
            data[col] = self.data[col][index]
        for col in self.sparse_cols:
            data[col] = self.data[col][index]
        for idx, col in enumerate(self.config['label_col']):
            if f'task{idx + 1}_label' in self.df.columns:
                data[f'task{idx + 1}_label'] = torch.Tensor([self.df[f'task{idx + 1}_label'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self) -> int:
        """
        Gets the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.df)
