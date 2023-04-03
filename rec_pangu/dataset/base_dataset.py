# -*- encoding: utf-8 -*-
# @ModuleName: base_dataset
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
from collections import defaultdict


class BaseDataset(Dataset):
    """
    This class implements a BaseDataset that inherits from Pytorch's Dataset class for loading and encoding data.

    Args:
        config: a dictionary that specifies the configuration of the dataset.
        df (pd.DataFrame): a Pandas DataFrame that contains the data to be loaded.
        enc_dict (Dict[str, dict], optional): a dictionary of encoding values for the data. Defaults to None.

    Attributes:
        config (dict): A dictionary that specifies dataset parameters.
        df(pd.DataFrame): A Pandas DataFrame that contains the data to be loaded.
        enc_dict(Dict[str, dict]): A dictionary of encoding values for the given data.
        dense_cols(list): A list of the dense columns extracted from the given data.
        sparse_cols(list): A list of the sparse columns extracted from the given data.
        feature_name(list): A list of all the extracted features from the given data.
        data(defaultdict): A dictionary containing the encoded data using the provided encoding dictionary.
    """

    def __init__(self, config: dict, df: pd.DataFrame, enc_dict: Dict[str, dict] = None):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        self.df = self.df.rename(columns={self.config['label_col']: 'label'})
        self.dense_cols = list(set(self.config['dense_cols']))
        self.sparse_cols = list(set(self.config['sparse_cols']))
        self.feature_name = self.dense_cols + self.sparse_cols

        if self.enc_dict is None:
            self.get_enc_dict()

        self.enc_data()

    def get_enc_dict(self) -> Dict[str, dict]:
        """
        This method generates the enc_dict for the dataset by encoding the sparse and dense data.

        Returns:
            A dictionary containing the encoding values for the dataset.
        """
        self.enc_dict = dict(zip(
            list(self.dense_cols + self.sparse_cols), [dict() for _ in range(len(self.dense_cols + self.sparse_cols))]))

        for f in self.sparse_cols:
            self.df[f] = self.df[f].astype('str')
            map_dict = dict(zip(sorted(self.df[f].unique()), range(self.df[f].nunique())))
            self.enc_dict[f] = map_dict
            self.enc_dict[f]['vocab_size'] = self.df[f].nunique()

        for f in self.dense_cols:
            self.enc_dict[f]['min'] = self.df[f].min()
            self.enc_dict[f]['max'] = self.df[f].max()

        return self.enc_dict

    def enc_dense_data(self, col: str) -> torch.Tensor:
        """
        This method encodes the given dense data column using the encoding dictionary.

        Args:
            col (str): The name of the column to be encoded.

        Returns:
            A torch.Tensor containing the encoded data.
        """
        return (self.df[col] - self.enc_dict[col]['min']) / (
                self.enc_dict[col]['max'] - self.enc_dict[col]['min'] + 1e-5)

    def enc_sparse_data(self, col: str) -> torch.Tensor:
        """
        This method encodes the given sparse data column using the encoding dictionary.

        Args:
            col (str): The name of the column to be encoded.

        Returns:
            A torch.Tensor containing the encoded data.
        """
        return self.df[col].apply(lambda x: self.enc_dict[col].get(x, self.enc_dict[col]['vocab_size']))

    def enc_data(self):
        """
        This method encodes the dataset using the encoding dictionary and stores it in the self.enc_data dictionary.
        """
        self.data_dict = defaultdict(np.array)

        for col in self.dense_cols:
            self.data_dict[col] = torch.Tensor(np.array(self.enc_dense_data(col)))
        for col in self.sparse_cols:
            self.data_dict[col] = torch.Tensor(np.array(self.enc_sparse_data(col))).long()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        This method returns a dictionary containing the encoded data at the given index.

        Args:
            index (int): The index of the row to return.

        Returns:
            A dictionary containing the encoded data at the given index.
        """
        data = {}

        for col in self.dense_cols:
            data[col] = self.data_dict[col][index]
        for col in self.sparse_cols:
            data[col] = self.data_dict[col][index]
        if 'label' in self.df.columns:
            data['label'] = torch.Tensor([self.df['label'].iloc[index]]).squeeze(-1)

        return data

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.df)
