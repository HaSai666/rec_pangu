# -*- ecoding: utf-8 -*-
# @ModuleName: multi_task_dataset
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from torch.utils.data import Dataset
import torch.utils.data as D
from .base_dataset import  BaseDataset
import copy
from collections import defaultdict
import numpy as np

class MultiTaskDataset(BaseDataset):
    def __init__(self,config,df,enc_dict=None):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        for idx, col in enumerate(self.config['label_col']):
            self.df = self.df.rename(columns={col :f'task{idx + 1}_label'})
        self.dense_cols = list(set(self.config['dense_cols']))
        self.sparse_cols = list(set(self.config['sparse_cols']))
        self.feature_name = self.dense_cols+self.sparse_cols+['label']

        #数据编码
        if self.enc_dict == None:
            self.get_enc_dict()
        self.enc_data()

    def __getitem__(self, index):
        data = defaultdict(np.array)
        for col in self.dense_cols:
            data[col] = self.enc_data[col][index]
        for col in self.sparse_cols:
            data[col] = self.enc_data[col][index]
        for idx, col in enumerate(self.config['label_col']):
            if f'task{idx + 1}_label' in self.df.columns:
                data[f'task{idx + 1}_label'] = torch.Tensor([self.df[f'task{idx + 1}_label'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.df)

