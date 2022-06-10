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
        data = dict()
        for col in self.feature_name:
            if col in self.dense_cols:
                data[col] = torch.Tensor([self.enc_df[col].iloc[index]]).squeeze(-1)
            elif col in self.sparse_cols:
                data[col] = torch.Tensor([self.enc_df[col].iloc[index]]).long().squeeze(-1)
        for idx, col in enumerate(self.config['label_col']):
            data[f'task{idx + 1}_label'] = torch.Tensor([self.enc_df[f'task{idx + 1}_label'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.enc_df)

