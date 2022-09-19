# -*- ecoding: utf-8 -*-
# @ModuleName: base_dataset
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class BaseDataset(Dataset):
    def __init__(self,config,df,enc_dict=None):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        self.df = self.df.rename(columns={self.config['label_col']:'label'})
        self.dense_cols = list(set(self.config['dense_cols']))
        self.sparse_cols = list(set(self.config['sparse_cols']))
        self.feature_name = self.dense_cols+self.sparse_cols

        #数据编码
        if self.enc_dict == None:
            self.get_enc_dict()
        self.enc_data()

    def get_enc_dict(self):
        #计算enc_dict
        self.enc_dict = dict(zip( list(self.dense_cols+self.sparse_cols),[dict() for _ in range(len(self.dense_cols+self.sparse_cols))]))
        for f in self.sparse_cols:
            self.df[f] = self.df[f].astype('str')
            map_dict = dict(zip(sorted(self.df[f].unique()), range(self.df[f].nunique())))
            self.enc_dict[f] = map_dict
            self.enc_dict[f]['vocab_size'] = self.df[f].nunique()

        for f in self.dense_cols:
            self.enc_dict[f]['min'] = self.df[f].min()
            self.enc_dict[f]['max'] = self.df[f].max()

        return self.enc_dict

    def enc_dense_data(self,col):
        return (self.df[col] - self.enc_dict[col]['min']) / (self.enc_dict[col]['max'] - self.enc_dict[col]['min'] + 1e-5)

    def enc_sparse_data(self,col):
        return self.df[col].apply(lambda x : self.enc_dict[col].get(x,self.enc_dict[col]['vocab_size']))

    def enc_data(self):
        #使用enc_dict对数据进行编码
        self.enc_data = defaultdict(np.array)

        for col in self.dense_cols:
            self.enc_data[col] = torch.Tensor(np.array(self.enc_dense_data(col)))
        for col in self.sparse_cols:
            self.enc_data[col] = torch.Tensor(np.array(self.enc_sparse_data(col))).long()

    def __getitem__(self, index):
        data = defaultdict(np.array)
        for col in self.dense_cols:
            data[col] = self.enc_data[col][index]
        for col in self.sparse_cols:
            data[col] = self.enc_data[col][index]
        if 'label' in self.df.columns :
            data['label'] = torch.Tensor([self.df['label'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.df)




