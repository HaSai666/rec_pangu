# -*- ecoding: utf-8 -*-
# @ModuleName: sequence_dataset
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/3 14:07
import torch
from torch.utils.data import Dataset
import random


class SequenceDataset(Dataset):
    def __init__(self, config, df, enc_dict=None, phase='train'):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        self.max_length = self.config['max_length']
        self.user_col = self.config['user_col']
        self.item_col = self.config['item_col']
        self.time_col = self.config.get('time_col', None)
        self.cate_cols = self.config.get('cate_cols', [])

        if self.time_col:
            self.df = self.df.sort_values(by=[self.user_col, self.time_col])

        if self.enc_dict == None:
            self.get_enc_dict()
        self.enc_data()

        self.user2item = self.df.groupby(self.user_col)[self.item_col].apply(list).to_dict()
        for col in self.cate_cols:
            setattr(self, f'user2{col}', self.df.groupby(self.user_col)[col].apply(list).to_dict())

        self.user_list = self.df[self.user_col].unique()
        self.phase = phase

    def get_enc_dict(self):
        # 计算enc_dict
        if self.enc_dict == None:
            sparse_cols = [self.item_col] + self.cate_cols
            self.enc_dict = dict(zip(list(sparse_cols), [dict() for _ in range(len(sparse_cols))]))
            for f in [self.item_col] + self.cate_cols:
                self.df[f] = self.df[f].astype('str')
                map_dict = dict(zip(sorted(self.df[f].unique()), range(1, 1 + self.df[f].nunique())))
                self.enc_dict[f] = map_dict
                self.enc_dict[f]['vocab_size'] = self.df[f].nunique() + 1
        else:
            return self.enc_dict

    def enc_data(self):
        sparse_cols = [self.item_col] + self.cate_cols
        for f in sparse_cols:
            self.df[f] = self.df[f].astype('str')
            self.df[f] = self.df[f].apply(lambda x: self.enc_dict[f].get(x, 0))

    def __getitem__(self, index):
        user_id = self.user_list[index]
        item_list = self.user2item[user_id]
        hist_item_list = []
        hist_mask_list = []
        if self.phase == 'train':

            k = random.choice(range(4, len(item_list)))  # 从[4,len(item_list))中随机选择一个index
            item_id = item_list[k]  # 该index对应的item加入item_id_list

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length: k])
                hist_mask_list.append([1.0] * self.max_length)
                for col in self.cate_cols:
                    cate_seq = getattr(self, f'user2{col}')[user_id]
                    setattr(self, f'hist_{col}_list', cate_seq[k - self.max_length: k])
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))
                for col in self.cate_cols:
                    cate_seq = getattr(self, f'user2{col}')[user_id]
                    setattr(self, f'hist_{col}_list', cate_seq[:k] + [0] * (self.max_length - k))
            data = {
                'hist_item_list': torch.Tensor(hist_item_list).squeeze(0).long(),
                'hist_mask_list': torch.Tensor(hist_mask_list).squeeze(0).long(),
                'target_item': torch.Tensor([item_id]).long()
            }

            for col in self.cate_cols:
                data.update({f'hist_{col}_list': torch.Tensor(getattr(self, f'hist_{col}_list')).squeeze(0).long()})
        else:
            k = int(0.8 * len(item_list))
            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length: k])
                hist_mask_list.append([1.0] * self.max_length)
                for col in self.cate_cols:
                    cate_seq = getattr(self, f'user2{col}')[user_id]
                    setattr(self, f'hist_{col}_list', cate_seq[k - self.max_length: k])
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))
                for col in self.cate_cols:
                    cate_seq = getattr(self, f'user2{col}')[user_id]
                    setattr(self, f'hist_{col}_list', cate_seq[:k] + [0] * (self.max_length - k))
            data = {
                'user': user_id,
                'hist_item_list': torch.Tensor(hist_item_list).squeeze(0).long(),
                'hist_mask_list': torch.Tensor(hist_mask_list).squeeze(0).long(),
            }
            for col in self.cate_cols:
                data.update({f'hist_{col}_list': torch.Tensor(getattr(self, f'hist_{col}_list')).squeeze(0).long()})
        return data

    def __len__(self):
        return len(self.user_list)

    def get_test_gd(self):
        self.test_gd = dict()
        for user in self.user2item:
            item_list = self.user2item[user]
            test_item_index = int(0.8 * len(item_list))
            self.test_gd[str(user)] = item_list[test_item_index:]
        return self.test_gd


def seq_collate(batch):
    hist_item = torch.rand(len(batch), batch[0][0].shape[0])
    hist_mask = torch.rand(len(batch), batch[0][0].shape[0])
    item_list = []
    for i in range(len(batch)):
        hist_item[i, :] = batch[i][0]
        hist_mask[i, :] = batch[i][1]
        item_list.append(batch[i][2])
    hist_item = hist_item.long()
    hist_mask = hist_mask.long()
    return hist_item, hist_mask, item_list
