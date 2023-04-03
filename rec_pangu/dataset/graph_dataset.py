from dgl import DGLGraph
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class GeneralGraphDataset(Dataset):
    def __init__(self, df, num_user, num_item, phase='train'):
        self.df = df
        self.n_item = self.df['item_id'].nunique()
        self.phase = phase
        self.num_user = num_user
        self.num_item = num_item
        self.generate_test_gd()
        if self.phase == 'train':
            self.encode_data()

    def encode_data(self):
        self.data = dict()
        self.data['user_id'] = torch.Tensor(np.array(self.df['user_id'])).long()
        self.data['item_id'] = torch.Tensor(np.array(self.df['item_id'])).long()

    def generate_test_gd(self):
        self.test_gd = self.df.groupby('user_id')['item_id'].apply(list).to_dict()
        self.user_list = list(self.test_gd.keys())

    def generate_graph(self):
        # torch.arange(self.num_user+self.num_item)
        src_node_list = torch.cat([self.data['user_id'], self.data['item_id'] + self.num_user], axis=0)
        dst_node_list = torch.cat([self.data['item_id'] + self.num_user, self.data['user_id']], axis=0)
        g = DGLGraph((src_node_list, dst_node_list))

        src_degree = g.out_degrees().float()
        norm = torch.pow(src_degree, -0.5).unsqueeze(1)  # compute norm
        g.ndata['norm'] = norm  # 节点粒度的norm
        return g

    def sample(self, batch_size=1024):
        users = random.sample(self.user_list, batch_size)

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.test_gd[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.num_item, size=1)[0]
                if neg_id not in self.test_gd[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        data = {
            'user_id': torch.Tensor(users).long(),
            'pos_item_id': torch.Tensor(pos_items).long(),
            'neg_item_id': torch.Tensor(neg_items).long()
        }

        return data

    def __getitem__(self, index):
        if self.phase == 'train':
            random_index = np.random.randint(0, self.num_item)
            while random_index in self.test_gd[self.df['user_id'].iloc[index]]:
                random_index = np.random.randint(0, self.num_item)
            neg_item_id = torch.Tensor([random_index]).squeeze().long()

            data = {
                'user_id': self.data['user_id'][index],
                'pos_item_id': self.data['item_id'][index],
                'neg_item_id': neg_item_id
            }
        else:
            data = {
                'user_id': torch.Tensor([self.user_list[index]]).squeeze().long(),
                'item_list': self.test_gd[self.user_list[index]]
            }
        return data

    def __len__(self):
        if self.phase == 'train':
            return len(self.df)
        else:
            return self.df['user_id'].nunique()
