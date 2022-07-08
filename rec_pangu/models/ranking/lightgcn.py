# -*- ecoding: utf-8 -*-
# @ModuleName: run_ranking_example
# @Author: jeremy
# @Email: jeremydzwang@126.com
# @Time: 2022/6/20 11:40 PM


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import LGConv

class lightgcn(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers):
        super(lightgcn,self).__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.loss_fun = nn.MSELoss()

        alpha = 1. / (num_layers + 1)
        self.alpha = torch.tensor([alpha] * (num_layers + 1))

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.decoder = torch.nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim),
                                           nn.ReLU(), nn.Linear(embedding_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def get_embedding(self, edge_index):
        x = self.embedding.weight # 输入特征
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]
        return out

    def forward(self, edge_index, edge_label_index, edge_label):
        """预测节点对的rating"""
        out = self.get_embedding(edge_index) # 每个节点的embedding
        out_src = out[edge_label_index[0]] # 起始节点的embedding
        out_dst = out[edge_label_index[1]] # 终止节点的embedding
        pred = self.decoder(torch.cat([out_src, out_dst], dim=-1)).to(torch.float32)
        target = edge_label.view(-1, 1).to(torch.float32)
        loss = self.loss_fun(pred, target)

        result = dict()
        result['pred'] = pred
        result['loss'] = loss

        return result