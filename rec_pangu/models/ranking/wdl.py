# -*- ecoding: utf-8 -*-
# @ModuleName: wdl
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from torch import nn
from ..layers import EmbeddingLayer, MLP_Layer, LR_Layer
from ..utils import get_dnn_input_dim, get_linear_input

class WDL(nn.Module):
    def __init__(self,
                 embedding_dim=32,
                 hidden_units=[64, 64, 64],
                 loss_fun='torch.nn.BCELoss()',
                 enc_dict=None):
        super(WDL, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        # Wide部分
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        # Deep部分
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units,
                             hidden_activations='relu', dropout_rates=0)

    def forward(self, data):
        # Wide
        wide_logit = self.lr(data)  # Batch,1

        # Deep
        sparse_emb = self.embedding_layer(data)
        sparse_emb = sparse_emb.flatten(start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)
        dnn_input = torch.cat([sparse_emb, dense_input], dim=1)  # Batch,num_sparse_fea*embedding_dim+num_dense
        deep_logit = self.dnn(dnn_input)

        # Wide+Deep
        y_pred = (wide_logit + deep_logit).sigmoid()

        # 输出
        loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
        output_dict = {'pred': y_pred, 'loss': loss}
        return output_dict