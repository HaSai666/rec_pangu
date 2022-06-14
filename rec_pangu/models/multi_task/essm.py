# -*- ecoding: utf-8 -*-
# @ModuleName: essm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from torch import nn
from ..layers import EmbeddingLayer,MLP_Layer
from ..utils import get_feature_num

class ESSM(nn.Module):
    def __init__(self,
                 embedding_dim=40,
                 hidden_dim=[128, 64],
                 dropouts=[0.2, 0.2],
                 enc_dict=None,
                 device=None):
        super(ESSM, self).__init__()
        self.enc_dict = enc_dict
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)

        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)

        hidden_size = self.num_sparse_fea * self.embedding_dim

        self.ctr_layer = MLP_Layer(input_dim=hidden_size, output_dim=1, hidden_units=self.hidden_dim,
                                   hidden_activations='relu', dropout_rates=self.dropouts)

        self.cvr_layer = MLP_Layer(input_dim=hidden_size, output_dim=1, hidden_units=self.hidden_dim,
                                   hidden_activations='relu', dropout_rates=self.dropouts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        click = self.sigmoid(self.ctr_layer(hidden))
        conversion = self.sigmoid(self.cvr_layer(hidden))

        click = click * conversion

        # get loss
        loss = self.loss(click, conversion, data)
        output_dict = {
            'task1_pred': click,
            'task2_pred': conversion,
            'loss': loss
        }

        return output_dict

    def loss(self, click, conversion, data, weight=0.5):
        ctr_loss = nn.functional.binary_cross_entropy(click.squeeze(-1), data['task1_label'])
        cvr_loss = nn.functional.binary_cross_entropy(conversion.squeeze(-1), data['task2_label'])

        loss = cvr_loss + weight * ctr_loss

        return loss