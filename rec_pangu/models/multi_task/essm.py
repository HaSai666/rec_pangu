# -*- ecoding: utf-8 -*-
# @ModuleName: essm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from torch import nn
from ..layers import EmbeddingLayer,MLP_Layer
from ..utils import get_feature_num
from ..base_model import BaseModel
class ESSM(BaseModel):
    def __init__(self,
                 embedding_dim=40,
                 hidden_dim=[128, 64],
                 dropouts=[0.2, 0.2],
                 enc_dict=None,
                 device=None):
        super(ESSM, self).__init__(enc_dict,embedding_dim)
        self.enc_dict = enc_dict
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts

        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)

        hidden_size = self.num_sparse_fea * self.embedding_dim

        self.ctr_layer = MLP_Layer(input_dim=hidden_size, output_dim=1, hidden_units=self.hidden_dim,
                                   hidden_activations='relu', dropout_rates=self.dropouts)

        self.cvr_layer = MLP_Layer(input_dim=hidden_size, output_dim=1, hidden_units=self.hidden_dim,
                                   hidden_activations='relu', dropout_rates=self.dropouts)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def forward(self, data,is_training=True):
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        click = self.sigmoid(self.ctr_layer(hidden))
        conversion = self.sigmoid(self.cvr_layer(hidden))

        pctrcvr = click * conversion

        # get loss
        if is_training:
            loss = self.loss(click, pctrcvr, data)
            output_dict = {
                'task1_pred': click,
                'task2_pred': conversion,
                'loss': loss
            }
        else:
            output_dict = {
                'task1_pred': click,
                'task2_pred': conversion,
            }
        return output_dict

    def loss(self, click, conversion, data, weight=0.5):
        ctr_loss = nn.functional.binary_cross_entropy(click.squeeze(-1), data['task1_label'])
        cvr_loss = nn.functional.binary_cross_entropy(conversion.squeeze(-1), data['task2_label'])

        loss = cvr_loss + weight * ctr_loss

        return loss