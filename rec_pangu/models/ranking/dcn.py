# -*- ecoding: utf-8 -*-
# @ModuleName: dcn
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/11 12:56 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, MLP_Layer, CrossNet
from ..utils import get_dnn_input_dim, get_linear_input, get_feature_num
from ..base_model import BaseModel
class DCN(BaseModel):
    def __init__(self,
                 embedding_dim=32,
                 dnn_hidden_units=[64, 64, 64],
                 loss_fun = 'torch.nn.BCELoss()',
                 crossing_layers = 3,
                 enc_dict=None):
        super(DCN, self).__init__(enc_dict,embedding_dim)

        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        input_dim = self.num_sparse * self.embedding_dim + self.num_dense
        self.crossnet = CrossNet(input_dim, crossing_layers)

        self.fc = nn.Linear(input_dim, 1)

        self.apply(self._init_weights)

    def forward(self, data,is_training=True):
        feature_emb = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(torch.cat([flat_feature_emb, dense_input],dim=1))
        y_pred = self.fc(cross_out).sigmoid()

        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1),data['label'])
            output_dict = {'pred':y_pred,'loss':loss}
        else:
            output_dict = {'pred':y_pred}
        return output_dict

