# -*- ecoding: utf-8 -*-
# @ModuleName: xdeepfm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, MLP_Layer, LR_Layer, CompressedInteractionNet
from ..utils import get_feature_num, get_linear_input
from ..base_model import BaseModel

class xDeepFM(BaseModel):
    def __init__(self,
                 embedding_dim=32,
                 dnn_hidden_units=[64, 64, 64],
                 cin_layer_units = [16,16,16],
                 loss_fun='torch.nn.BCELoss()',
                 enc_dict=None):
        super(xDeepFM, self).__init__(enc_dict,embedding_dim)

        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        # self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)

        self.dnn = MLP_Layer(input_dim=self.num_sparse*self.embedding_dim + self.num_dense,
                             output_dim=1,
                             hidden_units=self.dnn_hidden_units)
        self.lr_layer = LR_Layer(enc_dict=self.enc_dict)
        self.cin = CompressedInteractionNet(self.num_sparse, cin_layer_units, output_dim=1)

        self.apply(self._init_weights)

    def forward(self, data,is_training=True):

        feature_emb = self.embedding_layer(data)
        lr_logit = self.lr_layer(data)
        cin_logit = self.cin(feature_emb)
        if self.dnn is not None:
            dense_input = get_linear_input(self.enc_dict, data)
            emb_flatten = feature_emb.flatten(start_dim=1)
            dnn_logit = self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
            y_pred = lr_logit + cin_logit + dnn_logit
        else:
            y_pred = lr_logit + cin_logit

        y_pred = y_pred.sigmoid()
        # 输出
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1),data['label'])
            output_dict = {'pred':y_pred,'loss':loss}
        else:
            output_dict = {'pred':y_pred}
        return output_dict
