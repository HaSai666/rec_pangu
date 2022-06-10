# -*- ecoding: utf-8 -*-
# @ModuleName: xdeepfm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, MLP_Layer, LR_Layer, CompressedInteractionNet
from ..utils import get_feature_num, get_linear_input

class xDeepFM(nn.Module):
    def __init__(self,
                 embedding_dim=10,
                 dnn_hidden_units=[64, 64, 64],
                 cin_layer_units = [16,16,16],
                 loss_fun='torch.nn.BCELoss()',
                 enc_dict=None):
        super(xDeepFM, self).__init__()

        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)

        self.dnn = MLP_Layer(input_dim=self.num_sparse*self.embedding_dim,
                             output_dim=1,
                             hidden_units=self.dnn_hidden_units)
        self.lr_layer = LR_Layer(enc_dict=self.enc_dict)
        self.cin = CompressedInteractionNet(self.num_sparse, cin_layer_units, output_dim=1)


    def forward(self, data):

        sparse_emb_list = self.embedding_layer(data)
        feature_emb = torch.stack(sparse_emb_list, dim=1).squeeze(2)
        lr_logit = self.lr_layer(data)
        cin_logit = self.cin(feature_emb)
        if self.dnn is not None:
            dnn_logit = self.dnn(feature_emb.flatten(start_dim=1))
            y_pred = lr_logit + cin_logit + dnn_logit
        else:
            y_pred = lr_logit + cin_logit

        y_pred = y_pred.sigmoid()
        # 输出
        loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
        output_dict = {'pred': y_pred, 'loss': loss}
        return output_dict
