# -*- ecoding: utf-8 -*-
# @ModuleName: fm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, MLP_Layer, FM_Layer
from ..utils import get_feature_num, get_linear_input
from ..base_model import BaseModel
class FM(BaseModel):
    def __init__(self,
                 embedding_dim=32,
                 loss_fun='torch.nn.BCELoss()',
                 enc_dict=None):
        super(FM, self).__init__(enc_dict,embedding_dim)

        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.fm = FM_Layer()

        self.apply(self._init_weights)

    def forward(self, data,is_training=True):
        feature_emb = self.embedding_layer(data)
        y_pred = self.fm(feature_emb)
        y_pred = y_pred.sigmoid()
        # 输出
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1),data['label'])
            output_dict = {'pred':y_pred,'loss':loss}
        else:
            output_dict = {'pred':y_pred}
        return output_dict
