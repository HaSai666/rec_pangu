# -*- ecoding: utf-8 -*-
# @ModuleName: nfm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, LR_Layer, MLP_Layer, InnerProductLayer
from ..utils import get_dnn_input_dim
from ..base_model import BaseModel
class NFM(BaseModel):
    def __init__(self,
                 embedding_dim=32,
                 hidden_units=[64, 64, 64],
                 loss_fun='torch.nn.BCELoss()',
                 enc_dict=None):
        super(NFM, self).__init__(enc_dict,embedding_dim)

        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.lr = LR_Layer(enc_dict=self.enc_dict)

        self.inner_product_layer = InnerProductLayer(output="Bi_interaction_pooling")
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP_Layer(input_dim=self.embedding_dim, output_dim=1, hidden_units=self.hidden_units,
                             hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data,is_training=True):
        y_pred = self.lr(data)  # Batch,1
        batch_size = y_pred.shape[0]

        sparse_embedding = self.embedding_layer(data)
        inner_product_tensor = self.inner_product_layer(sparse_embedding)
        bi_pooling_tensor = inner_product_tensor.view(batch_size, -1)
        y_pred += self.dnn(bi_pooling_tensor)
        y_pred = y_pred.sigmoid()

        # 输出
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1),data['label'])
            output_dict = {'pred':y_pred,'loss':loss}
        else:
            output_dict = {'pred':y_pred}
        return output_dict
