# -*- ecoding: utf-8 -*-
# @ModuleName: shallow
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM

import torch
from torch import nn
from .embedding import EmbeddingLayer
from ..utils import get_dnn_input_dim, get_linear_input

# Wide部分
class LR_Layer(nn.Module):
    def __init__(self, enc_dict):
        super(LR_Layer, self).__init__()
        self.enc_dict = enc_dict
        self.emb_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=1)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, 1)
        self.fc = nn.Linear(self.dnn_input_dim, 1)

    def forward(self, data):
        sparse_emb = self.emb_layer(data).squeeze(-1)
        dense_input = get_linear_input(self.enc_dict, data)
        dnn_input = torch.cat((sparse_emb, dense_input), dim=1)
        out = self.fc(dnn_input)
        return out
