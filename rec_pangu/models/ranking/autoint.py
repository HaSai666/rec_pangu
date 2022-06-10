# -*- ecoding: utf-8 -*-
# @ModuleName: autoint
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, MLP_Layer, LR_Layer, MultiHeadSelfAttention
from ..utils import get_feature_num


class AutoInt(nn.Module):
    def __init__(self,
                 embedding_dim=32,
                 dnn_hidden_units=[64, 64, 64],
                 attention_layers = 1,
                 num_heads = 1,
                 attention_dim = 8,
                 loss_fun='torch.nn.BCELoss()',
                 enc_dict=None):
        super(AutoInt, self).__init__()

        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)

        self.lr_layer = LR_Layer(enc_dict=enc_dict)

        self.dnn = MLP_Layer(input_dim=self.embedding_dim* self.num_sparse,
                             output_dim=1,
                             hidden_units=self.dnn_hidden_units)

        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(self.embedding_dim if i == 0 else num_heads * attention_dim,
                                     attention_dim=attention_dim,
                                     num_heads=num_heads,
                                     align_to="output")
              for i in range(attention_layers)])
        self.fc = nn.Linear(self.num_sparse * attention_dim * num_heads, 1)

    def forward(self, data):

        sparse_emb_list = self.embedding_layer(data)
        feature_emb = torch.stack(sparse_emb_list, dim=1).squeeze(2)
        print(feature_emb.shape)

        attention_out = self.self_attention(feature_emb)
        attention_out = attention_out.flatten(start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(data)
        y_pred = y_pred.sigmoid()
        # 输出
        loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
        output_dict = {'pred': y_pred, 'loss': loss}
        return output_dict
