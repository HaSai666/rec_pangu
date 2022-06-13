# -*- ecoding: utf-8 -*-
# @ModuleName: afn
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, MLP_Layer, LR_Layer, SENET_Layer, BilinearInteractionLayer
from ..utils import get_feature_num, get_linear_input

class AFN(nn.Module):
    def __init__(self,
                 embedding_dim=32,
                 dnn_hidden_units=[64, 64, 64],
                 afn_hidden_units = [64, 64, 64],
                 ensemble_dnn = True,
                 loss_fun='torch.nn.BCELoss()',
                 logarithmic_neurons = 5,
                 enc_dict=None):
        super(AFN, self).__init__()

        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.afn_hidden_units = afn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.coefficient_W = nn.Linear(self.num_sparse, logarithmic_neurons, bias=False)

        self.dense_layer = MLP_Layer(input_dim=embedding_dim * logarithmic_neurons,
                                     output_dim=1,
                                     hidden_units=afn_hidden_units,
                                     use_bias=True)
        self.log_batch_norm = nn.BatchNorm1d(self.num_sparse)
        self.exp_batch_norm = nn.BatchNorm1d(logarithmic_neurons)
        self.ensemble_dnn = ensemble_dnn

        if ensemble_dnn:
            self.embedding_layer2 = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
            self.dnn = MLP_Layer(input_dim=embedding_dim * self.num_sparse,
                                 output_dim=1,
                                 hidden_units=dnn_hidden_units,
                                 use_bias=True)
            self.fc = nn.Linear(2, 1)

    def forward(self, data):

        feature_emb = self.embedding_layer(data)
        dnn_input = self.logarithmic_net(feature_emb)
        afn_out = self.dense_layer(dnn_input)

        if self.ensemble_dnn:
            feature_emb2 = self.embedding_layer2(data)
            dnn_out = self.dnn(feature_emb2.flatten(start_dim=1))
            y_pred = self.fc(torch.cat([afn_out, dnn_out], dim=-1))
        else:
            y_pred = afn_out
        y_pred = y_pred.sigmoid()
        # 输出
        loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
        output_dict = {'pred': y_pred, 'loss': loss}
        return output_dict

    def logarithmic_net(self, feature_emb):
        feature_emb = torch.abs(feature_emb)
        feature_emb = torch.clamp(feature_emb, min=1e-5) # ReLU with min 1e-5 (better than 1e-7 suggested in paper)
        log_feature_emb = torch.log(feature_emb) # element-wise log
        log_feature_emb = self.log_batch_norm(log_feature_emb) # batch_size * num_fields * embedding_dim
        logarithmic_out = self.coefficient_W(log_feature_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(logarithmic_out) # element-wise exp
        cross_out = self.exp_batch_norm(cross_out)  # batch_size * logarithmic_neurons * embedding_dim
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out