# -*- ecoding: utf-8 -*-
# @ModuleName: afn
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict
from torch import nn
import torch
from ..layers import EmbeddingLayer, MLP
from ..utils import get_feature_num
from ..base_model import BaseModel


class AFN(BaseModel):
    def __init__(self,
                 embedding_dim=32,
                 dnn_hidden_units=[64, 64, 64],
                 afn_hidden_units=[64, 64, 64],
                 ensemble_dnn=True,
                 loss_fun='torch.nn.BCELoss()',
                 logarithmic_neurons=5,
                 enc_dict=None):
        super(AFN, self).__init__(enc_dict, embedding_dim)
        """
        AFN model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            dnn_hidden_units (List[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            afn_hidden_units (List[int]): The list of hidden units for the AFN. Default is [64, 64, 64].
            ensemble_dnn (bool): Whether to use ensemble DNN. Default is True.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            logarithmic_neurons (int): The number of logarithmic neurons. Default is 5.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """
        self.dnn_hidden_units = dnn_hidden_units
        self.afn_hidden_units = afn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.coefficient_W = nn.Linear(self.num_sparse, logarithmic_neurons, bias=False)

        self.dense_layer = MLP(input_dim=embedding_dim * logarithmic_neurons,
                               output_dim=1,
                               hidden_units=afn_hidden_units,
                               use_bias=True)
        self.log_batch_norm = nn.BatchNorm1d(self.num_sparse)
        self.exp_batch_norm = nn.BatchNorm1d(logarithmic_neurons)
        self.ensemble_dnn = ensemble_dnn
        self.apply(self._init_weights)

        if ensemble_dnn:
            self.embedding_layer2 = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
            self.dnn = MLP(input_dim=embedding_dim * self.num_sparse,
                           output_dim=1,
                           hidden_units=dnn_hidden_units,
                           use_bias=True)
            self.fc = nn.Linear(2, 1)

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the AFN model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """

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
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict

    def logarithmic_net(self, feature_emb):
        feature_emb = torch.abs(feature_emb)
        feature_emb = torch.clamp(feature_emb, min=1e-5)  # ReLU with min 1e-5 (better than 1e-7 suggested in paper)
        log_feature_emb = torch.log(feature_emb)  # element-wise log
        log_feature_emb = self.log_batch_norm(log_feature_emb)  # batch_size * num_fields * embedding_dim
        logarithmic_out = self.coefficient_W(log_feature_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(logarithmic_out)  # element-wise exp
        cross_out = self.exp_batch_norm(cross_out)  # batch_size * logarithmic_neurons * embedding_dim
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out
