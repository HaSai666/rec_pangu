# -*- ecoding: utf-8 -*-
# @ModuleName: dcn
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/11 12:56 PM
from typing import Dict, List
from torch import nn
import torch
from ..layers import CrossNet
from ..utils import get_linear_input, get_feature_num
from ..base_model import BaseModel


class DCN(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 hidden_units: List[int] = [64, 64, 64],
                 crossing_layers: int = 3,
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None):
        super(DCN, self).__init__(enc_dict, embedding_dim)
        """
        Deep & Cross Network (DCN) model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            hidden_units (list[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            crossing_layers (int): num of cross layers in DCN Model.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        self.dnn_hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        input_dim = self.num_sparse * self.embedding_dim + self.num_dense
        self.crossnet = CrossNet(input_dim, crossing_layers)

        self.fc = nn.Linear(input_dim, 1)

        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the DCN model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(torch.cat([flat_feature_emb, dense_input], dim=1))
        y_pred = self.fc(cross_out).sigmoid()

        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict
