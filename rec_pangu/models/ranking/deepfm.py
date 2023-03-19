# -*- ecoding: utf-8 -*-
# @ModuleName: deepfm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, FM_Layer, MLP_Layer
from ..utils import get_dnn_input_dim, get_linear_input
from ..base_model import BaseModel
class DeepFM(BaseModel):
    def __init__(self,
                 embedding_dim=32,
                 hidden_units=[64, 64, 64],
                 loss_fun = 'torch.nn.BCELoss()',
                 enc_dict=None):
        super(DeepFM, self).__init__(enc_dict,embedding_dim)

        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.fm = FM_Layer()
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units,
                                 hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        f""" 
        Perform forward propagation on the DeepFM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        sparse_embedding = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        # FM
        fm_out = self.fm(sparse_embedding)
        # DNN
        emb_flatten = sparse_embedding.flatten(start_dim=1)
        dnn_input = torch.cat((emb_flatten, dense_input), dim=1)
        dnn_output = self.dnn(dnn_input)

        y_pred = torch.sigmoid(fm_out + dnn_output)
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1),data['label'])
            output_dict = {'pred':y_pred,'loss':loss}
        else:
            output_dict = {'pred':y_pred}
        return output_dict

