# -*- ecoding: utf-8 -*-
# @ModuleName: autoint
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict, List
from torch import nn
import torch
from ..layers import MLP, LR_Layer, MultiHeadSelfAttention
from ..utils import get_feature_num, get_linear_input
from ..base_model import BaseModel


class AutoInt(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 dnn_hidden_units: List[int] = [64, 64, 64],
                 attention_layers: int = 1,
                 num_heads: int = 1,
                 attention_dim: int = 8,
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None):
        super(AutoInt, self).__init__(enc_dict, embedding_dim)
        """
        AutoInt model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            dnn_hidden_units (List[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            attention_layers (int): The number of attention layers. Default is 1.
            num_heads (int): The number of attention heads. Default is 1.
            attention_dim (int): The dimension of the attention layer. Default is 8.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """

        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)

        self.lr_layer = LR_Layer(enc_dict=enc_dict)

        self.dnn = MLP(input_dim=self.embedding_dim * self.num_sparse + self.num_dense,
                       output_dim=1,
                       hidden_units=self.dnn_hidden_units)

        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(self.embedding_dim if i == 0 else num_heads * attention_dim,
                                     attention_dim=attention_dim,
                                     num_heads=num_heads,
                                     align_to="output")
              for i in range(attention_layers)])
        self.fc = nn.Linear(self.num_sparse * attention_dim * num_heads, 1)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the AutoInt model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """

        feature_emb = self.embedding_layer(data)
        attention_out = self.self_attention(feature_emb)
        attention_out = attention_out.flatten(start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            dense_input = get_linear_input(self.enc_dict, data)
            emb_flatten = feature_emb.flatten(start_dim=1)
            y_pred += self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(data)
        y_pred = y_pred.sigmoid()
        # 输出
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict
