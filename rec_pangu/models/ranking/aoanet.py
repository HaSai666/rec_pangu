# -*- ecoding: utf-8 -*-
# @ModuleName: aoanet
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict, List
from torch import nn
import torch
from ..layers import MLP
from ..utils import get_feature_num, get_linear_input
from ..base_model import BaseModel


class AOANet(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 dnn_hidden_units: List[int] = [64, 64, 64],
                 num_interaction_layers: int = 3,
                 num_subspaces: int = 4,
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None):
        super(AOANet, self).__init__(enc_dict, embedding_dim)
        """
        AOANet model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            dnn_hidden_units (List[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            num_interaction_layers (int): The number of interaction layers in the Generalized Interaction Net. Default is 3.
            num_subspaces (int): The number of subspaces for the interaction layer. Default is 4.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """

        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)

        self.dnn = MLP(input_dim=self.embedding_dim * self.num_sparse + self.num_dense,
                       output_dim=None,
                       hidden_units=self.dnn_hidden_units)
        self.gin = GeneralizedInteractionNet(num_interaction_layers,
                                             num_subspaces,
                                             self.num_sparse,
                                             self.embedding_dim)
        self.fc = nn.Linear(dnn_hidden_units[-1] + num_subspaces * self.embedding_dim, 1)
        # self.apply(self._init_weights)
        self.reset_parameters()


    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the AoaNet model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        emb_flatten = feature_emb.flatten(start_dim=1)
        dnn_out = self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
        interact_out = self.gin(feature_emb).flatten(start_dim=1)
        y_pred = self.fc(torch.cat([dnn_out, interact_out], dim=-1))
        y_pred = y_pred.sigmoid()

        # 输出
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class GeneralizedInteractionNet(nn.Module):
    def __init__(self, num_layers, num_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteractionNet, self).__init__()
        self.layers = nn.ModuleList([GeneralizedInteraction(num_fields if i == 0 else num_subspaces,
                                                            num_subspaces,
                                                            num_fields,
                                                            embedding_dim) \
                                     for i in range(num_layers)])

    def forward(self, B_0):
        B_i = B_0
        for layer in self.layers:
            B_i = layer(B_0, B_i)
        return B_i


class GeneralizedInteraction(nn.Module):
    def __init__(self, input_subspaces, output_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteraction, self).__init__()
        self.input_subspaces = input_subspaces
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.eye(embedding_dim, embedding_dim).unsqueeze(0).repeat(output_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(input_subspaces * num_fields, output_subspaces))
        self.h = nn.Parameter(torch.ones(output_subspaces, embedding_dim, 1))

    def forward(self, B_0, B_i):
        outer_product = torch.einsum("bnh,bnd->bnhd",
                                     B_0.repeat(1, self.input_subspaces, 1),
                                     B_i.repeat(1, 1, self.num_fields).view(B_i.size(0), -1,
                                                                            self.embedding_dim))  # b x (field*in) x d x d
        fusion = torch.matmul(outer_product.permute(0, 2, 3, 1), self.alpha)  # b x d x d x out
        fusion = self.W * fusion.permute(0, 3, 1, 2)  # b x out x d x d
        B_i = torch.matmul(fusion, self.h).squeeze(-1)  # b x out x d
        return B_i
