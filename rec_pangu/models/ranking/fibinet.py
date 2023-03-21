# -*- ecoding: utf-8 -*-
# @ModuleName: fibinet
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict, List
import torch
from ..layers import LR_Layer, MLP, BilinearInteractionLayer, SENET_Layer
from ..utils import get_feature_num, get_linear_input
from ..base_model import BaseModel


class FiBiNet(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 hidden_units: List[int] = [64, 64, 64],
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None):
        """
        FiBiNet model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            hidden_units (list[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        super(FiBiNet, self).__init__(enc_dict, embedding_dim)

        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)

        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.senet_layer = SENET_Layer(self.num_sparse, 3)
        self.bilinear_interaction = BilinearInteractionLayer(self.num_sparse, embedding_dim, 'field_interaction')

        input_dim = self.num_sparse * (self.num_sparse - 1) * self.embedding_dim + self.num_dense
        self.dnn = MLP(input_dim=input_dim, output_dim=1, hidden_units=self.hidden_units,
                       hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data: Dict[str, torch.Tensor],
                is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the FiBiNet model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        y_pred = self.lr(data)  # Batch,1

        feature_emb = self.embedding_layer(data)
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction(feature_emb)
        bilinear_q = self.bilinear_interaction(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)

        dense_input = get_linear_input(self.enc_dict, data)
        comb_out = torch.cat([comb_out, dense_input], dim=1)
        y_pred += self.dnn(comb_out)
        y_pred = y_pred.sigmoid()

        # Output
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict
