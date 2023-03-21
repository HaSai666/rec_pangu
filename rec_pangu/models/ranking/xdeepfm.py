# -*- ecoding: utf-8 -*-
# @ModuleName: xdeepfm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from typing import Dict, List
from ..layers import MLP, LR_Layer, CompressedInteractionNet
from ..utils import get_feature_num, get_linear_input
from ..base_model import BaseModel


class xDeepFM(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 dnn_hidden_units: List[int] = [64, 64, 64],
                 cin_layer_units: List[int] = [16, 16, 16],
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None) -> None:
        """
        xDeepFM model.

        Args:
            embedding_dim (int): The dimension of the embedding vector. Default is 32.
            dnn_hidden_units (List[int]): The number of units in the MLP hidden layers. Default is [64, 64, 64].
            cin_layer_units (List[int]): The number of units in the CIN layers. Default is [16, 16, 16].
            loss_fun (str): String representation of the loss function. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): Dictionary for encoding input features.
        """
        super(xDeepFM, self).__init__(enc_dict, embedding_dim)

        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)

        self.dnn = MLP(input_dim=self.num_sparse * self.embedding_dim + self.num_dense,
                       output_dim=1,
                       hidden_units=self.dnn_hidden_units)
        self.lr_layer = LR_Layer(enc_dict=self.enc_dict)
        self.cin = CompressedInteractionNet(self.num_sparse, cin_layer_units, output_dim=1)

        self.apply(self._init_weights)

    def forward(self, data: Dict[str, torch.Tensor], is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the xDeepFM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """

        feature_emb = self.embedding_layer(data)
        lr_logit = self.lr_layer(data)
        cin_logit = self.cin(feature_emb)
        if self.dnn is not None:
            dense_input = get_linear_input(self.enc_dict, data)
            emb_flatten = feature_emb.flatten(start_dim=1)
            dnn_logit = self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
            y_pred = lr_logit + cin_logit + dnn_logit
        else:
            y_pred = lr_logit + cin_logit

        y_pred = y_pred.sigmoid()

        # Output
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict
