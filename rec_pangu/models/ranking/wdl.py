# -*- ecoding: utf-8 -*-
# @ModuleName: wdl
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from typing import Dict, List
from ..layers import MLP, LR_Layer
from ..utils import get_dnn_input_dim, get_linear_input
from ..base_model import BaseModel


class WDL(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 hidden_units: List[int] = [64, 64, 64],
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None) -> None:
        """
        Wide and Deep (WDL) Model

        Args:
            embedding_dim (int): Dimension of the embedding vectors. Defaults to 32.
            hidden_units (list): Number of units in each hidden layer of the MLP. Defaults to [64, 64, 64].
            loss_fun (str): String representation of the loss function. Defaults to 'torch.nn.BCELoss()'.
            enc_dict (dict): Dictionary for encoding input features.
        """
        super(WDL, self).__init__(enc_dict, embedding_dim)

        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        # Wide part
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        # Deep part
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units,
                       hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data: Dict[str, torch.Tensor], is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the WDL model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        # Wide
        wide_logit = self.lr(data)  # Batch, 1

        # Deep
        sparse_emb = self.embedding_layer(data)
        sparse_emb = sparse_emb.flatten(start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)
        dnn_input = torch.cat([sparse_emb, dense_input], dim=1)  # Batch, num_sparse_fea*embedding_dim+num_dense
        deep_logit = self.dnn(dnn_input)

        # Wide + Deep
        y_pred = (wide_logit + deep_logit).sigmoid()

        # Output
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}

        return output_dict
