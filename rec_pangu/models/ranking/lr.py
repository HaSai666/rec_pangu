# -*- ecoding: utf-8 -*-
# @ModuleName: lr
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict, Union
from torch import nn
import torch
from ..layers import LR_Layer

class LR(nn.Module):
    def __init__(self,
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, int] = None):
        """
        Logistic Regression (LR) model.

        Args:
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """
        super(LR, self).__init__()

        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.lr_layer = LR_Layer(enc_dict=self.enc_dict)

    def forward(self, data: Dict[str, torch.Tensor],
                is_training: bool = True) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Forward pass of the LR model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): A flag to indicate whether the model is in training mode. Default is True.

        Returns:
            output_dict (Dict[str, Union[torch.Tensor, float]]): The output dictionary containing the predictions and
            optional loss value.
        """
        y_pred = self.lr_layer(data)
        y_pred = y_pred.sigmoid()

        # Output
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict
