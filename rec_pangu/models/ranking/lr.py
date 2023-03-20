# -*- ecoding: utf-8 -*-
# @ModuleName: lr
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict
from torch import nn
import torch
from ..layers import LR_Layer


class LR(nn.Module):
    def __init__(self,
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict]= None):
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
                is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the LR model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
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
