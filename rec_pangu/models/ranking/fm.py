# -*- ecoding: utf-8 -*-
# @ModuleName: fm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict
import torch
from ..layers import FM_Layer
from ..base_model import BaseModel


class FM(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None):
        """
        Factorization Machine (FM) model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """
        super(FM, self).__init__(enc_dict, embedding_dim)

        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.fm = FM_Layer()

        self.apply(self._init_weights)

    def forward(self, data: Dict[str, torch.Tensor],
                is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the FM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        y_pred = self.fm(feature_emb)
        y_pred = y_pred.sigmoid()

        # Output
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict
