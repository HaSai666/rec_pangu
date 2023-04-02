# -*- ecoding: utf-8 -*-
# @ModuleName: masknet
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/4/2 15:19
from typing import Dict, List
import torch
from ..layers import MaskBlock, MLP
from ..utils import get_dnn_input_dim, get_linear_input
from ..base_model import BaseModel


class MaskNet(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 block_num: int = 3,
                 use_parallel: bool = True,
                 reduction_factor: float = 0.3,
                 hidden_units: List[int] = [64, 64, 64],
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None):
        super(MaskNet, self).__init__(enc_dict, embedding_dim)
        """A class for the MaskNet

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            block_num (int): The number of MaskBlocks to use. Default is 3.
            use_parallel (bool): If True, use parallel processing for the MaskBlocks. Default is True.
            reduction_factor (float): The reduction factor used to scale the output size of the MaskBlocks. Default is 0.3.
            hidden_units (List[int]): A list of integers representing the number of hidden units in each layer of the MLP. Default is [64, 64, 64].
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """

        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.block_num = block_num
        self.hidden_units = hidden_units
        self.reduction_factor = reduction_factor
        self.use_parallel = use_parallel
        self.block_output_dim = self.mask_input_dim = self.input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)

        self.mask_block_list = torch.nn.ModuleList()
        for _ in range(self.block_num):
            self.mask_block_list.append(MaskBlock(self.input_dim, self.mask_input_dim,
                                                  self.block_output_dim, self.reduction_factor))

        self.mlp = MLP(self.block_output_dim,
                       hidden_units=self.hidden_units,
                       output_dim=1)

        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the MaskNet model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        sparse_embedding = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        emb_flatten = sparse_embedding.flatten(start_dim=1)
        dnn_input = torch.cat((emb_flatten, dense_input), dim=1)
        if self.use_parallel:
            mask_outputs = []
            for layer in self.mask_block_list:
                mask_outputs.append(layer(dnn_input, dnn_input))
            mask_outputs = torch.stack(mask_outputs, dim=1)
            mask_output = torch.mean(mask_outputs, dim=1)
        else:
            mask_output = dnn_input
            for layer in self.mask_block_list:
                mask_output = layer(mask_output, dnn_input)

        y_pred = self.mlp(mask_output).sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict
