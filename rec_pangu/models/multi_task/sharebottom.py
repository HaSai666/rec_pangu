# -*- ecoding: utf-8 -*-
# @ModuleName: sharebottom
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict, List
import torch
from torch import nn
from ..utils import get_feature_num, get_linear_input
import numpy as np
from ..base_model import BaseModel
class ShareBottom(BaseModel):
    def __init__(self,
                 num_task: int = 2,
                 embedding_dim: int = 40,
                 hidden_units: List[int] = [128, 64],
                 dropouts: List[float] = [0.2, 0.2],
                 enc_dict: Dict[str, dict] = None):
        super(ShareBottom, self).__init__(enc_dict,embedding_dim)
        """
        ShareBottom model.

        Args:
            num_task (int): The number of tasks to be performed. Default is 2.
            embedding_dim (int): The size of the embedding vector. Default is 40.
            hidden_units (List[int]): The list of hidden units for each layer. Default is [128, 64].
            dropouts (List[float]): The list of dropout rates for each layer. Default is [0.2, 0.2].
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.hidden_dim = hidden_units
        self.dropouts = dropouts

        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)

        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.apply(self._init_weights)

        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [hidden_size] + hidden_units
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the ShareBottom model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        out = torch.cat([feature_emb, dense_fea], axis=-1)

        # task tower
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = out
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        # get loss
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss

        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight is None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i, _ in enumerate(task_outputs):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1),
                                                                   data[f'task{i + 1}_label'])

        return loss
