# -*- ecoding: utf-8 -*-
# @ModuleName: omoe
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from torch import nn
from ..layers import EmbeddingLayer
from ..utils import get_feature_num, get_linear_input
import numpy as np
from ..base_model import BaseModel

class OMOE(BaseModel):
    def __init__(self,
                 num_task=2,
                 n_expert=3,
                 embedding_dim=40,
                 omoe_hidden_dim=128,
                 expert_activation=None,
                 hidden_dim=[128, 64],
                 dropouts=[0.2, 0.2],
                 enc_dict=None,
                 device=None):
        super(OMOE, self).__init__(enc_dict,embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.omoe_hidden_dim = omoe_hidden_dim
        self.expert_activation = expert_activation
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts

        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)

        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea

        # experts
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, omoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(omoe_hidden_dim, n_expert), requires_grad=True)
        # gate
        self.gate = torch.nn.Parameter(torch.rand(n_expert, 1), requires_grad=True)

        # esmm ctr和ctcvr独立任务的DNN结构
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [omoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())

        self.apply(self._init_weights)

    def forward(self, data,is_training=True):
        """
        Perform forward propagation on the OMOE model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        hidden = torch.cat([hidden, dense_fea], axis=-1)
        # mmoe
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)  # batch * hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        gate = nn.Softmax(dim=0)(self.gate)
        gate_out = torch.einsum('abc, cd -> abd', experts_out, gate).squeeze(-1)  # batch, hidden

        # task tower
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = gate_out
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
        if weight == None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i in range(len(task_outputs)):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1),
                                                                   data[f'task{i + 1}_label'])

        return loss