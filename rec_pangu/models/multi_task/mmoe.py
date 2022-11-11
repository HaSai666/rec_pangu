# -*- ecoding: utf-8 -*-
# @ModuleName: mmoe
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from torch import nn
from ..layers import EmbeddingLayer
from ..utils import get_feature_num, get_linear_input
import numpy as np
from ..base_model import BaseModel

class MMOE(BaseModel):
    def __init__(self,
                 num_task=2,
                 n_expert=3,
                 embedding_dim=40,
                 mmoe_hidden_dim=128,
                 expert_activation=None,
                 hidden_dim=[128, 64],
                 dropouts=[0.2, 0.2],
                 enc_dict=None,
                 device=None):
        super(MMOE, self).__init__(enc_dict,embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.mmoe_hidden_dim = mmoe_hidden_dim
        self.expert_activation = expert_activation
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts

        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)

        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea

        # experts
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        # self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        # gates
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in
                      range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]
        # esmm ctr和ctcvr独立任务的DNN结构
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())
        self.set_device(device)
        self.apply(self._init_weights)

    def set_device(self, device):
        for i in range(self.num_task):
            self.gates[i] = self.gates[i].to(device)
            self.gates_bias[i] = self.gates_bias[i].to(device)
        print(f'Successfully set device:{device}')

    def forward(self, data,is_training=True):
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        hidden = torch.cat([hidden, dense_fea], axis=-1)

        # mmoe
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)  # batch * mmoe_hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum('ab, bc -> ac', hidden, gate)  # batch * num_experts
            if self.gates_bias:
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)

        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)  # batch * 1 * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(
                experts_out)  # batch * mmoe_hidden_size * num_experts
            outs.append(torch.sum(weighted_expert_output, 2))  # batch * mmoe_hidden_size

        # task tower
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
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
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1)+1e-6,
                                                                   data[f'task{i + 1}_label'])

        return loss