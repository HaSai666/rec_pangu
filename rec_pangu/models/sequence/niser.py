# -*- ecoding: utf-8 -*-
# @ModuleName: niser
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/28 15:39
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from rec_pangu.models.utils import generate_graph
from rec_pangu.models.layers import SRGNNCell
from rec_pangu.models.base_model import SequenceBaseModel


class NISER(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(NISER, self).__init__(enc_dict, config)
        # define layers and loss

        self.step = self.config.get('step', 1)

        self.pos_embedding = nn.Embedding(self.max_length, self.embedding_dim)
        self.item_dropout = nn.Dropout(config.get('item_dropout', 0.1))

        self.gnncell = SRGNNCell(self.embedding_dim)
        self.linear_one = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_two = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_three = nn.Linear(self.embedding_dim, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        # self.apply(self._init_weights)
        self.reset_parameters()

    def forward(self, data: Dict[str, torch.tensor], is_training: bool = True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """

        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)
        # 生产Batch Session-Graph
        batch_data = generate_graph(data)

        # 获取Graph中的Embedding
        hidden = self.item_emb(batch_data['x'])
        # Dropout in NISER+
        hidden = self.item_dropout(hidden)
        # Normalize item embeddings
        hidden = F.normalize(hidden, dim=-1)
        # 调用GNN
        for i in range(self.step):
            hidden = self.gnncell(batch_data['in_graph'], batch_data['out_graph'], hidden)

        # 通过hidden取出序列里面item通过GNN的emb
        seq_hidden = hidden[batch_data['alias_inputs']]  # [batch,seq_num,emb]
        pos_emb = self.pos_embedding.weight[:seq_hidden.shape[1]]
        pos_emb = pos_emb.unsqueeze(0).expand(item_seq_len.shape[0], -1, -1)
        seq_hidden = seq_hidden + pos_emb
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)  # vn
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # 获得attention score
        '''
        在使用attention score进行加权求和的时候，我们对无效位置需要进行mask，即直接乘以mask即可
        '''
        a = torch.sum(alpha * seq_hidden * data['hist_mask_list'].view(seq_hidden.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        # Normalize session embeddings
        seq_output = F.normalize(seq_output, dim=-1)

        output_dict = dict()
        if is_training:
            loss = self.calculate_loss(seq_output, data['target_item'].squeeze())
            output_dict['loss'] = loss
        else:
            output_dict['user_emb'] = seq_output

        return output_dict