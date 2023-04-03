# -*- ecoding: utf-8 -*-
# @ModuleName: gcsan
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/27 19:30
from typing import Dict
import torch
from torch import nn
from rec_pangu.models.utils import generate_graph
from rec_pangu.models.layers import SRGNNCell, TransformerEncoder
from rec_pangu.models.base_model import SequenceBaseModel


class GCSAN(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(GCSAN, self).__init__(enc_dict, config)
        # define layers and loss

        self.n_layers = config.get('n_layers', 2)
        self.n_heads = config.get('n_heads', 4)
        self.hidden_size = config.get('hidden_size', 64)  # same as embedding_size
        self.inner_size = config.get('inner_size', 32)  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.get('hidden_dropout_prob', 0.1)
        self.attn_dropout_prob = config.get('attn_dropout_prob', 0.1)
        self.hidden_act = config.get('hidden_act', 'gelu')
        self.layer_norm_eps = config.get('layer_norm_eps', 0.001)
        self.step = config.get('step', 1)
        self.weight = config.get('weight', 0.1)

        self.gnncell = SRGNNCell(self.embedding_dim)
        self.linear_one = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_two = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_three = nn.Linear(self.embedding_dim, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        self.self_attention = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.apply(self._init_weights)

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
        # 调用GNN
        for i in range(self.step):
            hidden = self.gnncell(batch_data['in_graph'], batch_data['out_graph'], hidden)

        # 通过hidden取出序列里面item通过GNN的emb
        seq_hidden = hidden[batch_data['alias_inputs']]  # [batch,seq_num,emb]
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        attention_mask = self.get_attention_mask(data['hist_mask_list'])
        outputs = self.self_attention(seq_hidden, attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        at = self.gather_indexes(output, item_seq_len - 1)
        seq_output = self.weight * at + (1 - self.weight) * ht

        output_dict = dict()
        if is_training:
            loss = self.calculate_loss(seq_output, data['target_item'].squeeze())
            output_dict['loss'] = loss
        else:
            output_dict['user_emb'] = seq_output

        return output_dict
