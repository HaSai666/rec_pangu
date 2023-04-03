# -*- ecoding: utf-8 -*-
# @ModuleName: sasrec
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/14 16:24
from typing import Dict
import torch
from torch import nn
from rec_pangu.models.layers import TransformerEncoder
from rec_pangu.models.base_model import SequenceBaseModel


class SASRec(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(SASRec, self).__init__(enc_dict, config)

        self.n_layers = config.get('n_layers', 2)
        self.n_heads = config.get('n_heads', 4)
        self.hidden_size = config.get('hidden_size', 64)  # same as embedding_size
        self.inner_size = config.get('inner_size', 32)  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.get('hidden_dropout_prob', 0.1)
        self.attn_dropout_prob = config.get('attn_dropout_prob', 0.1)
        self.hidden_act = config.get('hidden_act', 'gelu')
        self.layer_norm_eps = config.get('layer_norm_eps', 0.001)

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
        item_seq_emb = self.item_emb(data['hist_item_list'])
        attention_mask = self.get_attention_mask(data['hist_mask_list'])
        outputs = self.self_attention(item_seq_emb, attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        user_emb = self.gather_indexes(output, item_seq_len - 1)

        if is_training:
            target_item = data['target_item'].squeeze()
            output_dict = {
                'user_emb': user_emb,
                'loss': self.calculate_loss(user_emb, target_item)
            }
        else:
            output_dict = {
                'user_emb': user_emb,
            }
        return output_dict
