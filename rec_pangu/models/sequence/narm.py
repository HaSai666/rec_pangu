# -*- ecoding: utf-8 -*-
# @ModuleName: narm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/14 16:10
from typing import Dict
import torch
from torch import nn
from rec_pangu.models.base_model import SequenceBaseModel


class NARM(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(NARM, self).__init__(enc_dict, config)

        self.n_layers = self.config.get('n_layers', 2)
        self.dropout_probs = self.config.get('dropout_probs', [0.1, 0.1])
        self.hidden_size = self.config.get('hidden_size', 32)

        self.emb_dropout = nn.Dropout(self.dropout_probs[0])
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers, bias=False, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs[1])
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_dim, bias=False)

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
        item_seq = data['hist_item_list']
        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)

        item_seq_emb = self.item_emb(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        user_emb = self.b(c_t)

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
