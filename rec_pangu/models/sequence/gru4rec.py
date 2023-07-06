# -*- ecoding: utf-8 -*-
# @ModuleName: yotubednn
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/5 15:25
from typing import Dict
import torch
from rec_pangu.models.base_model import SequenceBaseModel
from rec_pangu.models.layers import GRU4RecEncoder


class GRU4Rec(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(GRU4Rec, self).__init__(enc_dict, config)
        self.gru = GRU4RecEncoder(self.embedding_dim, self.embedding_dim)
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
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        item_seq_length = torch.sum(mask, dim=1)

        seq_emb = self.item_emb(item_seq)
        user_emb = self.gru(seq_emb, item_seq_length)
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            output_dict = {
                'user_emb': user_emb,
                'loss': loss
            }
        else:
            output_dict = {
                'user_emb': user_emb
            }
        return output_dict
