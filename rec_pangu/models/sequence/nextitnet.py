# -*- ecoding: utf-8 -*-
# @ModuleName: nextitnet
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/4/3 15:14
from typing import Dict
import torch
from rec_pangu.models.base_model import SequenceBaseModel
from rec_pangu.models.layers import NextItNetLayer


class NextItNet(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(NextItNet, self).__init__(enc_dict, config)

        self.dilations = self.config.get('dilations', None)
        self.one_masked = self.config.get('one_masked', False)
        self.kernel_size = self.config.get('kernel_size', 3)
        self.feat_drop = self.config.get('feat_drop', 0)

        self.nextit_layer = NextItNetLayer(
            self.embedding_dim, self.dilations, self.one_masked, self.kernel_size, feat_drop=self.feat_drop
        )
        self.fc = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

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
        mask = data['hist_mask_list']
        item_seq_len = torch.sum(mask, dim=1)

        item_seq_emb = self.item_emb(item_seq)
        user_emb = self.nextit_layer(item_seq_emb, item_seq_len)
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
