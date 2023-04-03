# -*- ecoding: utf-8 -*-
# @ModuleName: stamp
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/4/3 16:19
from typing import Dict
import torch
from rec_pangu.models.base_model import SequenceBaseModel
from rec_pangu.models.layers import STAMPLayer


class STAMP(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(STAMP, self).__init__(enc_dict, config)

        self.feat_drop = self.config.get('feat_drop', 0)

        self.stamp_layer = STAMPLayer(self.embedding_dim, feat_drop=self.feat_drop)
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
        user_emb = self.stamp_layer(item_seq_emb, item_seq_len)
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
