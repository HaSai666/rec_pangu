# -*- ecoding: utf-8 -*-
# @ModuleName: yotubednn
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/5 15:25
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..base_model import SequenceBaseModel

class YotubeDNN(SequenceBaseModel):
    def __init__(self, enc_dict,config):
        super(YotubeDNN, self).__init__(enc_dict,config)

        self.apply(self._init_weights)

    def forward(self,data, is_training=True):
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']

        user_emb = self.item_emb(item_seq)
        mask = mask.unsqueeze(-1).float()
        user_emb = torch.mean(user_emb*mask,dim=1)
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            output_dict = {
                'user_emb':user_emb,
                'loss':loss
            }
        else:
            output_dict = {
                'user_emb':user_emb
            }
        return output_dict

