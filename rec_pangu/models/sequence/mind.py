# -*- ecoding: utf-8 -*-
# @ModuleName: mind
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/5 15:08
from typing import Dict
import torch
from rec_pangu.models.layers import CapsuleNetwork
from rec_pangu.models.base_model import SequenceBaseModel


class MIND(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(MIND, self).__init__(enc_dict, config)

        self.capsule = CapsuleNetwork(self.embedding_dim, self.max_length, bilinear_type=0,
                                      interest_num=self.config['K'])
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

        if is_training:
            item = data['target_item'].squeeze()
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            item_e = self.item_emb(item).squeeze(1)

            multi_interest_emb = self.capsule(seq_emb, mask, self.device)  # Batch,K,Emb

            cos_res = torch.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            k_index = torch.argmax(cos_res, dim=1)

            best_interest_emb = torch.rand(multi_interest_emb.shape[0], multi_interest_emb.shape[2]).to(self.device)
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]

            loss = self.calculate_loss(best_interest_emb, item)
            output_dict = {
                'user_emb': multi_interest_emb,
                'loss': loss,
            }
        else:
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            multi_interest_emb = self.capsule(seq_emb, mask, self.device)  # Batch,K,Emb
            output_dict = {
                'user_emb': multi_interest_emb,
            }
        return output_dict
