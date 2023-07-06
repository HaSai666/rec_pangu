# -*- ecoding: utf-8 -*-
# @ModuleName: yotubednn
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/5 15:25
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F

from rec_pangu.models.base_model import SequenceBaseModel
from rec_pangu.models.layers import BERT4RecEncoder


class CLRec(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(CLRec, self).__init__(enc_dict, config)

        self.temp = self.config.get("temp", 0.1)
        self.encoder = BERT4RecEncoder(self.embedding_dim, self.max_length, num_layers=2, num_heads=2)
        self.contra_loss = ContraLoss(temperature=self.temp)

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
        user_emb = self.encoder(seq_emb, item_seq_length)

        if is_training:
            item = data['target_item'].squeeze()
            target_item_emb = self.item_emb(item)
            features = torch.stack([user_emb, target_item_emb], dim=1)  # bsz, 2, emb
            features = F.normalize(features, dim=-1)
            loss = self.calculate_loss(user_emb, item)
            loss += self.contra_loss(features)
            output_dict = {
                'loss': loss
            }
        else:
            output_dict = {
                'user_emb': user_emb
            }
        return output_dict


""" Contrastive Loss """


class ContraLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(ContraLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sequence j
                has the same target item as sequence i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size, device = features.shape[0], features.device
        if mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        # compute logits
        dot_contrast = torch.matmul(features[:, 0], features[:, 1].transpose(0, 1)) / self.temperature
        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()  # bsz, bsz

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)

        return -mean_log_prob_pos.mean()
