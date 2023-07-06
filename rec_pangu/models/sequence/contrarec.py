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
from rec_pangu.models.layers import BERT4RecEncoder, GRU4RecEncoder, CaserEncoder


class ContraRec(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(ContraRec, self).__init__(enc_dict, config)

        self.gamma = self.config.get("gamma", 1)
        self.beta_a = self.config.get("beta_a", 3)
        self.beta_b = self.config.get("beta_b", 3)
        self.ctc_temp = self.config.get("ctc_temp", 0.2)
        self.ccc_temp = self.config.get("ccc_temp", 0.2)
        self.encoder_name = self.config.get("encoder_name", "BERT4Rec")
        self.encoder = self.init_encoder(self.encoder_name)
        self.data_augmenter = DataAugmenter(beta_a=self.beta_a,
                                            beta_b=self.beta_b,
                                            num_items=self.enc_dict[self.config['item_col']]['vocab_size']-1)

        self.ccc_loss = ContraLoss(self.device, temperature=self.ccc_temp)

        self.reset_parameters()

    def init_encoder(self, encoder_name):
        if encoder_name == 'GRU4Rec':
            encoder = GRU4RecEncoder(self.embedding_dim, hidden_size=128)
        elif encoder_name == 'Caser':
            encoder = CaserEncoder(self.embedding_dim, self.max_length, num_horizon=16, num_vertical=8, l=5)
        elif encoder_name == 'BERT4Rec':
            encoder = BERT4RecEncoder(self.embedding_dim, self.max_length, num_layers=2, num_heads=2)
        else:
            raise ValueError('Invalid sequence encoder.')
        return encoder

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
            aug_seq1 = self.data_augmenter.augment(item_seq)
            aug_seq2 = self.data_augmenter.augment(item_seq)

            aug_seq_emb1 = self.item_emb(aug_seq1)
            aug_seq_emb2 = self.item_emb(aug_seq2)

            aug_user_emb1 = self.encoder(aug_seq_emb1, item_seq_length)
            aug_user_emb2 = self.encoder(aug_seq_emb2, item_seq_length)

            features = torch.stack([aug_user_emb1, aug_user_emb2], dim=1)
            features = F.normalize(features, dim=-1)

            loss = self.calculate_loss(user_emb, item) + self.gamma * self.ccc_loss(features, item)
            output_dict = {
                'loss': loss
            }
        else:
            output_dict = {
                'user_emb': user_emb
            }
        return output_dict


""" Context-Context Contrastive Loss """


class ContraLoss(nn.Module):
    def __init__(self, device, temperature=0.2):
        super(ContraLoss, self).__init__()
        self.device = device
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        If both `labels` and `mask` are None, it degenerates to InfoNCE loss
        Args:
            features: hidden vector of shape [bsz, n_views, dim].
            labels: target item of shape [bsz].
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.transpose(0, 1)).float().to(self.device)

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # bsz * n_views, -1

        # compute logits
        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.transpose(0, 1)) / self.temperature
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # bsz * n_views, bsz * n_views

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)  # bsz * n_views, bsz * n_views
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(mask.shape[0]).view(-1, 1).to(self.device), 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # loss
        loss = - self.temperature * mean_log_prob_pos
        return loss.mean()


class DataAugmenter:

    def __init__(self, beta_a, beta_b, num_items):
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.num_items = num_items

    def reorder_op(self, seq):
        ratio = torch.distributions.beta.Beta(self.beta_a, self.beta_b).sample().item()
        select_len = int(len(seq) * ratio)
        start = torch.randint(0, len(seq) - select_len + 1, (1,)).item()
        idx_range = torch.arange(len(seq))
        idx_range[start: start + select_len] = idx_range[start: start + select_len][torch.randperm(select_len)]
        return seq[idx_range]

    def mask_op(self, seq):
        ratio = torch.distributions.beta.Beta(self.beta_a, self.beta_b).sample().item()
        selected_len = int(len(seq) * ratio)
        mask = torch.full((len(seq),), False, dtype=torch.bool)
        mask[:selected_len] = True
        mask = mask[torch.randperm(len(mask))]
        seq[mask] = self.num_items
        return seq

    def augment(self, seqs):
        seqs = seqs.clone()
        for i, seq in enumerate(seqs):
            if torch.rand(1) > 0.5:
                seqs[i] = self.mask_op(seq.clone())
            else:
                seqs[i] = self.reorder_op(seq.clone())
        return seqs
