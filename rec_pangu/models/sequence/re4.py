# -*- ecoding: utf-8 -*-
# @ModuleName: re4
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/14 14:57
# <Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation> WWW 2022
from typing import Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rec_pangu.models.base_model import SequenceBaseModel


class Re4(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(Re4, self).__init__(enc_dict, config)

        self.num_interests = self.config.get('K', 4)
        self.att_thre = self.config.get('att_thre', -1)
        self.t_cont = self.config.get('t_cont', 0.02)
        self.att_lambda = self.config.get('att_lambda', 0.01)
        self.ct_lambda = self.config.get('ct_lambda', 0.1)
        self.cs_lambda = self.config.get('cs_lambda', 0.1)

        self.proposal_num = self.num_interests
        self.W1 = torch.nn.Parameter(data=torch.randn(256, self.embedding_dim), requires_grad=True)
        self.W1_2 = torch.nn.Parameter(data=torch.randn(self.proposal_num, 256), requires_grad=True)
        self.W2 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.W3 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.W3_2 = torch.nn.Parameter(data=torch.randn(self.max_length, self.embedding_dim), requires_grad=True)
        self.W5 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)

        self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc_cons = nn.Linear(self.embedding_dim, self.embedding_dim * self.max_length)

        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        self.recons_mse_loss = nn.MSELoss(reduce=False)

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
        item_mask = data['hist_mask_list']
        dim0, dim1 = item_seq.shape

        item_seq_len = torch.sum(item_mask, dim=-1)
        item_seq = torch.reshape(item_seq, (1, dim0 * dim1))
        item_seq_emb = self.item_emb(item_seq)
        item_seq_emb = torch.reshape(item_seq_emb, (dim0, dim1, -1))

        proposals_weight = torch.matmul(self.W1_2,
                                        torch.tanh(torch.matmul(self.W1, torch.transpose(item_seq_emb, 1, 2))))
        proposals_weight_logits = proposals_weight.masked_fill(item_mask.unsqueeze(1).bool(), -1e9)
        proposals_weight = torch.softmax(proposals_weight_logits, dim=2)
        user_interests = torch.matmul(proposals_weight, torch.matmul(item_seq_emb, self.W2))

        if is_training:
            target_item = data['target_item']
            item_e = self.item_emb(target_item)
            # main loss
            cos_res = torch.bmm(user_interests, item_e.squeeze(1).unsqueeze(-1))
            k_index = torch.argmax(cos_res, dim=1)

            best_interest_emb = torch.rand(user_interests.shape[0], user_interests.shape[2]).to(self.device)
            for k in range(user_interests.shape[0]):
                best_interest_emb[k, :] = user_interests[k, k_index[k], :]

            loss = self.calculate_loss(best_interest_emb, target_item.squeeze())

            # re-attend
            product = torch.matmul(user_interests, torch.transpose(item_seq_emb, 1, 2))
            product = product.masked_fill(item_mask.unsqueeze(1).bool(), -1e9)
            re_att = torch.softmax(product, dim=2)
            att_pred = F.log_softmax(proposals_weight_logits, dim=-1)
            loss_attend = -(re_att * att_pred).sum() / (re_att).sum()

            # re-contrast
            norm_watch_interests = F.normalize(user_interests, p=2, dim=-1)
            norm_watch_movie_embedding = F.normalize(item_seq_emb, p=2, dim=-1)
            cos_sim = torch.matmul(norm_watch_interests, torch.transpose(norm_watch_movie_embedding, 1, 2))
            if self.att_thre == -1:
                gate = np.repeat(1 / (item_seq_len.cpu() * 1.0), self.max_length, axis=0)
            else:
                gate = np.repeat(torch.FloatTensor([self.att_thre]).repeat(item_seq_len.size(0)), self.max_length,
                                 axis=0)
            gate = torch.reshape(gate, (dim0, 1, self.max_length)).to(self.device)
            positive_weight_idx = (proposals_weight > gate) * 1  # value is 1 or 0
            mask_cos = cos_sim.masked_fill(item_mask.unsqueeze(1).bool(), -1e9)
            pos_cos = mask_cos.masked_fill(positive_weight_idx != 1, -1e9)
            import pdb
            # cons_pos = torch.sum(torch.exp(pos_cos / t_cont), dim=2)
            cons_pos = torch.exp(pos_cos / self.t_cont)
            cons_neg = torch.sum(torch.exp(mask_cos / self.t_cont), dim=2)

            in2in = torch.matmul(norm_watch_interests, torch.transpose(norm_watch_interests, 1, 2))
            in2in = in2in.masked_fill(torch.eye(self.proposal_num).to(in2in.device).unsqueeze(0) == 1, -1e9)
            cons_neg = cons_neg + torch.sum(torch.exp(in2in / self.t_cont), dim=2)

            item_rolled = torch.roll(norm_watch_movie_embedding, 1, 0)
            in2i = torch.matmul(norm_watch_interests, torch.transpose(item_rolled, 1, 2))
            in2i_mask = torch.roll((item_seq == 0).reshape(dim0, dim1), 1, 0)
            in2i = in2i.masked_fill(in2i_mask.unsqueeze(1), -1e9)
            cons_neg = cons_neg + torch.sum(torch.exp(in2i / self.t_cont), dim=2)

            cons_div = cons_pos / cons_neg.unsqueeze(-1)
            cons_div = cons_div.masked_fill(item_mask.unsqueeze(1).bool(), 1)
            cons_div = cons_div.masked_fill(positive_weight_idx != 1, 1)
            # loss_contrastive = -torch.log(cons_pos / cons_neg.unsqueeze(-1))
            loss_contrastive = -torch.log(cons_div)
            loss_contrastive = torch.mean(loss_contrastive)

            # re-construct
            recons_item = self.fc_cons(user_interests)
            recons_item = recons_item.reshape([dim0 * self.proposal_num, dim1, -1])
            recons_weight = torch.matmul(self.W3_2,
                                         torch.tanh(torch.matmul(self.W3, torch.transpose(recons_item, 1, 2))))
            recons_weight = recons_weight.reshape([dim0, self.proposal_num, dim1, dim1])
            recons_weight = recons_weight.masked_fill((item_seq == 0).reshape(dim0, 1, 1, dim1), -1e9).reshape(
                [-1, dim1, dim1])
            recons_weight = torch.softmax(recons_weight, dim=-1)
            recons_item = torch.matmul(recons_weight, torch.matmul(recons_item, self.W5)).reshape(
                [dim0, self.proposal_num, dim1, -1])
            target_emb = item_seq_emb.unsqueeze(1).repeat(1, self.proposal_num, 1, 1)
            loss_construct = self.recons_mse_loss(recons_item, target_emb)
            loss_construct = loss_construct.masked_fill((positive_weight_idx == 0).unsqueeze(-1), 0.)
            loss_construct = loss_construct.masked_fill(item_mask.unsqueeze(-1).unsqueeze(1).bool(), 0.)
            loss_construct = torch.mean(loss_construct)

            loss = loss + self.att_lambda * loss_attend + self.ct_lambda * loss_contrastive + self.cs_lambda * loss_construct
            output_dict = {
                'user_emb': user_interests,
                'loss': loss
            }
        else:
            output_dict = {
                'user_emb': user_interests
            }

        return output_dict
