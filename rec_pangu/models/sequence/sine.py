# -*- ecoding: utf-8 -*-
# @ModuleName: yotubednn
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/5 15:25
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from rec_pangu.models.base_model import SequenceBaseModel


class SINE(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(SINE, self).__init__(enc_dict, config)
        self.layer_norm_eps = self.config.get("layer_norm_eps", 1e-4)

        self.D = self.embedding_dim
        self.L = self.config.get("prototype_size", 500)  # 500 for movie-len dataset
        self.k = self.config.get("interest_size", 4)  # 4 for movie-len dataset
        self.tau = self.config.get("tau_ratio", 0.1)  # 0.1 in paper
        self.reg_loss_ratio = self.config.get("reg_loss_ratio", 0.1)  # 0.1 in paper

        self.initializer_range = 0.01

        self.w1 = self._init_weight((self.D, self.D))
        self.w2 = self._init_weight(self.D)
        self.w3 = self._init_weight((self.D, self.D))
        self.w4 = self._init_weight(self.D)

        self.C = nn.Embedding(self.L, self.D)

        self.w_k_1 = self._init_weight((self.k, self.D, self.D))
        self.w_k_2 = self._init_weight((self.k, self.D))

        self.ln2 = nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)
        self.ln4 = nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)

        # self.apply(self._init_weights)
        self.reset_parameters()

    def _init_weight(self, shape):
        mat = torch.FloatTensor(np.random.normal(0, self.initializer_range, shape))
        return nn.Parameter(mat, requires_grad=True)

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
        x_u = self.item_emb(item_seq).to(self.device)  # [B, N, D]

        # concept activation
        # sort by inner product
        x = torch.matmul(x_u, self.w1)
        x = torch.tanh(x)
        x = torch.matmul(x, self.w2)
        a = F.softmax(x, dim=1)
        z_u = torch.matmul(a.unsqueeze(2).transpose(1, 2), x_u).transpose(1, 2)
        s_u = torch.matmul(self.C.weight, z_u)
        s_u = s_u.squeeze(2)
        idx = s_u.argsort(1)[:, -self.k:]
        s_u_idx = s_u.sort(1)[0][:, -self.k:]
        c_u = self.C(idx)
        sigs = torch.sigmoid(s_u_idx.unsqueeze(2).repeat(1, 1, self.embedding_dim))
        C_u = c_u.mul(sigs)

        # intention assignment
        # use matrix multiplication instead of cos()
        w3_x_u_norm = F.normalize(x_u.matmul(self.w3), p=2, dim=2)
        C_u_norm = self.ln2(C_u)
        P_k_t = torch.bmm(w3_x_u_norm, C_u_norm.transpose(1, 2))
        P_k_t_b = F.softmax(P_k_t, dim=2)
        P_k_t_b_t = P_k_t_b.transpose(1, 2)

        # attention weighting
        a_k = x_u.unsqueeze(1).repeat(1, self.k, 1, 1).matmul(self.w_k_1)
        P_t_k = F.softmax(
            torch.tanh(a_k)
            .matmul(self.w_k_2.reshape(self.k, self.embedding_dim, 1))
            .squeeze(3),
            dim=2,
        )

        # interest embedding generation
        mul_p = P_k_t_b_t.mul(P_t_k)
        x_u_re = x_u.unsqueeze(1).repeat(1, self.k, 1, 1)
        mul_p_re = mul_p.unsqueeze(3)
        delta_k = x_u_re.mul(mul_p_re).sum(2)
        delta_k = F.normalize(delta_k, p=2, dim=2)

        # prototype sequence
        x_u_bar = P_k_t_b.matmul(C_u)
        C_apt = F.softmax(torch.tanh(x_u_bar.matmul(self.w3)).matmul(self.w4), dim=1)
        C_apt = C_apt.reshape(-1, 1, self.max_length).matmul(x_u_bar)
        C_apt = self.ln4(C_apt)

        # aggregation weight
        e_k = delta_k.bmm(C_apt.reshape(-1, self.embedding_dim, 1)) / self.tau
        e_k_u = F.softmax(e_k.squeeze(2), dim=1)
        user_emb = e_k_u.unsqueeze(2).mul(delta_k).sum(dim=1)

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
