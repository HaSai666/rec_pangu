# -*- ecoding: utf-8 -*-
# @ModuleName: cmi
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/10 15:28
# <Improving Micro-video Recommendation via Contrastive Multiple Interests> SIGIR 2022
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from rec_pangu.models.base_model import SequenceBaseModel


class CMI(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(CMI, self).__init__(enc_dict, config)

        self.hidden_size = self.config.get('hidden_size', 64)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout_prob = self.config.get('dropout_prob', 0)
        self.temp = self.config.get('temp', 0.1)
        self.w_uniform = self.config.get('w_uniform', 1)  # 约束在全局交互数据在各个兴趣向量上分布均匀
        self.w_orth = self.config.get('w_orth', 10)  # 约束全局兴趣向量比较正交
        self.w_sharp = self.config.get('w_sharp', 1)  # 约束item属于一个全局兴趣向量
        self.w_clloss = self.config.get('w_clloss', 0.05)
        self.n_interest = self.config.get('K', 8)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.W = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.selfatt_W = nn.Linear(self.n_interest, self.n_interest, bias=False)
        self.interest_embedding = nn.Embedding(self.n_interest, self.embedding_dim)
        self.temperature = 0.1

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
            nn.ReLU()
        )

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
        with torch.no_grad():
            w = self.item_emb.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.item_emb.weight.copy_(w)

            w = self.interest_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.interest_embedding.weight.copy_(w)

        item_seq = data['hist_item_list']

        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)

        batch_size, n_seq = item_seq.shape
        item_seq_emb = self.item_emb(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)

        psnl_interest = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1,
                                                                           1)  # bs * n_interest * embed_size
        interest_cl = self.w_orth * self.get_orth_loss(self.interest_embedding.weight)

        for i in range(1):  # 迭代次数可以变成超参数
            scores = item_seq_emb.matmul(psnl_interest.transpose(1, 2)) / self.temp
            scores = scores.reshape(batch_size * n_seq, -1)
            mask = (item_seq > 0).reshape(-1)

            probs = torch.softmax(scores.reshape(batch_size, n_seq, -1), dim=-1) * (item_seq > 0).float().unsqueeze(-1)

            if self.w_uniform:
                interest_prb_vec = torch.sum(probs.reshape(batch_size * n_seq, -1), dim=0) / torch.sum(
                    mask)  # n_interest 1-dim vector
                # print(probs.shape, interest_prb_vec.shape)
                interest_cl += self.w_uniform * interest_prb_vec.std() / interest_prb_vec.mean()
                # todo: 求和均匀向量的交叉熵
            psnl_interest = probs.transpose(1, 2).matmul(item_seq_emb)
            psnl_interest = F.normalize(psnl_interest, dim=-1, p=2)

            sys_interest_vec = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            interest_mask = torch.sum(probs, dim=1)  # batch_size * n_interest
            psnl_interest = torch.where(interest_mask.unsqueeze(-1) > 0, psnl_interest,
                                        sys_interest_vec)  # todo: 这里可以设置一个阈值
            batch_size, seq_len, n_interest = probs.shape

        # add global psnl embedding with GRU，用户对物品的偏好分数 = 某个单独的兴趣对物品的偏好分数 + 全局个性化偏好对物品的偏好分数
        gru_output, _ = self.gru(item_seq_emb)
        gru_output = self.mlp(gru_output)
        full_psnl_emb = self.gather_indexes(gru_output, item_seq_len - 1)
        full_psnl_emb = F.normalize(full_psnl_emb, p=2, dim=-1)

        # 计算用户整体兴趣向量与各个兴趣点之间的相关性 interest importance scores
        # imp_probs = torch.softmax(
        #     full_psnl_emb.unsqueeze(1).matmul(psnl_interest.transpose(1, 2)).squeeze() / self.temp, dim=-1)
        # interest_mask = imp_probs  # 将 interest_mask 用于表示各个兴趣向量的重要程度

        psnl_interest = psnl_interest + full_psnl_emb.unsqueeze(1)
        psnl_interest = F.normalize(psnl_interest, p=2, dim=-1)

        if is_training:
            output_dict = {
                'global_user_emb': full_psnl_emb,
                'user_emb': psnl_interest,
                'loss': self.calculate_cmi_loss(psnl_interest, data['target_item'].squeeze())
            }
        else:
            output_dict = {
                'user_emb': psnl_interest,
            }
        return output_dict

    def get_neg_item(self, batch_size):
        n_item = self.item_emb.weight.shape[0]
        return torch.randint(1, n_item - 1, (batch_size, 1)).squeeze()

    def calculate_cmi_loss(self, psnl_interest, pos_items):
        batch_size, n_interest, embed_size = psnl_interest.shape

        neg_items = self.get_neg_item(batch_size)
        neg_items = neg_items.to(psnl_interest.device)

        pos_items_emb = self.item_emb(pos_items)
        neg_items_emb = self.item_emb(neg_items)
        pos_scores = torch.sum(psnl_interest * pos_items_emb.unsqueeze(1), dim=-1)
        neg_scores = psnl_interest.reshape(-1, embed_size).matmul(neg_items_emb.transpose(0, 1)).reshape(
            batch_size, -1, batch_size)
        scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)

        scores = torch.max(scores, dim=1)[0]
        loss = self.loss_fun(scores / self.temp, torch.zeros(batch_size, device=pos_items.device).long())

        multi_clloss = self.multi_inter_clloss(psnl_interest)
        loss += self.w_clloss * multi_clloss

        return loss

    def multi_inter_clloss(self, user_interests):
        '''
        下标 0 和 1 是同一个用户数据增强 后所学的不同兴趣，2 和 3 是同一个用户，以此类推，同一个用户同一兴趣之间是正样本，不同用户或不同兴趣之间是负样本
        Args:
            user_interests: batch_size * n_interest * embed_size
        Returns: loss
        '''
        device = user_interests.device
        batch_size, n_interest, embed_size = user_interests.shape
        user_interests = user_interests.reshape(batch_size // 2, 2, n_interest, embed_size)
        user_interests_a = user_interests[:, 0].reshape(-1, embed_size)
        user_interests_b = user_interests[:, 1].reshape(-1, embed_size)
        user_interests_a = F.normalize(user_interests_a, p=2, dim=-1)
        user_interests_b = F.normalize(user_interests_b, p=2, dim=-1)
        sim_matrix = user_interests_a.matmul(user_interests_b.transpose(0, 1)) / self.temperature
        loss = F.cross_entropy(sim_matrix, torch.arange(sim_matrix.shape[0], device=device)) + F.cross_entropy(
            sim_matrix.transpose(0, 1), torch.arange(sim_matrix.shape[0], device=device))
        return loss

    def get_orth_loss(self, x):
        '''
        Args:
            x: batch_size * embed_size; Orthogonal embeddings
        Returns:
        '''
        num, embed_size = x.shape
        sim = x.reshape(-1, embed_size).matmul(x.reshape(-1, embed_size).transpose(0, 1))
        try:
            diff = sim - torch.eye(sim.shape[1]).to(x.device)
        except RuntimeError:
            print('hello')
        regloss = diff.pow(2).sum() / (num * num)
        return regloss

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
