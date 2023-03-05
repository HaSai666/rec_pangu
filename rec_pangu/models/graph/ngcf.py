# -*- ecoding: utf-8 -*-
# @ModuleName: ngcf.py
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/12/19 20:22
from torch import nn
import torch
from ..layers import NGCFLayer
from ..utils import get_feature_num, get_linear_input
from ..base_model import GraphBaseModel

class NGCF(GraphBaseModel):
    def __init__(self, g, num_user, num_item, embedding_dim, hidden_size, dropout=0.1, lmbd=1e-5):
        super(NGCF, self).__init__(num_user, num_item, embedding_dim)
        self.g = g
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lmbd = lmbd

        self.ngcf_layers = nn.ModuleList()
        self.hidden_size = [self.embedding_dim] + self.hidden_size

        for i in range(len(self.hidden_size) - 1):
            self.ngcf_layers.append(NGCFLayer(self.hidden_size[i], self.hidden_size[i + 1], self.dropout))

        self.apply(self._init_weights)

    def forward(self, data, is_training=True):

        ego_embedding = self.get_ego_embedding()
        user_embeds = []
        item_embeds = []
        user_embeds.append(self.user_emb_layer.weight)
        item_embeds.append(self.item_emb_layer.weight)

        for ngcf_layer in self.ngcf_layers:
            ego_embedding = ngcf_layer(self.g, ego_embedding)
            temp_user_emb, temp_item_emb = torch.split(ego_embedding, [self.num_user, self.num_item])
            user_embeds.append(temp_user_emb)
            item_embeds.append(temp_item_emb)

        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)

        output_dict = dict()
        if is_training:
            u_g_embeddings = user_embd[data['user_id'], :]
            pos_i_g_embeddings = item_embd[data['pos_item_id'], :]
            neg_i_g_embeddings = item_embd[data['neg_item_id'], :]
            loss = self.create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
            output_dict['loss'] = loss
        else:
            output_dict['user_emb'] = user_embd
            output_dict['item_emb'] = item_embd

        return output_dict