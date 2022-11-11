# -*- ecoding: utf-8 -*-
# @ModuleName: aitm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from torch import nn
from ..layers import EmbeddingLayer,MLP_Layer,MultiHeadSelfAttention
from ..utils import get_feature_num
from ..base_model import BaseModel
class AITM(BaseModel):
    def __init__(self,
                 embedding_dim=32,
                 tower_dims=[400, 400, 400],
                 drop_prob=[0.1, 0.1, 0.1],
                 enc_dict=None,
                 device=None):
        super(AITM, self).__init__(enc_dict,embedding_dim)
        self.enc_dict = enc_dict
        self.tower_dims = tower_dims
        self.drop_prob = drop_prob

        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)

        self.tower_input_size = self.num_sparse_fea * self.embedding_dim

        self.click_tower = MLP_Layer(input_dim=self.tower_input_size, hidden_units=self.tower_dims,
                                     hidden_activations='relu', dropout_rates=self.drop_prob)
        self.conversion_tower = MLP_Layer(input_dim=self.tower_input_size, hidden_units=self.tower_dims,
                                          hidden_activations='relu', dropout_rates=self.drop_prob)
        self.attention_layer = MultiHeadSelfAttention(self.tower_dims[-1])

        self.info_layer = nn.Sequential(nn.Linear(tower_dims[-1], tower_dims[-1]), nn.ReLU(),
                                        nn.Dropout(drop_prob[-1]))

        self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                         nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                              nn.Sigmoid())
        self.apply(self._init_weights)

    def forward(self, data,is_training=True):
        feature_embedding = self.embedding_layer(data)
        feature_embedding = feature_embedding.flatten(start_dim=1)

        tower_click = self.click_tower(feature_embedding)

        tower_conversion = torch.unsqueeze(
            self.conversion_tower(feature_embedding), 1)

        info = torch.unsqueeze(self.info_layer(tower_click), 1)
        ait = self.attention_layer(torch.cat([tower_conversion, info], 1))

        ait = torch.sum(ait, dim=1)
        click = torch.squeeze(self.click_layer(tower_click), dim=1)
        conversion = torch.squeeze(self.conversion_layer(ait), dim=1)

        if is_training:
            loss = self.loss(data['task1_label'], click, data['task2_label'], conversion)
            output_dict = {
                'task1_pred': click,
                'task2_pred': conversion,
                'loss': loss
            }
        else:
            output_dict = {
                'task1_pred': click,
                'task2_pred': conversion,
            }
        return output_dict

    def loss(self,
             click_label,
             click_pred,
             conversion_label,
             conversion_pred,
             constraint_weight=0.6):
        click_label = click_label
        conversion_label = conversion_label

        click_loss = nn.functional.binary_cross_entropy(click_pred, click_label)
        conversion_loss = nn.functional.binary_cross_entropy(
            conversion_pred, conversion_label)

        label_constraint = torch.maximum(conversion_pred - click_pred,
                                         torch.zeros_like(click_label))
        constraint_loss = torch.sum(label_constraint)

        loss = click_loss + conversion_loss + constraint_weight * constraint_loss
        return loss