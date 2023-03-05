# -*- ecoding: utf-8 -*-
# @ModuleName: base_model
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/8/16 5:10 PM
import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_
import numpy as np
from .layers import EmbeddingLayer
from loguru import logger

class BaseModel(nn.Module):
    def __init__(self,enc_dict,embedding_dim):
        super(BaseModel, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def set_pretrained_weights(self, col_name, pretrained_dict, trainable=True):
        assert col_name in self.enc_dict.keys(),"Pretrained Embedding Col: {} must be in the {}".fotmat(col_name,self.enc_dict.keys())
        pretrained_emb_dim = len(list(pretrained_dict.values())[0])
        assert self.embedding_dim == pretrained_emb_dim,"Pretrained Embedding Dim:{} must be equal to Model Embedding Dim:{}".format(pretrained_emb_dim, self.embedding_dim)
        pretrained_emb = np.random.rand(self.enc_dict[col_name]['vocab_size'], pretrained_emb_dim)
        for k, v in self.enc_dict[col_name].items():
            if k == 'vocab_size':
                continue
            pretrained_emb[v, :] = pretrained_dict.get(k, np.random.rand(pretrained_emb_dim))

        embeddings = torch.from_numpy(pretrained_emb).float()
        embedding_matrix = torch.nn.Parameter(embeddings)
        self.embedding_layer.set_weights(col_name=col_name, embedding_matrix=embedding_matrix, trainable=trainable)
        logger.info('Successfully Set The Pretrained Embedding Weights for the column:{} With Trainable={}'.format(col_name, trainable))

class GraphBaseModel(nn.Module):
    def __int__(self,num_user,num_item,embedding_dim):
        super(GraphBaseModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_user = num_item
        self.num_item = num_item

        self.user_emb_layer = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_emb_layer = nn.Embedding(self.num_item, self.embedding_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]

        return mf_loss + emb_loss
    def get_ego_embedding(self):
        user_emb = self.user_emb_layer.weight
        item_emb = self.item_emb_layer.weight

        return torch.cat([user_emb, item_emb], 0)

class SequenceBaseModel(nn.Module):
    def __init__(self,enc_dict,config):
        super(SequenceBaseModel, self).__init__()

        self.enc_dict = enc_dict
        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.device = self.config['device']

        self.item_emb = nn.Embedding(self.enc_dict[self.config['item_col']]['vocab_size'], self.embedding_dim,padding_idx=0)
        for col in self.config['cate_cols']:
            setattr(self,f'{col}_emb',nn.Embedding(self.enc_dict[col]['vocab_size'], self.embedding_dim,padding_idx=0))

        self.loss_fun = nn.CrossEntropyLoss()

    def calculate_loss(self, user_emb, pos_item):
        all_items = self.output_items()
        scores = torch.matmul(user_emb, all_items.transpose(1, 0))
        loss = self.loss_fun(scores, pos_item)
        return loss

    def output_items(self):
        self.eval()
        return self.item_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)