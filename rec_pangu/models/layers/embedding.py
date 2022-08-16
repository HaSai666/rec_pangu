# -*- ecoding: utf-8 -*-
# @ModuleName: embedding
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM

from torch import nn
import torch

class EmbeddingLayer(nn.Module):
    def __init__(self,
                 enc_dict = None,
                 embedding_dim = None):
        super(EmbeddingLayer, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()

        self.emb_feature = []

        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col : nn.Embedding(
                    self.enc_dict[col]['vocab_size']+1,
                    self.embedding_dim,
                )})

    def set_weights(self, col_name, embedding_matrix, trainable=True):
        self.embedding_layer[col_name].weight = embedding_matrix
        if not trainable:
            self.embedding_layer[col_name].weight.requires_grad = False

    def forward(self, X,name=None):
        if name == None:
            feature_emb_list = []
            for col in self.emb_feature:
                inp = X[col].long().view(-1, 1)
                feature_emb_list.append(self.embedding_layer[col](inp))
            return torch.stack(feature_emb_list,dim=1).squeeze(2)
        else:
            if 'seq' in name:
                inp = X[name].long()
                fea = self.embedding_layer[name.replace('_seq','')](inp)
            else:
                inp = X[name].long().view(-1, 1)
                fea = self.embedding_layer[name](inp)
            return fea