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
