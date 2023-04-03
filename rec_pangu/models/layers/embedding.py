# -*- ecoding: utf-8 -*-
# @ModuleName: embedding
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import Dict, Union, Optional
import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 enc_dict: Dict[str, Dict[str, Union[int, str]]],
                 embedding_dim: int) -> None:
        """
        Initialize EmbeddingLayer instance.
        Args:
            enc_dict: Encoding dictionary containing vocabulary size for each categorical feature
            embedding_dim: Number of dimensions in the embedding space
        """
        super().__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()
        self.emb_feature = []

        # Loop through all columns and create nn.Embedding layer for columns with 'vocab_size' key in dictionary
        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col: nn.Embedding(
                    num_embeddings=self.enc_dict[col]['vocab_size'] + 1,
                    embedding_dim=self.embedding_dim,
                )})

    def set_weights(self, col_name: str, embedding_matrix: torch.Tensor,
                    trainable: Optional[bool] = True) -> None:
        """
        Set the weights for the embedding layer.
        Args:
            col_name: Column name
            embedding_matrix: Embedding weight tensor for the column name
            trainable: Boolean indicating if the embedding layer should be trained
        """
        self.embedding_layer[col_name].weight = nn.Parameter(embedding_matrix)
        if not trainable:
            self.embedding_layer[col_name].weight.requires_grad = False

    def forward(self, X: Dict[str, torch.Tensor], name: Optional[str] = None) -> torch.Tensor:
        """
        Compute the embeddings for a batch of input tensors.
        Args:
            X: Input tensor of shape [batch_size,feature_dim] where feature_dim is the number of features
            name: String indicating the column name
        Returns:
            feature_emb_list: Tensor of shape [batch_size, num_embeddings] containing embeddings for each input feature.
        """
        if name is None:
            feature_emb_list = []
            for col in self.emb_feature:
                inp = X[col].long().view(-1, 1)
                feature_emb_list.append(self.embedding_layer[col](inp))
            return torch.stack(feature_emb_list, dim=1).squeeze(2)
        else:
            if 'seq' in name:
                inp = X[name].long()
                fea = self.embedding_layer[name.replace('_seq', '')](inp)
            else:
                inp = X[name].long().view(-1, 1)
                fea = self.embedding_layer[name](inp)
            return fea
