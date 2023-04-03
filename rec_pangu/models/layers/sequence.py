# -*- ecoding: utf-8 -*-
# @ModuleName: sequence
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM

from torch import nn
import torch


class MaskedAveragePooling(nn.Module):
    """
    This module takes as input an embedding matrix,
    applies masked pooling, i.e. ignores zero-padding,
    and computes the average embedding vector for each input.
    """
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes the average embedding vector.

        Args:
            embedding_matrix (torch.Tensor): Input embedding of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size) representing the averaged embedding vector.
        """
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix != 0).sum(dim=1)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return embedding_vec


class MaskedSumPooling(nn.Module):
    """
    This module takes as input an embedding matrix,
    applies masked pooling, i.e. ignores zero-padding,
    and computes the sum embedding vector for each input.
    """
    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes the sum embedding vector.

        Args:
            embedding_matrix (torch.Tensor): Input embedding of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size) representing the summed embedding vector.
        """
        # mask by zeros
        return torch.sum(embedding_matrix, dim=1)


class KMaxPooling(nn.Module):
    """
    This module takes as input an embedding matrix,
    and returns the k-max pooling along the specified axis.
    """
    def __init__(self, k: int, dim: int):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the k-max pooling.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, k, hidden_size) representing the k-max pooled embedding vector.
        """
        index = X.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
        output = X.gather(self.dim, index)
        return output


class STAMPLayer(nn.Module):
    def __init__(self, embedding_dim: int, feat_drop: float = 0.0):
        """
        Args:
        embedding_dim(int): the input/output dimensions of the STAMPLayer
        feat_drop(float): Dropout rate to be applied to the input features
        """
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_a = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.fc_t = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.attn_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_t = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.attn_s = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_e = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, emb_seqs: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """
        Applies the STAMP mechanism to a batch of input sequences

        Args:
        emb_seqs(torch.Tensor): Batch of input sequences [batch_size, max_len, embedding_dim]
        lens(torch.Tensor): A tensor of actual sequence lengths for each sequence in the batch [batch_size]

        Returns:
        sr(torch.Tensor): Output scores of the STAMP mechanism [batch_size, embedding_dim]
        """
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)
        batch_size, max_len, _ = emb_seqs.size()

        # mask out padded inputs
        mask = torch.arange(
            max_len, device=lens.device
        ).unsqueeze(0).expand(batch_size, max_len) >= lens.unsqueeze(-1)
        emb_seqs = torch.masked_fill(emb_seqs, mask.unsqueeze(-1), 0)

        # calculate mean sequence vector
        ms = emb_seqs.sum(dim=1) / lens.unsqueeze(-1)  # [batch_size, embedding_dim]

        # calculate target vector and attention weights
        xt = emb_seqs[torch.arange(batch_size), lens - 1]  # [batch_size, embedding_dim]
        ei = self.attn_i(emb_seqs)  # [batch_size, max_len, embedding_dim]
        et = self.attn_t(xt).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        es = self.attn_s(ms).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        e = self.attn_e(torch.sigmoid(ei + et + es)).squeeze(-1)  # [batch_size, max_len]
        alpha = torch.masked_fill(e, mask, 0)
        alpha = alpha.unsqueeze(-1)  # [batch_size, max_len, 1]
        ma = torch.sum(alpha * emb_seqs, dim=1)  # [batch_size, embedding_dim]

        # calculate final output scores
        ha = self.fc_a(ma)
        ht = self.fc_t(xt)
        sr = ha * ht  # [batch_size, embedding_dim]

        return sr
