# -*- ecoding: utf-8 -*-
# @ModuleName: sequence
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


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


""" Encoder Layers """


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention.
        """
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same

        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

    def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        return x.view(*new_x_shape).transpose(-2, -3)

    def forward(self, q, k, v, mask=None):
        origin_shape = q.size()

        # perform linear operation and split into h heads
        if not self.kq_same:
            q = self.head_split(self.q_linear(q))
        else:
            q = self.head_split(self.k_linear(q))
        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
        output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3).reshape(origin_shape)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, d_k, mask=None):
        """
        This is called by Multi-head attention object to find the values.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        scores = (scores - scores.max()).softmax(dim=-1)
        scores = scores.masked_fill(torch.isnan(scores), 0)
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
        super().__init__()
        """
        This is a Basic Block of Transformer. It contains one Multi-head attention object. 
        Followed by layer norm and position wise feedforward net and dropout layer.
        """
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
        context = self.masked_attn_head(seq, seq, seq, mask)
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output


class GRU4RecEncoder(nn.Module):
    def __init__(self, emb_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, emb_size, bias=False)

    def forward(self, seq, lengths):
        # Sort and Pack
        sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_seq = seq.index_select(dim=0, index=sort_idx)
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths.cpu(), batch_first=True)

        # RNN
        output, hidden = self.rnn(seq_packed, None)

        # Unsort
        sort_rnn_vector = self.out(hidden[-1])
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)

        return rnn_vector


class CaserEncoder(nn.Module):
    def __init__(self, emb_size, max_his, num_horizon=16, num_vertical=8, l=5):
        super().__init__()
        self.max_his = max_his
        lengths = [i + 1 for i in range(l)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=num_horizon, kernel_size=(i, emb_size)) for i in lengths])
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=num_vertical, kernel_size=(max_his, 1))
        self.fc_dim_h = num_horizon * len(lengths)
        self.fc_dim_v = num_vertical * emb_size
        fc_dim_in = self.fc_dim_v + self.fc_dim_h
        self.fc = nn.Linear(fc_dim_in, emb_size)

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        pad_len = self.max_his - seq_len
        seq = F.pad(seq, [0, 0, 0, pad_len]).unsqueeze(1)

        # Convolution Layers
        out_v = self.conv_v(seq).view(-1, self.fc_dim_v)
        out_hs = list()
        for conv in self.conv_h:
            conv_out = conv(seq).squeeze(3).relu()
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)

        # Fully-connected Layers
        his_vector = self.fc(torch.cat([out_v, out_h], 1))
        return his_vector


class BERT4RecEncoder(nn.Module):
    def __init__(self, emb_size, max_his, num_layers=2, num_heads=2):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(seq.device)
        valid_mask = len_range[None, :] < lengths[:, None]

        # Position embedding
        position = len_range[None, :] * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors

        # Self-attention
        attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()

        his_vector = seq[torch.arange(batch_size), lengths - 1]
        return his_vector
