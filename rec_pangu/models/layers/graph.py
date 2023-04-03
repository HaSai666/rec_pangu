# -*- ecoding: utf-8 -*-
# @ModuleName: graph
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import dgl.function as fn


class FiGNN_Layer(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 gnn_layers=3,
                 reuse_graph_layer=False,
                 use_gru=True,
                 use_residual=True,
                 device=None):
        super(FiGNN_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        self.device = device
        if reuse_graph_layer:
            self.gnn = GraphLayer(num_fields, embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(num_fields, embedding_dim)
                                      for _ in range(gnn_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embedding_dim * 2, 1, bias=False)

    def build_graph_with_attention(self, feature_emb):
        src_emb = feature_emb[:, self.src_nodes, :]
        dst_emb = feature_emb[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        mask = torch.eye(self.num_fields).to(self.device)
        alpha = alpha.masked_fill(mask.byte(), float('-inf'))
        graph = F.softmax(alpha, dim=-1)  # batch x field x field without self-loops
        return graph

    def forward(self, feature_emb):
        g = self.build_graph_with_attention(feature_emb)
        h = feature_emb
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feature_emb
        return h


class GraphLayer(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)  # broadcast multiply
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class NGCFLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout

        self.in_size = in_size
        self.out_size = out_size

        # weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias=False)
        self.W2 = nn.Linear(in_size, out_size, bias=False)

        # leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def message_fun(self, edges):
        edge_feature = self.W1(edges.src['h']) + self.W2(edges.src['h'] * edges.dst['h'])
        edge_feature = edge_feature * (edges.src['norm'] * edges.dst['norm'])
        return {'e': edge_feature}

    def forward(self, g, ego_embedding):
        g.ndata['h'] = ego_embedding
        g.update_all(message_func=self.message_fun, reduce_func=fn.sum('e', 'h_N'))

        g.ndata['h_N'] = g.ndata['h_N'] + self.W1(g.ndata['h'])

        h = self.leaky_relu(g.ndata['h_N'])  # leaky relu
        h = self.dropout(h)  # dropout
        h = F.normalize(h, dim=1, p=2)  # l2 normalize

        return h


class SRGNNConv(nn.Module):
    """
    只是一个图，实现公式(1)
    """

    def __init__(self, dim):
        super(SRGNNConv, self).__init__()
        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, g, ego_embedding):
        hidden = self.lin(ego_embedding)
        g.ndata['h'] = hidden

        g.update_all(message_func=fn.u_mul_e('h', 'edge_weight', 'm'),
                     reduce_func=fn.sum(msg="m", out="h"))
        return g.ndata['h']


class SRGNNCell(nn.Module):
    """
    实现公式(2)(3)(4)(5)
    """

    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)  # 3个W a
        self.lin_hh = nn.Linear(dim, 3 * dim)  # 3个U v

    def forward(self, in_graph, out_graph, hidden):
        # 图相关
        input_in = self.incomming_conv(in_graph, hidden)
        input_out = self.outcomming_conv(out_graph, hidden)
        # 将两个图的结果进行拼接
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)  # gi: 3个W*a
        gh = self.lin_hh(hidden)  # gh: 3个U*v
        i_r, i_i, i_n = gi.chunk(3, -1)  # 分裂成3个W*a
        h_r, h_i, h_n = gh.chunk(3, -1)  # 分裂成3个U*v
        reset_gate = torch.sigmoid(i_r + h_r)  # 公式(2)
        input_gate = torch.sigmoid(i_i + h_i)  # 公式(3)
        new_gate = torch.tanh(i_n + reset_gate * h_n)  # 公式(4)
        hy = (1 - input_gate) * hidden + input_gate * new_gate  # 公式(5)
        return hy
