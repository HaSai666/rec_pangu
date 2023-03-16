# -*- ecoding: utf-8 -*-
# @ModuleName: utils
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM

import dgl
import os
import numpy as np
import torch
from torch import nn
import random

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def set_optimizer(optimizer):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
        return getattr(torch.optim, optimizer)

def set_loss(loss):
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
        else:
            raise NotImplementedError("loss={} is not supported.".format(loss))
    return loss

def set_regularizer(reg):
    reg_pair = [] # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.append((1, float(l1_reg)))
                reg_pair.append((2, float(l2_reg)))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError("regularizer={} is not supported.".format(reg))
    return reg_pair

def set_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation

def get_linear_input(enc_dict,data):
    res_data = []
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            res_data.append(data[col])
    res_data = torch.stack(res_data,axis=1)
    return res_data

def get_dnn_input_dim(enc_dict,embedding_dim):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense+=1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse+=1
    return num_sparse*embedding_dim+num_dense

def get_feature_num(enc_dict):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense+=1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse+=1
    return num_sparse,num_dense


def pad_sequence(seqs, max_len):
    padded_seqs = []
    for seq in seqs:
        seq_len = seq.shape[0]
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq[:max_len]
        padded_seqs.append(padded_seq)
    padded_seqs = torch.stack(padded_seqs, dim=0)
    return padded_seqs

def generate_graph(batch_data):
    '''
    构建Session Graph
    每一条序列都是一个独立的图，为了把一个batch的数据放在一起建模，我们对这个batch每一条序列在图中index必须独立
    '''
    x = []
    edge_index = []
    alias_inputs = []
    item_seq_len = torch.sum(batch_data['hist_mask_list'], dim=-1).cpu().numpy()
    # 对每个session graph进行构建
    """
    # 小例子
    seq = torch.tensor([22,23,21,22,23,23,24,25,21])
    map_index, idx = torch.unique(seq, return_inverse=True)

    print(map_index,idx)
    (tensor([21, 22, 23, 24, 25]), tensor([1, 2, 0, 1, 2, 2, 3, 4, 0]))

    #通过以下方式过去图的emb
    g = item_emb(map_index)

    """
    for i, seq in enumerate((list(torch.chunk(batch_data['hist_item_list'], batch_data['hist_item_list'].shape[0])))):
        seq = seq[seq > 0]
        seq, idx = torch.unique(seq, return_inverse=True)
        x.append(seq)
        alias_seq = idx.squeeze(0)
        alias_inputs.append(alias_seq)
        # No repeat click
        edge = torch.stack([alias_seq[:-1], alias_seq[1:]])
        edge_index.append(edge)
    """
    对一个batch内的所有session graph进行防冲突处理
    核心逻辑给每个序列的index加上前一个序列的index的最大值，
    保证每个序列在图中对应的节点的index范围互不冲突
    """
    tot_node_num = torch.zeros([1], dtype=torch.long)
    for i in range(batch_data['hist_item_list'].shape[0]):
        edge_index[i] = edge_index[i] + tot_node_num
        alias_inputs[i] = alias_inputs[i] + tot_node_num
        tot_node_num += x[i].shape[0]

    x = torch.cat(x)
    alias_inputs = pad_sequence(alias_inputs, max_len=batch_data['hist_item_list'].shape[1])

    # SRGNN有两个图，第二个图可以简单通过torch.flip进行构建
    edge_index = torch.cat(edge_index, dim=1)
    reversed_edge_index = torch.flip(edge_index, dims=[0])

    in_graph = dgl.graph((edge_index[0], edge_index[1]))
    src_degree = in_graph.out_degrees().float()
    norm = torch.pow(src_degree, -1).unsqueeze(1)  # 节点力度的norm
    edge_weight = norm[edge_index[0]]  # 边粒度的norm
    in_graph.edata['edge_weight'] = edge_weight  # 边粒度norm赋值

    out_graph = dgl.graph((reversed_edge_index[0], reversed_edge_index[1]))
    src_degree = out_graph.out_degrees().float()
    norm = torch.pow(src_degree, -1).unsqueeze(1)
    edge_weight = norm[reversed_edge_index[0]]
    out_graph.edata['edge_weight'] = edge_weight

    new_batch_data = {
        'x': x,
        'alias_inputs': alias_inputs,
        'in_graph': in_graph,
        'out_graph': out_graph

    }
    return new_batch_data