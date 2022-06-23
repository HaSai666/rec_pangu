import torch
import pandas as pd

def to_undirected(edge_index):
    edge_index_rev = torch.stack([edge_index[1], edge_index[0]]) # 反向边
    edge_index_sym = torch.cat([edge_index, edge_index_rev], dim=1)
    return edge_index_sym

def read_graph(graph_path,train_rate):

    df = pd.read_csv(graph_path)

    # itenID不是连续的，这里我们重新映射一下，使序号变得连续 (后面会用到)并且从零开始
    item_mapping = {idx: i for i, idx in enumerate(df['movieId'].unique())}

    # 在转换成图之前，我们把用户的ID也映射（重新排序）一下，使其连续且从零开始
    user_mapping = {idx: i for i, idx in enumerate(df['userId'].unique())}

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    user_src = [user_mapping[idx] for idx in df['userId']] # 起始节点
    item_dst = [item_mapping[idx]+num_users for idx in df['movieId']] # 终止节点
    edge_index = torch.tensor([user_src, item_dst])
    rating = torch.from_numpy(df['rating'].values).to(torch.long)

    num_nodes = edge_index.max().item() + 1
    _N = len(rating)
    indicies_perm = torch.randperm(_N)


    idx_train = indicies_perm[: int(train_rate*_N)]
    train_edge_index = edge_index[:, idx_train]
    train_edge_label = rating[idx_train]


    idx_test = indicies_perm[int(train_rate*_N): ]
    test_edge_index = edge_index[:, idx_test]
    test_edge_label = rating[idx_test]

    train_graph_edge_index = to_undirected(train_edge_index)
    test_graph_edge_index = to_undirected(test_edge_index)

    train_data = dict()
    train_data['train_edge_index'] = train_edge_index
    train_data['train_graph_edge_index'] = train_graph_edge_index
    train_data['train_edge_label'] = train_edge_label

    test_data = dict()
    test_data['test_edge_index'] = test_edge_index
    test_data['test_graph_edge_index'] = test_graph_edge_index
    test_data['test_edge_label'] = test_edge_label

    graph_information = dict()
    graph_information['num_nodes'] = num_nodes

    return train_data, test_data, graph_information