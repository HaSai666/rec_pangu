from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class LGConv(MessagePassing):
    def __init__(self, normalize=True, **kwargs):
        kwargs.setdefault('aggr', 'add') # 设置聚合信息的方式为求和(add)
        super().__init__()
        self.normalize = normalize

    def forward(self, x, edge_index, edge_weight=None):
        """前向传播，聚合邻居的信息"""
        if self.normalize:
            out = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                           add_self_loops=False, dtype=x.dtype) # LightGCN中不需要对图加一个自环。
            edge_index, edge_weight = out

        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

    def message(self, x_j, edge_weight):
        """聚合信息的时候，怎么加权"""
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j