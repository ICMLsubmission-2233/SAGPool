import random
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv as GraphConv
from torch_geometric.nn.inits import uniform
from torch_geometric.utils.num_nodes import maybe_num_nodes


def topk(x, ratio, batch):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    x_min_value = x.min().item()
    dense_x = x.new_full((batch_size * max_num_nodes, ), x_min_value - 1)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)
    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + i *
        max_num_nodes for i in range(len(num_nodes))
    ]
    mask = torch.cat(mask, dim=0)

    perm = perm[mask]

    return perm


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class SAGPool(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Net"
    <https://openreview.net/forum?id=HJePRoAct7>`_ and `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_ papers

    .. math::
        \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}

        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            (default: :obj:`0.5`)
    """

    def __init__(self, in_channels, ratio=0.5):
        super(SAGPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio

        self.conv_1 = GraphConv(in_channels,1)
        self.conv_2 = GraphConv(in_channels,1)
        self.conv_3 = GraphConv(in_channels,1)
        self.conv_4 = GraphConv(in_channels,1)
        self.add_module('attention_layer',self.conv_1)
        self.add_module('attention_layer',self.conv_2)
        self.add_module('attention_layer',self.conv_3)
        self.add_module('attention_layer',self.conv_4)
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = (torch.tanh(self.conv_1(x,edge_index)).view(-1) + torch.tanh(self.conv_2(x,edge_index)).view(-1)+torch.tanh(self.conv_3(x,edge_index)).view(-1) + torch.tanh(self.conv_4(x,edge_index)).view(-1))/4.

        perm = topk(score, self.ratio, batch)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__,
                                         self.in_channels, self.ratio)