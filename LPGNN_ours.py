import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.parameter import Parameter

from torch.nn.modules.module import Module
import math
class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=F.relu, dropout=0):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        if self.activation is F.relu:
            gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        #feats = self.dropout(feats)
        feats = self.linear(feats)
        # if self.activation:
        #     feats = self.activation(feats)

        return feats

class LPLayer(nn.Module):
    def __init__(self, in_features, out_features,out_size, dropout=0.5, alpha=0.2, concat=False):
        super(LPLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.out_size=out_size

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.W2 = nn.Parameter(torch.empty(size=(out_size, out_features)))
        # nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        # self.a = nn.Parameter(torch.empty(size=(2*out_size, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, label):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)

        # 计算稀疏注意力值
        e, indices = self._prepare_attentional_mechanism_input(label, adj)

        # 对注意力值进行 softmax
        attention = sparse_softmax(indices, e, adj.shape)

        # 对注意力值应用 dropout
        attention_values = F.dropout(attention._values(), self.dropout, training=self.training)

        # 构造应用 dropout 后的稀疏矩阵
        attention = torch.sparse_coo_tensor(attention._indices(), attention_values, adj.shape)

        # 使用稀疏矩阵乘法计算新的特征
        h_prime = torch.sparse.mm(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, label, adj):
        """
        计算稀疏注意力矩阵的输入。
        只计算边相连的节点的注意力值。
        """
        # 获取邻接矩阵的非零索引
        indices = adj._indices()  # shape: (2, num_edges)

        # 取出边的两端节点的特征
        src_features = label[indices[0]]  # 边的起点特征
        dst_features = label[indices[1]]  # 边的终点特征

        # 计算注意力值（例如点积）
        edge_attention = torch.sum(src_features * dst_features, dim=1)  # shape: (num_edges,)

        # 应用 LeakyReLU 激活
        edge_attention = self.leakyrelu(edge_attention)

        return edge_attention, indices

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class Ours(nn.Module):
    def __init__(self, in_size, hid_size, out_size,is_res,args):
        super().__init__()
        self.MLPLayers=nn.ModuleList()
        self.MLPLayers.append(
            MLPLayer(in_size, out_size)
        )
        # self.MLPLayers.append(
        #     MLPLayer(hid_size,out_size)
        # )

        self.args=args
        self.LPlayers = nn.ModuleList()
        # two-layer GAT
        self.LPlayers.append(
            LPLayer(in_size,out_size,out_size)
        )
        
        # self.LPlayers.append(
        #     LPLayer(hid_size,out_size,out_size)
        # )
        self.strucLayers=nn.ModuleList()
        self.strucLayers.append(
            GraphConvolution(in_size,hid_size)
        )
        
        self.strucLayers.append(
            GraphConvolution(hid_size,out_size)
        )

        #self.weight4 = nn.Parameter(torch.rand(out_size, out_size))
        self.weight5 = nn.Parameter(torch.rand(out_size, out_size))
        # self.weight7 = nn.Parameter(torch.rand(out_size, out_size))
        self.weight9=nn.Parameter(torch.rand(out_size*2, out_size))
        #self.weight10 = torch.ones(out_size*2).to('cuda:1')
        self.is_res=is_res
        #self.train_mask=train_mask
        #self.truelabels=truelabels
        
        self.adaptive_weight = nn.Parameter(torch.ones(2))
    def forward(self, inputs, adj, weightAdj,featureAdj):
        h = inputs
        label = inputs
        h2 = inputs

        for i, layer in enumerate(self.MLPLayers):
            label = layer(label)

        for i, layer in enumerate(self.LPlayers):
            h = layer(h, adj, label)

        
        for i, layer in enumerate(self.strucLayers):
            h2 = layer(h2, weightAdj)

            # if i != len(self.strucLayers) - 1:
            #     h2 = F.relu(h2)
            #     h2 = F.dropout(h2, 0.2, training=self.training)
        # for i, layer in enumerate(self.gcnlayers):
        #     h2 = layer(h2, adj)
        #self.args.compensate=False
        
        if self.args.compensate:
            # weights = F.softmax(self.adaptive_weight, dim=0)
            # # 加权融合
            # h = weights[0] * h + weights[1] * h2
            h=h+h2
        else:
            h = h2

        return h, label

    def processPlabel(self, pseudo_labels, label, train_mask):
        result = pseudo_labels.clone()

        # 从train_mask中为True的索引随机选取一定比例（例如50%）
        num_trues = train_mask.sum().item()
        num_to_select = int(num_trues * 0.5)
        indices = torch.where(train_mask)[0]
        selected_indices = indices[torch.randperm(len(indices))[:num_to_select]]

        # 使用选中的索引替换伪标签张量中的对应行
        result[selected_indices] = label[selected_indices]
        return result

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self,in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input,adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def sparse_softmax(indices, values, size):
    """
    对稀疏矩阵的邻居注意力值进行 softmax 操作。
    :param indices: 稀疏矩阵的非零索引，形状为 (2, num_edges)。
    :param values: 稀疏矩阵的非零值，形状为 (num_edges,)。
    :param size: 稀疏矩阵的形状 (num_nodes, num_nodes)。
    :return: 归一化后的稀疏矩阵。
    """
    # 获取每个节点的邻居注意力值的总和（按行分组）
    row_sum = torch.zeros(size[0], device=values.device).scatter_add(0, indices[0], values.exp())

    # 避免除以零
    row_sum = torch.clamp(row_sum, min=1e-9)

    # 对每个节点的邻居注意力值进行归一化
    normalized_values = values.exp() / row_sum[indices[0]]

    # 构造归一化后的稀疏矩阵
    normalized_attention = torch.sparse_coo_tensor(indices, normalized_values, size)

    return normalized_attention